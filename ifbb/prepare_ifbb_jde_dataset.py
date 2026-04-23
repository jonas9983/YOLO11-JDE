import os
import sqlite3
import torch
import shutil
import argparse
import zipfile
import cv2
from pathlib import Path
from tqdm import tqdm
import numpy as np
from ultralytics import YOLO

class IFBBJDEDatasetBuilder:
    def __init__(self, source_root, output_dir, device="cpu"):
        self.source_root = Path(source_root)
        self.output_dir = Path(output_dir)
        self.db_path = self.source_root / "npc_database.db"
        self.tracking_db_path = self.output_dir.parent / "dataset_builder.db"
        self.device = device
        self.temp_dir = Path("/tmp/ifbb_jde_unzip")
        self.detector = YOLO("yolo11n.pt").to(self.device)
        
        self.id_map = {}
        self.next_id = 0
        self._init_tracking_db()
        self._load_id_map()

    def _init_tracking_db(self):
        self.tracking_db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.tracking_db_path)
        cursor = conn.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS processed_contests (year INTEGER, contest_name TEXT, UNIQUE(year, contest_name))')
        cursor.execute('CREATE TABLE IF NOT EXISTS id_mapping (athlete_name TEXT PRIMARY KEY, athlete_id INTEGER)')
        conn.commit(); conn.close()

    def _load_id_map(self):
        conn = sqlite3.connect(self.tracking_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT athlete_name, athlete_id FROM id_mapping")
        for row in cursor.fetchall():
            self.id_map[row[0]] = row[1]
            if row[1] >= self.next_id: self.next_id = row[1] + 1
        conn.close()

    def _get_id(self, athlete_name):
        athlete_name = athlete_name.strip().upper()
        if athlete_name not in self.id_map:
            self.id_map[athlete_name] = self.next_id
            conn = sqlite3.connect(self.tracking_db_path)
            conn.execute("INSERT OR IGNORE INTO id_mapping VALUES (?, ?)", (athlete_name, self.next_id))
            conn.commit(); conn.close()
            self.next_id += 1
        return self.id_map[athlete_name]

    def is_contest_processed(self, year, contest_name):
        conn = sqlite3.connect(self.tracking_db_path)
        res = conn.execute("SELECT 1 FROM processed_contests WHERE year=? AND contest_name=?", (year, contest_name)).fetchone()
        conn.close()
        return res is not None

    def process_contest(self, zip_path, year, contest_name):
        # Setup clean local workspace
        if self.temp_dir.exists(): shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Unzip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.temp_dir)
        
        # Get DB mapping
        conn = sqlite3.connect(self.db_path)
        file_to_athlete = {row[0]: row[1] for row in conn.execute("SELECT image_filename, athlete_name FROM athletes WHERE year=? AND contest_name=?", (year, contest_name)).fetchall()}
        conn.close()

        images = [img for img in self.temp_dir.glob("**/*") if img.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        if not images: return True

        # Prepare batching
        batch_size = 256
        
        # Process and save to a local temporary folder first
        contest_out_dir = Path(f"/tmp/contest_output/{year}_{contest_name.replace(' ', '_')}")
        contest_out_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            results = self.detector([str(p) for p in batch], verbose=False, device=self.device)
            
            for j, res in enumerate(results):
                athlete_name = file_to_athlete.get(batch[j].name)
                if not athlete_name or "COMPARISON" in athlete_name.upper() or "AWARD" in athlete_name.upper(): continue
                
                boxes = [b for b in res.boxes if int(b.cls) == 0]
                if not boxes: continue
                
                best = sorted(boxes, key=lambda x: x.conf, reverse=True)[0]
                xywh = best.xywhn[0].cpu().numpy()
                
                # Clean Filename: {athlete_id}_{filename}
                athlete_id = self._get_id(athlete_name)
                save_name = f"{athlete_id}_{batch[j].name}"
                
                shutil.copy(batch[j], contest_out_dir / save_name)
                with open(contest_out_dir / f"{Path(save_name).stem}.txt", "w") as f:
                    f.write(f"0 {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f} {athlete_id}")
        
        # Zip contest and move to final Drive output
        final_zip = self.output_dir / f"{year}_{contest_name.replace(' ', '_')}.zip"
        shutil.make_archive(str(final_zip.with_suffix('')), 'zip', contest_out_dir)
        shutil.rmtree(contest_out_dir)
        shutil.rmtree(self.temp_dir)
        return True

    def run(self, years):
        conn = sqlite3.connect(self.db_path)
        contests = conn.execute("SELECT DISTINCT year, contest_name FROM athletes").fetchall()
        conn.close()
        
        for year, contest in contests:
            if str(year) not in years or self.is_contest_processed(year, contest): continue
            zip_path = self.source_root / str(year) / f"{year}_{contest.replace(' ', '_')}.zip"
            if zip_path.exists():
                if self.process_contest(zip_path, year, contest):
                    conn = sqlite3.connect(self.tracking_db_path)
                    conn.execute("INSERT INTO processed_contests VALUES (?, ?)", (year, contest))
                    conn.commit(); conn.close()
                    print(f"Finished {contest}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--years", default="2024,2025,2026")
    args = parser.parse_args()
    IFBBJDEDatasetBuilder(args.source, args.output).run(args.years.split(','))
