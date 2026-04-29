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

class BodybuildingDatasetBuilder:
    def __init__(self, source_root, db_path, output_dir, device="cpu"):
        self.source_root = Path(source_root)
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Internal ID tracking (so IDs are consistent across different runs)
        base_temp = Path("/content") if Path("/content").exists() else Path("/tmp")
        self.id_mapping_db = self.output_dir / "dataset_id_mapping.db"
        
        self.device = device
        self.detector = YOLO("yolo11n.pt").to(self.device)
        self.temp_unzip_dir = base_temp / "ifbb_temp_unzip"
        
        print(f"\n--- BODYBUILDING JDE DATASET BUILDER ---")
        print(f"Source Root: {self.source_root.absolute()}")
        print(f"Output Dataset: {self.output_dir.absolute()}")
        print(f"Device: {self.device.upper()}")
        print(f"-----------------------------------------\n")
        
        self.id_map = {}
        self.next_id = 0
        self._init_id_mapping_db()
        self._load_id_map()

    def _init_id_mapping_db(self):
        conn = sqlite3.connect(self.id_mapping_db)
        conn.execute('CREATE TABLE IF NOT EXISTS id_mapping (athlete_name TEXT PRIMARY KEY, athlete_id INTEGER)')
        conn.execute('CREATE TABLE IF NOT EXISTS processed_contests (year INTEGER, contest_name TEXT, UNIQUE(year, contest_name))')
        conn.commit()
        conn.close()

    def is_contest_processed(self, year, contest_name):
        conn = sqlite3.connect(self.id_mapping_db)
        res = conn.execute("SELECT 1 FROM processed_contests WHERE year=? AND contest_name=?", (year, contest_name)).fetchone()
        conn.close()
        return res is not None

    def _mark_contest_processed(self, year, contest_name):
        conn = sqlite3.connect(self.id_mapping_db)
        conn.execute("INSERT OR IGNORE INTO processed_contests VALUES (?, ?)", (year, contest_name))
        conn.commit()
        conn.close()

    def _load_id_map(self):
        conn = sqlite3.connect(self.id_mapping_db)
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
            conn = sqlite3.connect(self.id_mapping_db)
            conn.execute("INSERT OR IGNORE INTO id_mapping VALUES (?, ?)", (athlete_name, self.next_id))
            conn.commit(); conn.close()
            self.id_map[athlete_name] = self.next_id
            self.next_id += 1
        return self.id_map[athlete_name]

    def _sanitize(self, name):
        return "".join(c for c in name if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")

    def build(self, target_divisions=["MEN'S BODYBUILDING", "212", "MEN'S CLASSIC PHYSIQUE"]):
        if not self.db_path.exists():
            print(f"ERROR: Database not found at {self.db_path}")
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 1. FIND RELEVANT CONTESTS AND IMAGES
        div_placeholders = ','.join(['?'] * len(target_divisions))
        query = f"SELECT year, contest_name, athlete_name, image_filename FROM athletes WHERE division IN ({div_placeholders})"
        params = list(target_divisions)

        cursor.execute(query, params)
        all_rows = cursor.fetchall()
        
        # Group by contest to minimize unzipping
        contests = {}
        for year, contest, athlete, filename in all_rows:
            key = (year, contest)
            if key not in contests: contests[key] = []
            contests[key].append((athlete, filename))
        
        print(f"Found {len(all_rows)} images across {len(contests)} contests in selected divisions.")
        
        # Create Flat structure
        img_out = self.output_dir / "images"
        lbl_out = self.output_dir / "labels"
        img_out.mkdir(exist_ok=True)
        lbl_out.mkdir(exist_ok=True)
        
        total_saved = 0
        
        # 2. PROCESS EACH CONTEST
        for (year, contest_name), images in tqdm(contests.items(), desc="Processing Contests"):
            if self.is_contest_processed(year, contest_name):
                # tqdm.write(f"  [SKIP] {year} {contest_name} already processed.")
                continue

            zip_name = f"{year}_{self._sanitize(contest_name)}.zip"
            zip_path = self.source_root / str(year) / zip_name
            
            if not zip_path.exists():
                print(f"  [SKIP] Zip not found: {zip_path}")
                continue
            
            # Extract to temp
            if self.temp_unzip_dir.exists(): shutil.rmtree(self.temp_unzip_dir)
            self.temp_unzip_dir.mkdir(parents=True)
            
            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(self.temp_unzip_dir)
            except Exception as e:
                print(f"  [ERROR] Extraction failed for {zip_name}: {e}")
                continue
            
            # Filter and Detect
            file_to_athlete = {f: a for a, f in images}
            available_files = [f for f in self.temp_unzip_dir.glob("**/*") if f.name in file_to_athlete]
            
            if not available_files: 
                self._mark_contest_processed(year, contest_name)
                continue
            
            # Use batch inference for speed
            batch_size = 64
            contest_saved = 0
            for i in range(0, len(available_files), batch_size):
                batch = available_files[i:i+batch_size]
                results = self.detector([str(p) for p in batch], verbose=False, device=self.device)
                
                # Use min() to avoid IndexError if detector returns more results than batch
                for j in range(min(len(batch), len(results))):
                    res = results[j]
                    # We only care about people
                    boxes = [b for b in res.boxes if int(b.cls) == 0]
                    if not boxes: continue
                    
                    # Take the most confident box
                    best = sorted(boxes, key=lambda x: x.conf, reverse=True)[0]
                    xywh = best.xywhn[0].cpu().numpy()
                    
                    orig_file = batch[j]
                    athlete_name = file_to_athlete[orig_file.name]
                    athlete_id = self._get_id(athlete_name)
                    
                    # Final filename: athleteID_originalName
                    save_name = f"{athlete_id}_{orig_file.name}"
                    shutil.copy(orig_file, img_out / save_name)
                    
                    # YOLO label: cls x y w h athlete_id
                    with open(lbl_out / f"{Path(save_name).stem}.txt", "w") as f:
                        f.write(f"0 {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f} {athlete_id}")
                    
                    contest_saved += 1
            
            total_saved += contest_saved
            self._mark_contest_processed(year, contest_name)
        
        # Cleanup
        if self.temp_unzip_dir.exists(): shutil.rmtree(self.temp_unzip_dir)
        
        print(f"\n--- DONE ---")
        print(f"Total images saved to flat dataset: {total_saved}")
        print(f"Unique athletes identified: {len(self.id_map)}")
        
        # Create single ZIP
        print(f"Zipping final dataset...")
        final_zip = self.output_dir.parent / "bodybuilding_jde_dataset.zip"
        shutil.make_archive(str(final_zip.with_suffix('')), 'zip', self.output_dir)
        print(f"MASTER DATASET CREATED: {final_zip.absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to Year folders (2024, 2025, etc.)")
    parser.add_argument("--db", required=True, help="Path to npc_database.db")
    parser.add_argument("--output", required=True, help="Where to save the flat dataset")
    parser.add_argument("--divisions", default="MEN'S BODYBUILDING,212,MEN'S CLASSIC PHYSIQUE", help="Comma-separated divisions to include")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    builder = BodybuildingDatasetBuilder(args.source, args.db, args.output, device=device)
    
    builder.build(target_divisions=args.divisions.split(','))
