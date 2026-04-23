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
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.source_root / "npc_database.db"
        
        # Keep tracking DB locally to prevent Google Drive FUSE crashes
        self.local_tracking_db = Path("/content/dataset_builder.db")
        self.drive_tracking_db = self.output_dir.parent / "dataset_builder.db"
        
        if self.drive_tracking_db.exists() and not self.local_tracking_db.exists():
            shutil.copy(self.drive_tracking_db, self.local_tracking_db)
            
        self.tracking_db_path = self.local_tracking_db
        self.device = device
        self.temp_dir = Path("/tmp/ifbb_jde_unzip")
        self.detector = YOLO("yolo11n.pt").to(self.device)
        
        print(f"\n--- IFBB JDE DATASET BUILDER ---")
        print(f"Source Root: {self.source_root.absolute()}")
        print(f"Output Dataset: {self.output_dir.absolute()}")
        print(f"Tracking DB (Local): {self.tracking_db_path.absolute()}")
        print(f"Device: {self.device.upper()}")
        print(f"--------------------------------\n")
        
        self.id_map = {}
        self.next_id = 0
        self._init_tracking_db()
        self._load_id_map()

    def _sync_db_to_drive(self):
        """Safely copy the local DB to Drive to backup progress."""
        try:
            shutil.copy(self.local_tracking_db, self.drive_tracking_db)
        except Exception as e:
            print(f"  [WARNING] Failed to sync DB to Drive: {e}")

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
        if self.temp_dir.exists(): shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy to local temp storage to prevent Drive IO crashes
        local_zip_path = Path("/tmp/current_contest.zip")
        try:
            print(f"  [COPY] Copying {zip_path.name} from Drive to Colab local storage...")
            shutil.copy(zip_path, local_zip_path)
            
            with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.temp_dir)
            local_zip_path.unlink() # Cleanup
        except Exception as e:
            print(f"  [ERROR] Failed to extract {zip_path.name}: {e}")
            return False
        
        conn = sqlite3.connect(self.db_path)
        file_to_athlete = {row[0]: row[1] for row in conn.execute("SELECT image_filename, athlete_name FROM athletes WHERE year=? AND contest_name=?", (year, contest_name)).fetchall()}
        conn.close()

        images = [img for img in self.temp_dir.glob("**/*") if img.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        if not images: return True

        batch_size = 256
        contest_out_dir = Path(f"/tmp/contest_output/{year}_{contest_name.replace(' ', '_')}")
        img_dir = contest_out_dir / "images"
        lbl_dir = contest_out_dir / "labels"
        vis_dir = contest_out_dir / "visual_check"
        
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        processed_imgs = 0
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            results = self.detector([str(p) for p in batch], verbose=False, device=self.device)
            
            for j, res in enumerate(results):
                athlete_name = file_to_athlete.get(batch[j].name)
                if not athlete_name or "COMPARISON" in athlete_name.upper() or "AWARD" in athlete_name.upper() or "OVERALL" in athlete_name.upper() or "GUEST POSING" in athlete_name.upper(): 
                    continue
                
                boxes = [b for b in res.boxes if int(b.cls) == 0]
                if not boxes: continue
                
                best = sorted(boxes, key=lambda x: x.conf, reverse=True)[0]
                xywh = best.xywhn[0].cpu().numpy()
                
                athlete_id = self._get_id(athlete_name)
                save_name = f"{athlete_id}_{batch[j].name}"
                
                shutil.copy(batch[j], img_dir / save_name)
                with open(lbl_dir / f"{Path(save_name).stem}.txt", "w") as f:
                    f.write(f"0 {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f} {athlete_id}")
                
                # Save visual check for the first 10 images per contest to verify crops
                if processed_imgs < 10:
                    img_cv = cv2.imread(str(batch[j]))
                    h, w, _ = img_cv.shape
                    cx, cy, bw, bh = xywh
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_cv, f"ID: {athlete_id}", (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.imwrite(str(vis_dir / save_name), img_cv)

                processed_imgs += 1
        
        final_zip = self.output_dir / f"{year}_{contest_name.replace(' ', '_')}.zip"
        shutil.make_archive(str(final_zip.with_suffix('')), 'zip', contest_out_dir)
        shutil.rmtree(contest_out_dir)
        shutil.rmtree(self.temp_dir)
        print(f"  [SUCCESS] Zipped {processed_imgs} valid athlete images to {final_zip.name}")
        return True

    def run(self, years):
        if not self.db_path.exists():
            print(f"[CRITICAL ERROR] Database file not found at: {self.db_path.absolute()}")
            return

        conn = sqlite3.connect(self.db_path)
        contests = conn.execute("SELECT DISTINCT year, contest_name FROM athletes").fetchall()
        conn.close()
        
        print(f"Found {len(contests)} contests in the database.")
        
        processed_contests = 0
        skipped_contests = 0

        for year, contest in tqdm(contests, desc="Processing Contests"):
            if str(year) not in years:
                continue
                
            if self.is_contest_processed(year, contest):
                skipped_contests += 1
                continue
                
            zip_path = self.source_root / str(year) / f"{year}_{contest.replace(' ', '_')}.zip"
            if zip_path.exists():
                tqdm.write(f"\nProcessing NEW contest: {year} {contest}")
                if self.process_contest(zip_path, year, contest):
                    conn = sqlite3.connect(self.tracking_db_path)
                    conn.execute("INSERT INTO processed_contests VALUES (?, ?)", (year, contest))
                    conn.commit(); conn.close()
                    self._sync_db_to_drive()
                    processed_contests += 1
            else:
                tqdm.write(f"\n  [WARNING] Zip not found for {year} {contest} at {zip_path}")
        
        print(f"\n--- COMPLETE ---")
        print(f"Processed: {processed_contests}")
        print(f"Skipped (already done): {skipped_contests}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--years", default="2024,2025,2026")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    IFBBJDEDatasetBuilder(args.source, args.output, device=device).run(args.years.split(','))
