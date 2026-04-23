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
    def __init__(self, 
                 source_root, 
                 output_dir,
                 device="cpu"):
        self.source_root = Path(source_root)
        self.output_dir = Path(output_dir)
        self.db_path = self.source_root / "npc_database.db"
        self.tracking_db_path = self.output_dir.parent / "dataset_builder.db"
        self.device = device
        
        # We need a local temp dir for unzipping since Drive is slow for many small operations
        self.temp_dir = Path("/tmp/ifbb_jde_unzip")
        self.debug_vis_dir = self.output_dir / "visual_check"
        
        print(f"\n--- IFBB JDE DATASET BUILDER ---")
        print(f"Source Root: {self.source_root.absolute()}")
        print(f"Output Dataset: {self.output_dir.absolute()}")
        print(f"Tracking DB: {self.tracking_db_path.absolute()}")
        
        self.detector = YOLO("yolo11n.pt").to(self.device)
        
        self._init_tracking_db()
        self._load_id_map()
        self.vis_count = 0

    def _init_tracking_db(self):
        """Initialize the SQLite database to track processed contests."""
        self.tracking_db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.tracking_db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_contests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                year INTEGER,
                contest_name TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(year, contest_name)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS id_mapping (
                athlete_name TEXT PRIMARY KEY,
                athlete_id INTEGER
            )
        ''')
        conn.commit()
        conn.close()

    def _load_id_map(self):
        """Load existing identity map from tracking DB."""
        self.id_map = {}
        self.next_id = 0
        if self.tracking_db_path.exists():
            conn = sqlite3.connect(self.tracking_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT athlete_name, athlete_id FROM id_mapping")
            for row in cursor.fetchall():
                self.id_map[row[0]] = row[1]
                if row[1] >= self.next_id:
                    self.next_id = row[1] + 1
            conn.close()

    def _save_id(self, athlete_name, athlete_id):
        """Save a new identity to tracking DB."""
        conn = sqlite3.connect(self.tracking_db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO id_mapping (athlete_name, athlete_id) VALUES (?, ?)", 
                       (athlete_name, athlete_id))
        conn.commit()
        conn.close()

    def is_contest_processed(self, year, contest_name):
        """Check if a contest has already been processed."""
        conn = sqlite3.connect(self.tracking_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM processed_contests WHERE year = ? AND contest_name = ?", (year, contest_name))
        result = cursor.fetchone()
        conn.close()
        return result is not None

    def mark_contest_processed(self, year, contest_name):
        """Mark a contest as successfully processed."""
        conn = sqlite3.connect(self.tracking_db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO processed_contests (year, contest_name) VALUES (?, ?)", (year, contest_name))
        conn.commit()
        conn.close()

    def _sanitize(self, name):
        """Standard sanitization used during ingestion."""
        return "".join(c for c in name if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")

    def _get_id(self, athlete_name):
        athlete_name = athlete_name.strip().upper()
        if athlete_name not in self.id_map:
            self.id_map[athlete_name] = self.next_id
            self._save_id(athlete_name, self.next_id)
            self.next_id += 1
        return self.id_map[athlete_name]

    def prepare_directories(self):
        for split in ['train', 'val']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.debug_vis_dir.mkdir(parents=True, exist_ok=True)

    def is_group_shot(self, name):
        n = name.upper()
        return "COMPARISON" in n or "AWARD" in n or "OVERALL" in n or "GUEST POSING" in n

    def process_zip(self, zip_path, year, contest_name, split_ratio=0.9):
        print(f"  [ZIP] Extracting {zip_path.name} to fast local temp storage...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.temp_dir)
        except Exception as e:
            print(f"  [ERROR] Failed to unzip {zip_path.name}: {e}")
            return False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT image_filename, athlete_name FROM athletes WHERE year = ? AND contest_name = ?", (year, contest_name))
        file_to_athlete = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()

        images = [img for img in self.temp_dir.glob("**/*") if img.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        print(f"  [INFO] Found {len(images)} images to process.")
        
        if not images:
            shutil.rmtree(self.temp_dir)
            self.temp_dir.mkdir()
            return True # Successfully processed an empty/invalid contest, mark it done

        np.random.shuffle(images)
        split_idx = int(len(images) * split_ratio)
        
        processed_count = 0
        skipped_group_count = 0
        missing_mapping_count = 0

        # We will batch writes to Drive to make it slightly more resilient, but saving file-by-file is fine.
        for i, img_path in enumerate(images):
            filename = img_path.name
            athlete_name = file_to_athlete.get(filename)
            
            if not athlete_name:
                missing_mapping_count += 1
                continue
                
            if self.is_group_shot(athlete_name):
                skipped_group_count += 1
                continue
                
            athlete_id = self._get_id(athlete_name)
            split = 'train' if i < split_idx else 'val'
            
            # YOLO Auto-labeling
            results = self.detector(img_path, verbose=False)[0]
            person_boxes = [box for box in results.boxes if int(box.cls) == 0]
            
            if not person_boxes:
                continue
            
            best_box = sorted(person_boxes, key=lambda x: x.conf, reverse=True)[0]
            xywh = best_box.xywhn[0].cpu().numpy()
            
            label_line = f"0 {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f} {athlete_id}\n"
            new_img_name = f"{athlete_id}_{zip_path.stem}_{filename}"
            
            dest_img_path = self.output_dir / split / 'images' / new_img_name
            dest_lbl_path = self.output_dir / split / 'labels' / f"{Path(new_img_name).stem}.txt"
            
            # Copy to persistent Drive folder
            shutil.copy(img_path, dest_img_path)
            with open(dest_lbl_path, "w") as f:
                f.write(label_line)
            
            # Save visual check
            if self.vis_count < 50:
                img_cv = cv2.imread(str(img_path))
                h, w, _ = img_cv.shape
                cx, cy, bw, bh = xywh
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_cv, f"ID: {athlete_id}", (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imwrite(str(self.debug_vis_dir / new_img_name), img_cv)
                self.vis_count += 1
                
            processed_count += 1
        
        print(f"  [DONE] Processed {processed_count} images (skipped {skipped_group_count} group shots, {missing_mapping_count} missing DB map).")

        shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir()
        return True

    def run(self, years=['2024', '2025', '2026'], limit_zips=None):
        if not self.db_path.exists():
            print(f"[CRITICAL ERROR] Database file not found at: {self.db_path.absolute()}")
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT DISTINCT year, contest_name FROM athletes")
            contests = cursor.fetchall()
        except sqlite3.OperationalError as e:
            print(f"[ERROR] Database error: {e}")
            return
        finally:
            conn.close()

        print(f"Ingestion Database contains {len(contests)} unique contests.")
        
        zip_count = 0
        for year, contest_name in contests:
            s_year = str(year)
            if s_year not in years:
                continue

            if self.is_contest_processed(year, contest_name):
                # print(f"[SKIPPING] {contest_name} ({year}) already processed.")
                continue

            c_clean = self._sanitize(contest_name)
            zip_filename = f"{year}_{c_clean}.zip"
            zip_path = self.source_root / s_year / zip_filename

            if not zip_path.exists():
                print(f"  [NOT FOUND] {zip_path}")
                continue

            print(f"\nProcessing: {contest_name} ({year})")
            success = self.process_zip(zip_path, year, contest_name)
            
            if success:
                self.mark_contest_processed(year, contest_name)
                zip_count += 1
            
            if limit_zips and zip_count >= limit_zips:
                print(f"Reached limit of {limit_zips} new zips.")
                break

        self.create_dataset_yaml()
        print(f"\nIncremental update complete. {zip_count} new contests processed and saved to {self.output_dir}.")

    def create_dataset_yaml(self):
        yaml_content = f"""
path: {self.output_dir.absolute()}
train: train/images
val: val/images
names:
  0: athlete
"""
        with open(self.output_dir / "ifbb_jde.yaml", "w") as f:
            f.write(yaml_content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to Bodybuilding_Dataset")
    parser.add_argument("--output", type=str, required=True, help="Path to Bodybuilding_Model_Training/ifbb_jde_dataset")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--years", type=str, default="2024,2025,2026")
    args = parser.parse_args()

    builder = IFBBJDEDatasetBuilder(source_root=args.source, output_dir=args.output)
    builder.prepare_directories()
    builder.run(years=args.years.split(','), limit_zips=args.limit)
