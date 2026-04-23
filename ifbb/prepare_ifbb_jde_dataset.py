import os
import sqlite3
import torch
import shutil
import argparse
import zipfile
from pathlib import Path
from tqdm import tqdm
import numpy as np
from ultralytics import YOLO

class IFBBJDEDatasetBuilder:
    def __init__(self, 
                 source_root, 
                 output_dir="datasets/ifbb_jde",
                 device="cpu"):
        self.source_root = Path(source_root)
        self.db_path = self.source_root / "npc_database.db"
        self.output_dir = Path(output_dir)
        self.device = device
        self.temp_dir = Path("temp_unzip")
        
        print(f"\n--- DEBUG START ---")
        print(f"Source Root: {self.source_root.absolute()}")
        print(f"Database Path: {self.db_path.absolute()}")
        
        # Load a base model for auto-labeling (person detection)
        self.detector = YOLO("yolo11n.pt").to(self.device)
        
        self.id_map = {}
        self.next_id = 0

    def _sanitize(self, name):
        """Standard sanitization used during ingestion."""
        return "".join(c for c in name if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")

    def _get_id(self, athlete_name):
        athlete_name = athlete_name.strip().upper()
        if athlete_name not in self.id_map:
            self.id_map[athlete_name] = self.next_id
            self.next_id += 1
        return self.id_map[athlete_name]

    def prepare_directories(self):
        print(f"Preparing output directories at: {self.output_dir}")
        for split in ['train', 'val']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def process_zip(self, zip_path, year, contest_name, split_ratio=0.9):
        print(f"  [ZIP] Unzipping {zip_path.name}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.temp_dir)
        except Exception as e:
            print(f"  [ERROR] Failed to unzip {zip_path.name}: {e}")
            return

        # Fetch mapping of filename -> athlete_name from DB for this contest
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT image_filename, athlete_name FROM athletes WHERE year = ? AND contest_name = ?", (year, contest_name))
        file_to_athlete = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()

        # Find all images recursively inside the unzipped folder
        images = [img for img in self.temp_dir.glob("**/*") if img.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        print(f"  [INFO] Found {len(images)} images in ZIP.")
        
        if not images:
            print(f"    [SKIP] No images found inside ZIP.")
            shutil.rmtree(self.temp_dir)
            self.temp_dir.mkdir()
            return

        np.random.shuffle(images)
        split_idx = int(len(images) * split_ratio)
        
        processed_count = 0
        missing_mapping_count = 0

        for i, img_path in enumerate(images):
            # Find the athlete name using the filename
            filename = img_path.name
            athlete_name = file_to_athlete.get(filename)
            
            if not athlete_name:
                missing_mapping_count += 1
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
            
            # Global unique filename: {athlete_id}_{zip_name}_{orig_name}
            new_img_name = f"{athlete_id}_{zip_path.stem}_{filename}"
            dest_img_path = self.output_dir / split / 'images' / new_img_name
            dest_lbl_path = self.output_dir / split / 'labels' / f"{Path(new_img_name).stem}.txt"
            
            shutil.copy(img_path, dest_img_path)
            with open(dest_lbl_path, "w") as f:
                f.write(label_line)
            processed_count += 1
        
        print(f"    [DONE] Processed {processed_count} images (skipped {missing_mapping_count} due to missing DB map).")

        shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir()

    def run(self, years=['2024', '2025', '2026'], limit_zips=None):
        if not self.db_path.exists():
            print(f"[CRITICAL ERROR] Database file not found at: {self.db_path.absolute()}")
            if self.source_root.exists():
                print(f"Contents of {self.source_root.absolute()}:")
                print(os.listdir(self.source_root))
            else:
                print(f"Source Root folder does not exist!")
            return

        print(f"Connecting to database...")
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

        print(f"Database contains {len(contests)} unique contests.")
        if not contests:
            print("[ERROR] Database returned 0 contests. Check if 'athletes' table has data.")
            return
        
        zip_count = 0
        for year, contest_name in contests:
            s_year = str(year)
            if s_year not in years:
                continue

            c_clean = self._sanitize(contest_name)
            zip_filename = f"{year}_{c_clean}.zip"
            zip_path = self.source_root / s_year / zip_filename

            if not zip_path.exists():
                print(f"  [NOT FOUND] {zip_path}")
                continue

            print(f"Processing: {contest_name} ({year})")
            self.process_zip(zip_path, year, contest_name)
            zip_count += 1
            
            if limit_zips and zip_count >= limit_zips:
                print(f"Reached limit of {limit_zips} zips.")
                break

        with open(self.output_dir / "id_map.txt", "w") as f:
            for name, idx in self.id_map.items():
                f.write(f"{idx}: {name}\n")
        
        self.create_dataset_yaml()
        print(f"\nTotal contests processed: {zip_count}")
        print(f"--- DEBUG END ---\n")

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
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output", type=str, default="datasets/ifbb_jde")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--years", type=str, default="2024,2025,2026")
    args = parser.parse_args()

    builder = IFBBJDEDatasetBuilder(source_root=args.source, output_dir=args.output)
    builder.prepare_directories()
    builder.run(years=args.years.split(','), limit_zips=args.limit)
