import os
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
        self.output_dir = Path(output_dir)
        self.device = device
        self.temp_dir = Path("temp_unzip")
        
        # Load a base model for auto-labeling (person detection)
        self.detector = YOLO("yolo11n.pt").to(self.device)
        
        self.id_map = {}
        self.next_id = 0

    def _get_id(self, athlete_name):
        if athlete_name not in self.id_map:
            self.id_map[athlete_name] = self.next_id
            self.next_id += 1
        return self.id_map[athlete_name]

    def prepare_directories(self):
        for split in ['train', 'val']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def process_zip(self, zip_path, split_ratio=0.9):
        # 1. Unzip to temp
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.temp_dir)
        
        # 2. Process athlete folders inside the unzip
        # Inside the zip, structure is usually athlete_name/images...
        athlete_folders = [f for f in self.temp_dir.iterdir() if f.is_dir()]
        
        for athlete_folder in athlete_folders:
            athlete_name = athlete_folder.name
            athlete_id = self._get_id(athlete_name)
            
            images = [img for img in athlete_folder.glob("**/*") if img.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            if not images:
                continue

            np.random.shuffle(images)
            split_idx = int(len(images) * split_ratio)
            
            for i, img_path in enumerate(images):
                split = 'train' if i < split_idx else 'val'
                
                # Auto-labeling
                results = self.detector(img_path, verbose=False)[0]
                person_boxes = [box for box in results.boxes if int(box.cls) == 0]
                
                if not person_boxes:
                    continue
                
                best_box = sorted(person_boxes, key=lambda x: x.conf, reverse=True)[0]
                xywh = best_box.xywhn[0].cpu().numpy()
                
                # Format: class x y w h identity
                label_line = f"0 {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f} {athlete_id}\n"
                
                # Unique name to avoid collisions across contests
                new_img_name = f"{athlete_id}_{img_path.stem}_{img_path.name}"
                dest_img_path = self.output_dir / split / 'images' / new_img_name
                dest_lbl_path = self.output_dir / split / 'labels' / f"{Path(new_img_name).stem}.txt"
                
                shutil.copy(img_path, dest_img_path)
                with open(dest_lbl_path, "w") as f:
                    f.write(label_line)

        # 3. Cleanup temp
        shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir()

    def run(self, years=['2024', '2025', '2026'], limit_zips=None):
        zip_count = 0
        for year in years:
            year_dir = self.source_root / year
            if not year_dir.exists():
                continue
            
            zips = list(year_dir.glob("*.zip"))
            for z in zips:
                print(f"Processing {z.name}...")
                self.process_zip(z)
                zip_count += 1
                if limit_zips and zip_count >= limit_zips:
                    print(f"Reached limit of {limit_zips} zips.")
                    return

        # Save mapping
        with open(self.output_dir / "id_map.txt", "w") as f:
            for name, idx in self.id_map.items():
                f.write(f"{idx}: {name}\n")
        
        self.create_dataset_yaml()

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
    parser.add_argument("--source", type=str, required=True, help="Path to BodybuildingDataset root")
    parser.add_argument("--output", type=str, default="datasets/ifbb_jde", help="Output dataset dir")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of zips for smoke test")
    parser.add_argument("--years", type=str, default="2024,2025,2026", help="Comma separated years")
    args = parser.parse_args()

    builder = IFBBJDEDatasetBuilder(source_root=args.source, output_dir=args.output)
    builder.prepare_directories()
    builder.run(years=args.years.split(','), limit_zips=args.limit)
    print(f"Dataset preparation complete!")
