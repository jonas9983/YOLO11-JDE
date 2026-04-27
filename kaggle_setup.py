# === KAGGLE SETUP SCRIPT (V24 - PERSISTENT STATE FIX) ===
# 1. Select 'T4 x2' Accelerator in Kaggle
# 2. Enable 'Internet'
# 3. RUN THIS SCRIPT!

import os
import re
import random
import shutil
import hashlib
from collections import defaultdict
from pathlib import Path

# --- 1. CONFIGURATION ---
# If you uploaded your zips directly to Kaggle (Add Data -> New Dataset), paste the folder path here:
# e.g., KAGGLE_DATASET_FOLDER = "/kaggle/input/ifbb-jde-zips"
KAGGLE_DATASET_FOLDER = None 

# OR, if using Google Drive, paste the link here (make sure it's "Anyone with the link"):
DRIVE_LINK = "https://drive.google.com/drive/folders/1tvndv5V1O2RI_04-BhPIWdIpa06nd_gP?usp=sharing" 

RESUME_TRAINING = False 
REPO_URL = "https://github.com/jonas9983/YOLO11-JDE.git"
BRANCH = "feat/multi-gpu-training"

# --- 2. REPO SETUP ---
%cd /kaggle/working
if not os.path.exists("YOLO11-JDE"):
    !git clone {REPO_URL}
%cd YOLO11-JDE
!git checkout {BRANCH}
!git pull origin {BRANCH}

os.environ["PYTHONPATH"] = f"{os.getcwd()}:{os.environ.get('PYTHONPATH', '')}"

# --- 3. DEPENDENCIES ---
print("Installing dependencies...")
!pip install -r requirements.txt --quiet
!pip install --upgrade gdown mlflow --quiet
!pip uninstall ray -y --quiet
!pip install -e tracker/evaluation/TrackEval --quiet

os.makedirs("ultralytics/assets", exist_ok=True)
if not os.path.exists("ultralytics/assets/bus.jpg"):
    import cv2
    import numpy as np
    cv2.imwrite("ultralytics/assets/bus.jpg", np.zeros((640, 640, 3), dtype=np.uint8))

os.environ["WANDB_MODE"] = "disabled"

# --- 4. SMART DATA DOWNLOAD & EXTRACTION ---
def extract_id(link):
    match = re.search(r"(?:/d/|id=|folders/)([a-zA-Z0-9_-]+)", link)
    return match.group(1) if match else link

# Check if data already exists AND is valid
data_ready = False
check_path = Path("datasets/ifbb_jde/train/images")
if check_path.exists() and len(list(check_path.glob("*.jpg"))) > 0:
    data_ready = True
    print("Dataset already exists and contains images. Skipping download.")

if not data_ready:
    print("Cleaning up old/empty dataset folders...")
    shutil.rmtree("datasets/ifbb_jde", ignore_errors=True)
    
    !mkdir -p /tmp/jde_downloads
    !mkdir -p /tmp/jde_raw
    
    if KAGGLE_DATASET_FOLDER and os.path.exists(KAGGLE_DATASET_FOLDER):
        print(f"Detected Native Kaggle Dataset at {KAGGLE_DATASET_FOLDER}...")
        contest_zips = list(Path(KAGGLE_DATASET_FOLDER).rglob("*.zip"))
    else:
        print("Downloading dataset from Google Drive...")
        if "folder" in DRIVE_LINK or "drive.google.com/drive/folders/" in DRIVE_LINK:
            print("Detected Google Drive Folder Link. Downloading contents...")
            !gdown --folder "{DRIVE_LINK}" -O /tmp/jde_downloads
            contest_zips = list(Path("/tmp/jde_downloads").rglob("*.zip"))
        else:
            print("Detected Single File Link. Downloading zip...")
            ZIP_ID = extract_id(DRIVE_LINK)
            !gdown {ZIP_ID} -O /tmp/jde_downloads/dataset.zip
            contest_zips = [Path("/tmp/jde_downloads/dataset.zip")]
    
    print(f"Found {len(contest_zips)} contest zips. Unzipping...")
    
    if len(contest_zips) == 0:
        print("[CRITICAL ERROR] No zip files found! If using Google Drive, ensure permissions are set to 'Anyone with the link'.")
        import sys; sys.exit(1)
        
    for z in contest_zips:
        !unzip -qo "{str(z)}" -d /tmp/jde_raw/
        
    print("Splitting Data into Train/Val...")
    for p in ["train/images", "train/labels", "val/images", "val/labels"]:
        os.makedirs(f"datasets/ifbb_jde/{p}", exist_ok=True)
    
    # Robust search for images anywhere inside the extracted folders
    all_images = list(Path("/tmp/jde_raw").rglob("*.jpg")) + list(Path("/tmp/jde_raw").rglob("*.png")) + list(Path("/tmp/jde_raw").rglob("*.jpeg"))
    
    if len(all_images) == 0:
        print("[CRITICAL ERROR] No images found inside the extracted zip files!")
        print("Checking contents of /tmp/jde_raw:")
        !ls -la /tmp/jde_raw
        import sys; sys.exit(1)

    random.shuffle(all_images)
    
    seen_hashes = set()
    athlete_images = defaultdict(list)
    duplicates_skipped = 0

    print("Hashing images and grouping by athlete ID...")
    for img_path in all_images:
        # 1. Hash-based de-duplication
        with open(img_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        if file_hash in seen_hashes:
            duplicates_skipped += 1
            continue
        seen_hashes.add(file_hash)

        # 2. Extract athlete_id from filename
        # Expected format: {athlete_id}_{original_filename}.jpg
        athlete_id = img_path.name.split('_')[0]
        
        # Check potential label locations
        label_name = f"{img_path.stem}.txt"
        potential_labels = [
            img_path.parent.parent / "labels" / label_name, # standard YOLO structure
            img_path.parent / label_name,                   # Same folder
            Path(str(img_path.parent).replace("images", "labels")) / label_name # Sibling folder
        ]
        
        label_path = None
        for p in potential_labels:
            if p.exists():
                label_path = p
                break
                
        if label_path:
            athlete_images[athlete_id].append((img_path, label_path))

    unique_ids = list(athlete_images.keys())
    random.shuffle(unique_ids)
    
    # Identity-aware splitting: roughly 10% of athletes go to validation
    val_id_count = max(1, int(len(unique_ids) * 0.10))
    val_ids = set(unique_ids[:val_id_count])
    
    print(f"Skipped {duplicates_skipped} exact duplicate images.")
    print(f"Found {len(unique_ids)} unique athletes.")
    print(f"Assigning {val_id_count} athletes to Validation...")
    
    # Create a mapping from athlete_id string to a unique integer
    athlete_to_idx = {athlete_name: i for i, athlete_name in enumerate(unique_ids)}
    
    valid_pairs = 0
    for athlete_id, pairs in athlete_images.items():
        split = "val" if athlete_id in val_ids else "train"
        idx = athlete_to_idx[athlete_id]
        
        for img_path, label_path in pairs:
            dest_img = f"datasets/ifbb_jde/{split}/images/{img_path.name}"
            shutil.copy(img_path, dest_img)
            
            # Read, inject ID, and save
            with open(label_path, "r") as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # class x y w h -> class x y w h identity_id
                    new_lines.append(f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]} {idx}\n")
            
            dest_label = f"datasets/ifbb_jde/{split}/labels/{label_path.name}"
            with open(dest_label, "w") as f:
                f.writelines(new_lines)
            valid_pairs += 1
            
    print(f"Successfully processed {valid_pairs} image/label pairs with JDE IDs.")
    
    if valid_pairs == 0:
        print("[CRITICAL ERROR] No valid image/label pairs!")
        import sys; sys.exit(1)
            
    data_yaml = f"path: /kaggle/working/YOLO11-JDE/datasets/ifbb_jde\ntrain: train/images\nval: val/images\nnc: 1\nnames: ['person']\n"
    with open("datasets/ifbb_jde/ifbb_jde.yaml", "w") as f: f.write(data_yaml)

    shutil.rmtree("/tmp/jde_downloads", ignore_errors=True)
    shutil.rmtree("/tmp/jde_raw", ignore_errors=True)

# --- 5. START TRAINING ---
import torch
import wandb
from kaggle_secrets import UserSecretsClient

# Setup WandB
try:
    user_secrets = UserSecretsClient()
    wandb_key = user_secrets.get_secret("WANDB_API_KEY")
    wandb.login(key=wandb_key)
    os.environ["WANDB_MODE"] = "online"
except:
    print("WandB Key not found in Kaggle Secrets. Logging as anonymous or disabled.")
    os.environ["WANDB_MODE"] = "dryrun"

num_gpus = torch.cuda.device_count()
batch_size = 16 * max(1, num_gpus)
device = ",".join([str(i) for i in range(num_gpus)]) if num_gpus > 0 else "cpu"

print(f"\n--- STARTING CORRECT JDE TRAINING ON {device.upper()} ---")

!python train.py --data datasets/ifbb_jde/ifbb_jde.yaml \
                --project ifbb_jde \
                --name bodybuilding_model \
                --epochs 100 \
                --batch {batch_size} \
                --imgsz 960 \
                --device {device} \
                --amp \
                {"--resume True" if RESUME_TRAINING else ""}

# --- 6. AUTO-ZIP RESULTS ---
print("\n--- ZIPPING RESULTS FOR DOWNLOAD ---")
%cd /kaggle/working
!zip -rq mlflow_results.zip YOLO11-JDE/runs/mlflow
!zip -rq weights_results.zip YOLO11-JDE/ifbb_jde/bodybuilding_model/weights
print("Done! Look for 'mlflow_results.zip' and 'weights_results.zip' in the Kaggle 'Output' tab.")