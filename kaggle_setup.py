# === KAGGLE SETUP SCRIPT (V25 - MASTER ZIP & SYNC FIX) ===
# 1. Select 'T4 x2' Accelerator in Kaggle
# 2. Enable 'Internet'
# 3. RUN THIS SCRIPT!

import os
import re
import random
import shutil
import zipfile
import hashlib
from collections import defaultdict
from pathlib import Path

# --- 1. CONFIGURATION ---
# Path to your master zip or folder on Kaggle
KAGGLE_DATASET_FOLDER = None 

# Google Drive Link to your MASTER ZIP (ifbb_jde_master.zip)
DRIVE_LINK = "https://drive.google.com/drive/folders/1tvndv5V1O2RI_04-BhPIWdIpa06nd_gP?usp=sharing" 

# OPTIONAL: GDrive folder path to sync weights (requires gcsfuse or similar, or just use for final copy)
SYNC_DIR = None 

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
!pip install --upgrade gdown mlflow wandb --quiet
!pip uninstall ray -y --quiet
!pip install -e tracker/evaluation/TrackEval --quiet

os.makedirs("ultralytics/assets", exist_ok=True)
if not os.path.exists("ultralytics/assets/bus.jpg"):
    import cv2
    import numpy as np
    cv2.imwrite("ultralytics/assets/bus.jpg", np.zeros((640, 640, 3), dtype=np.uint8))

# --- 4. SMART DATA DOWNLOAD & EXTRACTION ---
def extract_id(link):
    match = re.search(r"(?:/d/|id=|folders/)([a-zA-Z0-9_-]+)", link)
    return match.group(1) if match else link

def recursive_unzip(zip_path, extract_to):
    """Unzip a file, and if it contains more zips, unzip those too."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
    
    # Check for nested zips
    nested_zips = list(Path(extract_to).rglob("*.zip"))
    if nested_zips:
        print(f"  [NESTED] Found {len(nested_zips)} nested zips. Extracting...")
        for nz in nested_zips:
            with zipfile.ZipFile(nz, 'r') as z:
                z.extractall(extract_to)
            nz.unlink() # Remove nested zip after extraction

# Check if data already exists AND is valid (Force check for 6th column)
data_ready = False
check_path = Path("datasets/ifbb_jde/train/labels")
if check_path.exists():
    first_label = list(check_path.glob("*.txt"))
    if first_label:
        with open(first_label[0], "r") as f:
            line = f.readline().split()
            if len(line) == 6: # Already has the JDE ID!
                data_ready = True
                print("Dataset already exists with JDE IDs. Skipping download.")

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
            print("Detected Single File Link. Downloading master zip...")
            ZIP_ID = extract_id(DRIVE_LINK)
            !gdown {ZIP_ID} -O /tmp/jde_downloads/master_dataset.zip
            contest_zips = [Path("/tmp/jde_downloads/master_dataset.zip")]
    
    print(f"Found {len(contest_zips)} archive(s). Unzipping recursively...")
    
    if len(contest_zips) == 0:
        print("[CRITICAL ERROR] No zip files found!")
        import sys; sys.exit(1)
        
    for z in contest_zips:
        recursive_unzip(z, "/tmp/jde_raw/")
        
    print("Splitting Data and Injecting JDE IDs...")
    for p in ["train/images", "train/labels", "val/images", "val/labels"]:
        os.makedirs(f"datasets/ifbb_jde/{p}", exist_ok=True)
    
    # Robust search for images anywhere inside the extracted folders
    all_images = list(Path("/tmp/jde_raw").rglob("*.jpg")) + list(Path("/tmp/jde_raw").rglob("*.png")) + list(Path("/tmp/jde_raw").rglob("*.jpeg"))
    
    if len(all_images) == 0:
        print("[CRITICAL ERROR] No images found inside the extracted zip files!")
        !ls -R /tmp/jde_raw | head -n 20
        import sys; sys.exit(1)

    random.shuffle(all_images)
    seen_hashes = set()
    athlete_images = defaultdict(list)
    duplicates_skipped = 0

    for img_path in all_images:
        # Deduplication using MD5
        try:
            with open(img_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
        except:
            continue
        
        if file_hash in seen_hashes:
            duplicates_skipped += 1
            continue
        seen_hashes.add(file_hash)

        athlete_id = img_path.name.split('_')[0]
        label_name = f"{img_path.stem}.txt"
        potential_labels = [
            img_path.parent.parent / "labels" / label_name, 
            img_path.parent / label_name,                   
            Path(str(img_path.parent).replace("images", "labels")) / label_name 
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
    
    val_id_count = max(1, int(len(unique_ids) * 0.10))
    val_ids = set(unique_ids[:val_id_count])
    athlete_to_idx = {name: i for i, name in enumerate(unique_ids)}
    
    valid_pairs = 0
    for athlete_id, pairs in athlete_images.items():
        split = "val" if athlete_id in val_ids else "train"
        idx = athlete_to_idx[athlete_id]
        
        for img_path, label_path in pairs:
            dest_img = f"datasets/ifbb_jde/{split}/images/{img_path.name}"
            shutil.copy(img_path, dest_img)
            
            with open(label_path, "r") as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # Append the JDE tracker ID as the 6th column
                    new_lines.append(f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]} {idx}\n")
            
            dest_label = f"datasets/ifbb_jde/{split}/labels/{label_path.name}"
            with open(dest_label, "w") as f:
                f.writelines(new_lines)
            valid_pairs += 1
            
    print(f"Processed {valid_pairs} image/label pairs with JDE IDs ({len(unique_ids)} unique athletes).")
    if duplicates_skipped > 0:
        print(f"Skipped {duplicates_skipped} duplicate images.")
            
    data_yaml = f"path: /kaggle/working/YOLO11-JDE/datasets/ifbb_jde\ntrain: train/images\nval: val/images\nnc: 1\nnames: ['person']\n"
    with open("datasets/ifbb_jde/ifbb_jde.yaml", "w") as f: f.write(data_yaml)

    # Clean up to save disk space
    shutil.rmtree("/tmp/jde_downloads", ignore_errors=True)
    shutil.rmtree("/tmp/jde_raw", ignore_errors=True)

# --- 5. START TRAINING ---
import torch
import wandb
from kaggle_secrets import UserSecretsClient

try:
    user_secrets = UserSecretsClient()
    wandb_key = user_secrets.get_secret("WANDB_API_KEY")
    wandb.login(key=wandb_key)
    os.environ["WANDB_MODE"] = "online"
except:
    print("WandB Key not found. Using dryrun.")
    os.environ["WANDB_MODE"] = "dryrun"

num_gpus = torch.cuda.device_count()
batch_size = 16 * max(1, num_gpus)
device = ",".join([str(i) for i in range(num_gpus)]) if num_gpus > 0 else "cpu"

print(f"\n--- STARTING TRAINING ON {device.upper()} (GPUs: {num_gpus}) ---")

RESUME_WEIGHTS = "ifbb_jde/bodybuilding_model/weights/last.pt" 
sync_cmd = f"--sync_dir {SYNC_DIR}" if SYNC_DIR else ""

if RESUME_TRAINING:
    resume_cmd = f"--resume {RESUME_WEIGHTS}"
    amp_cmd = "" 
else:
    resume_cmd = ""
    amp_cmd = "--amp" 

# CRITICAL: AMP is False to prevent NaN issues on Kaggle T4
!python train.py --data datasets/ifbb_jde/ifbb_jde.yaml \
                --project ifbb_jde \
                --name bodybuilding_model \
                --epochs 100 \
                --batch {batch_size} \
                --imgsz 960 \
                --device {device} \
                --patience 15 \
                {sync_cmd} \
                {amp_cmd} \
                {resume_cmd}

# --- 6. AUTO-ZIP RESULTS FOR DOWNLOAD ---
print("\n--- ZIPPING RESULTS FOR DOWNLOAD ---")
%cd /kaggle/working
!zip -rq mlflow_results.zip YOLO11-JDE/runs/mlflow
!zip -rq weights_results.zip YOLO11-JDE/ifbb_jde/bodybuilding_model/weights
print("Done! Look for 'mlflow_results.zip' and 'weights_results.zip' in the Output tab.")