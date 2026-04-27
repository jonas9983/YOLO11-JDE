# === KAGGLE SETUP SCRIPT (V20 - DIRECT FOLDER DOWNLOAD) ===
# 1. Select 'T4 x2' Accelerator in Kaggle
# 2. Enable 'Internet'
# 3. RUN THIS SCRIPT!

import os
import re
import random
import shutil
from pathlib import Path

# --- 1. CONFIGURATION ---
# Right-click your "ifbb_jde_dataset" folder in Google Drive -> Share -> Copy Link
# Paste that folder link below:
DRIVE_FOLDER_LINK = "PASTE_YOUR_FOLDER_LINK_HERE" 

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

# --- 4. MULTI-ZIP FOLDER DOWNLOAD & EXTRACTION ---
if not os.path.exists("datasets/ifbb_jde"):
    print("Downloading entire contest folder from Google Drive...")
    !mkdir -p /tmp/jde_downloads
    !mkdir -p /tmp/jde_raw
    
    # Use gdown to download the entire folder!
    # gdown automatically handles the folder link if we pass --folder
    !gdown --folder "{DRIVE_FOLDER_LINK}" -O /tmp/jde_downloads
    
    # Find all the downloaded zips
    contest_zips = list(Path("/tmp/jde_downloads").rglob("*.zip"))
    print(f"Found {len(contest_zips)} contest zips. Unzipping all of them...")
    
    for z in contest_zips:
        !unzip -qo "{str(z)}" -d /tmp/jde_raw/
        
    print("Splitting Data into Train/Val...")
    for p in ["train/images", "train/labels", "val/images", "val/labels"]:
        os.makedirs(f"datasets/ifbb_jde/{p}", exist_ok=True)
    
    all_images = list(Path("/tmp/jde_raw").rglob("*.jpg")) + list(Path("/tmp/jde_raw").rglob("*.png"))
    random.shuffle(all_images)
    
    val_count = int(len(all_images) * 0.10) 
    
    print(f"Moving {len(all_images)} total images ({val_count} to Validation)...")
    for i, img_path in enumerate(all_images):
        split = "val" if i < val_count else "train"
        label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            label_path = img_path.parent / f"{img_path.stem}.txt"
        
        if label_path.exists():
            shutil.copy(img_path, f"datasets/ifbb_jde/{split}/images/{img_path.name}")
            shutil.copy(label_path, f"datasets/ifbb_jde/{split}/labels/{label_path.name}")
            
    data_yaml = f"path: /kaggle/working/YOLO11-JDE/datasets/ifbb_jde\ntrain: train/images\nval: val/images\nnc: 1\nnames: ['person']\n"
    with open("datasets/ifbb_jde/ifbb_jde.yaml", "w") as f: f.write(data_yaml)

    # Clean up to save massive disk space on Kaggle
    shutil.rmtree("/tmp/jde_downloads", ignore_errors=True)
    shutil.rmtree("/tmp/jde_raw", ignore_errors=True)

# --- 5. START TRAINING ---
import torch
num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    device = ",".join([str(i) for i in range(num_gpus)])
    batch_size = 16 * num_gpus 
elif num_gpus == 1:
    device = "0"
    batch_size = 16
else:
    device = "cpu"
    batch_size = 4

print(f"\n--- STARTING TRAINING ON {device.upper()} (GPUs: {num_gpus}) ---")

RESUME_WEIGHTS = "ifbb_jde/bodybuilding_model/weights/last.pt" 
if RESUME_TRAINING:
    resume_cmd = f"--resume {RESUME_WEIGHTS}"
    amp_cmd = "" 
else:
    resume_cmd = ""
    amp_cmd = "--amp" 

!python train.py --data datasets/ifbb_jde/ifbb_jde.yaml \
                --project ifbb_jde \
                --name bodybuilding_model \
                --epochs 100 \
                --batch {batch_size} \
                --imgsz 960 \
                --device {device} \
                {amp_cmd} \
                {resume_cmd}

# --- 6. AUTO-ZIP RESULTS ---
print("\n--- ZIPPING RESULTS FOR DOWNLOAD ---")
%cd /kaggle/working
!zip -rq mlflow_results.zip YOLO11-JDE/runs/mlflow
!zip -rq weights_results.zip YOLO11-JDE/ifbb_jde/bodybuilding_model/weights
print("Done! Look for 'mlflow_results.zip' and 'weights_results.zip' in the Kaggle 'Output' tab.")