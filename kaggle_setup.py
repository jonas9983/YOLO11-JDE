# === KAGGLE SETUP SCRIPT (V17 - BATCH OPTIMIZED FOR JDE) ===
# 1. Select 'T4 x2' Accelerator in Kaggle
# 2. Enable 'Internet'
# 3. Paste, update CONFIG, and run!

import os
import re
import random
import shutil
from pathlib import Path

# --- CONFIGURATION ---
# Paste the Google Drive link to your new dataset zip (e.g., 2026 data or the full dataset)
DRIVE_LINK = "PASTE_YOUR_FULL_DRIVE_LINK_HERE" 

RESUME_TRAINING = False
REPO_URL = "https://github.com/jonas9983/YOLO11-JDE.git"
BRANCH = "feat/multi-gpu-training"

# --- 1. REPO SETUP ---
%cd /kaggle/working
if not os.path.exists("YOLO11-JDE"):
    !git clone {REPO_URL}
%cd YOLO11-JDE
!git checkout {BRANCH}
!git pull origin {BRANCH}

# Set PYTHONPATH
os.environ["PYTHONPATH"] = f"{os.getcwd()}:{os.environ.get('PYTHONPATH', '')}"

# --- 2. DEPENDENCIES ---
print("Installing dependencies...")
!pip install -r requirements.txt --quiet
!pip install --upgrade gdown mlflow --quiet
!pip uninstall ray -y --quiet
!pip install -e tracker/evaluation/TrackEval --quiet

# --- 3. FIX ASSETS ---
os.makedirs("ultralytics/assets", exist_ok=True)
if not os.path.exists("ultralytics/assets/bus.jpg"):
    import cv2
    import numpy as np
    cv2.imwrite("ultralytics/assets/bus.jpg", np.zeros((640, 640, 3), dtype=np.uint8))

os.environ["WANDB_MODE"] = "disabled"

# --- 4. DATA DOWNLOAD & SPLIT ---
def extract_id(link):
    match = re.search(r"(?:/d/|id=)([a-zA-Z0-9_-]+)", link)
    return match.group(1) if match else link

if not os.path.exists("datasets/ifbb_jde"):
    ZIP_ID = extract_id(DRIVE_LINK)
    print(f"Downloading dataset (ID: {ZIP_ID})...")
    if os.path.exists("dataset.zip"): os.remove("dataset.zip")
    !python -m gdown {ZIP_ID} -O dataset.zip

    if os.path.exists("dataset.zip") and os.path.getsize("dataset.zip") > 1000000:
        print("Unzipping and Splitting Data...")
        !mkdir -p /tmp/jde_raw
        !unzip -qo dataset.zip -d /tmp/jde_raw/
        
        # If the zip already contains a train/val split, just move it
        if os.path.exists("/tmp/jde_raw/train") and os.path.exists("/tmp/jde_raw/val"):
            shutil.move("/tmp/jde_raw", "datasets/ifbb_jde")
        else:
            # Auto-split if it's just a folder of images and labels
            for p in ["train/images", "train/labels", "val/images", "val/labels"]:
                os.makedirs(f"datasets/ifbb_jde/{p}", exist_ok=True)
            
            # Find images wherever they are
            all_images = list(Path("/tmp/jde_raw").rglob("*.jpg")) + list(Path("/tmp/jde_raw").rglob("*.png"))
            random.shuffle(all_images)
            val_count = int(len(all_images) * 0.15) # 15% validation
            
            for i, img_path in enumerate(all_images):
                split = "val" if i < val_count else "train"
                label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
                # Fallback if labels are in the same folder
                if not label_path.exists():
                    label_path = img_path.parent / f"{img_path.stem}.txt"
                
                if label_path.exists():
                    shutil.copy(img_path, f"datasets/ifbb_jde/{split}/images/{img_path.name}")
                    shutil.copy(label_path, f"datasets/ifbb_jde/{split}/labels/{label_path.name}")
                    
        data_yaml = f"path: /kaggle/working/YOLO11-JDE/datasets/ifbb_jde\ntrain: train/images\nval: val/images\nnc: 1\nnames: ['person']\n"
        with open("datasets/ifbb_jde/ifbb_jde.yaml", "w") as f: f.write(data_yaml)

# --- 5. START TRAINING ---
import torch
# Check if multiple GPUs are available
num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    device = ",".join([str(i) for i in range(num_gpus)])
    batch_size = 16 * num_gpus # 32 total on dual GPUs
elif num_gpus == 1:
    device = "0"
    batch_size = 16
else:
    device = "cpu"
    batch_size = 4

print(f"\n--- STARTING TRAINING ON {device.upper()} (GPUs: {num_gpus}) ---")

RESUME_WEIGHTS = "ifbb_jde/bodybuilding_model/weights/last.pt" 
resume_cmd = f"--resume {RESUME_WEIGHTS}" if RESUME_TRAINING else ""

# --- OPTIMIZATIONS ---
# 1. Batch size increased to 16/32 to allow proper Contrastive Learning (Re-ID)
# 2. Imgsz reduced to 960 to fit the larger batch size in T4 VRAM
# 3. AMP enabled to save VRAM and train faster
!python train.py --data datasets/ifbb_jde/ifbb_jde.yaml \
                --project ifbb_jde \
                --name bodybuilding_model \
                --epochs 100 \
                --batch {batch_size} \
                --imgsz 960 \
                --device {device} \
                --amp \
                {resume_cmd}

# --- 6. AUTO-ZIP RESULTS FOR DOWNLOAD ---
print("\n--- ZIPPING RESULTS FOR DOWNLOAD ---")
%cd /kaggle/working
!zip -rq mlflow_results.zip YOLO11-JDE/runs/mlflow
!zip -rq weights_results.zip YOLO11-JDE/ifbb_jde/bodybuilding_model/weights
print("Done! Look for 'mlflow_results.zip' and 'weights_results.zip' in the Kaggle 'Output' tab.")

