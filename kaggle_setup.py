# === KAGGLE SETUP SCRIPT (V15 - MULTI-GPU SUPPORT) ===
# 1. Select 'T4 x2' Accelerator
# 2. Enable 'Internet'
# 3. Paste, update CONFIG, and run!

import os
import re
import random
import shutil
from pathlib import Path

# --- CONFIGURATION ---
DRIVE_LINK = "PASTE_YOUR_FULL_DRIVE_LINK_HERE" 
RESUME_TRAINING = False  # SET TO FALSE TO START FRESH AFTER THE NAN ISSUE

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
    if os.path.exists("prague_pro.zip"): !rm prague_pro.zip
    !python -m gdown {ZIP_ID} -O prague_pro.zip

    if os.path.exists("prague_pro.zip") and os.path.getsize("prague_pro.zip") > 1000000:
        print("Unzipping and Splitting Data...")
        !mkdir -p /tmp/jde_raw
        !unzip -qo prague_pro.zip -d /tmp/jde_raw/
        for p in ["train/images", "train/labels", "val/images", "val/labels"]:
            os.makedirs(f"datasets/ifbb_jde/{p}", exist_ok=True)
        all_images = list(Path("/tmp/jde_raw/images").glob("*.*"))
        random.shuffle(all_images)
        val_count = int(len(all_images) * 0.2)
        for i, img_path in enumerate(all_images):
            split = "val" if i < val_count else "train"
            label_path = Path("/tmp/jde_raw/labels") / f"{img_path.stem}.txt"
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
    batch_size = 4 * num_gpus # Double the batch size for 2 GPUs
elif num_gpus == 1:
    device = "0"
    batch_size = 4
else:
    device = "cpu"
    batch_size = 4

print(f"\n--- STARTING TRAINING ON {device.upper()} (GPUs: {num_gpus}) ---")

# Using specific weights for resume if needed, otherwise starts fresh
RESUME_WEIGHTS = "ifbb_jde/prague_pro_final/weights/best.pt" 

if RESUME_TRAINING:
    resume_cmd = f"--resume {RESUME_WEIGHTS}"
else:
    resume_cmd = ""

# CRITICAL: AMP is False to prevent NaN issues on Kaggle T4
!python train.py --data datasets/ifbb_jde/ifbb_jde.yaml \
                --project ifbb_jde \
                --name prague_pro_final \
                --epochs 50 \
                --batch {batch_size} \
                --imgsz 1280 \
                --device {device} \
                {resume_cmd}

# --- 6. AUTO-ZIP RESULTS FOR DOWNLOAD ---
print("\n--- ZIPPING RESULTS FOR DOWNLOAD ---")
%cd /kaggle/working
!zip -rq mlflow_results.zip YOLO11-JDE/runs/mlflow
!zip -rq weights_results.zip YOLO11-JDE/ifbb_jde/prague_pro_final/weights
print("Done! Look for 'mlflow_results.zip' and 'weights_results.zip' in the Output tab.")

