# === KAGGLE SETUP SCRIPT (V13 - RESUME SUPPORT) ===
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
RESUME_TRAINING = True  # SET TO TRUE TO RESUME, FALSE TO START NEW

REPO_URL = "https://github.com/jonas9983/YOLO11-JDE.git"
BRANCH = "feat/ifbb-jde-training"

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

# --- 4. DATA DOWNLOAD & SPLIT (Only if not resuming) ---
def extract_id(link):
    match = re.search(r"(?:/d/|id=)([a-zA-Z0-9_-]+)", link)
    return match.group(1) if match else link

if not RESUME_TRAINING or not os.path.exists("datasets/ifbb_jde"):
    ZIP_ID = extract_id(DRIVE_LINK)
    print(f"Downloading dataset (ID: {ZIP_ID})...")
    if os.path.exists("prague_pro.zip"): !rm prague_pro.zip
    !python -m gdown {ZIP_ID} -O prague_pro.zip

    if os.path.exists("prague_pro.zip") and os.path.getsize("prague_pro.zip") > 1000000:
        if os.path.exists("datasets/ifbb_jde"): shutil.rmtree("datasets/ifbb_jde")
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
device = "0" if torch.cuda.is_available() else "cpu"
print(f"\n--- STARTING TRAINING ON {device.upper()} (RESUME={RESUME_TRAINING}) ---")

resume_flag = "--resume" if RESUME_TRAINING else ""

!python train.py --data datasets/ifbb_jde/ifbb_jde.yaml \
                --project ifbb_jde \
                --name prague_pro_final \
                --epochs 50 \
                --batch 4 \
                --imgsz 1280 \
                --device {device} \
                --amp {resume_flag}
