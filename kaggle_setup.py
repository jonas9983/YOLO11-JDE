# === KAGGLE SETUP SCRIPT (V11 - CLEAN ARGS) ===
# 1. Select 'T4 x2' Accelerator
# 2. Enable 'Internet'
# 3. Paste, update DRIVE_LINK, and run!

import os
import re

# --- CONFIGURATION ---
DRIVE_LINK = "PASTE_YOUR_FULL_DRIVE_LINK_HERE" 
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

# --- 3. FIX MISSING ASSETS (Bypass AMP error) ---
os.makedirs("ultralytics/assets", exist_ok=True)
if not os.path.exists("ultralytics/assets/bus.jpg"):
    import cv2
    import numpy as np
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.imwrite("ultralytics/assets/bus.jpg", dummy_img)
    print("Created dummy assets/bus.jpg")

os.environ["WANDB_MODE"] = "disabled"

# --- 4. DATA DOWNLOAD ---
def extract_id(link):
    match = re.search(r"(?:/d/|id=)([a-zA-Z0-9_-]+)", link)
    return match.group(1) if match else link

ZIP_ID = extract_id(DRIVE_LINK)
print(f"Downloading dataset (ID: {ZIP_ID})...")

if os.path.exists("prague_pro.zip"): !rm prague_pro.zip
direct_url = f"https://drive.google.com/uc?id={ZIP_ID}"
!gdown {direct_url} -O prague_pro.zip

if os.path.exists("prague_pro.zip") and os.path.getsize("prague_pro.zip") > 1000000:
    print("Unzipping data...")
    !mkdir -p datasets/ifbb_jde
    !unzip -qo prague_pro.zip -d datasets/ifbb_jde/
    
    # Create YAML
    data_yaml = """
path: /kaggle/working/YOLO11-JDE/datasets/ifbb_jde
train: images
val: images
nc: 1
names: ['person']
"""
    with open("datasets/ifbb_jde/ifbb_jde.yaml", "w") as f:
        f.write(data_yaml)

    # --- 5. START TRAINING ---
    import torch
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"\n--- STARTING TRAINING ON {device.upper()} ---")

    # Final training command
    !python train.py --data datasets/ifbb_jde/ifbb_jde.yaml \
                    --project ifbb_jde \
                    --name prague_pro_test \
                    --epochs 50 \
                    --batch 4 \
                    --imgsz 1280 \
                    --device {device} \
                    --amp
else:
    print(f"[ERROR] Download failed. Verify Drive link permissions.")
