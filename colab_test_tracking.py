# === COLAB TRACKING SCRIPT (V1 - TEST YOUR WEIGHTS) ===
# 1. Select 'T4 GPU' Accelerator
# 2. Mount your Google Drive
# 3. Paste, update PATHS, and run!

import os
from google.colab import drive

# --- 1. MOUNT DRIVE ---
if not os.path.exists("/content/drive"):
    drive.mount('/content/drive')

# --- 2. CONFIGURATION (UPDATE THESE!) ---
# Path to your weights zip in Drive
WEIGHTS_ZIP = "/content/drive/MyDrive/YOLO11_Results/weights_results.zip" 
# Path to your test video in Drive
TEST_VIDEO = "/content/drive/MyDrive/IFBB_Videos/prague_pro_stage.mp4"
# Your GitHub details
REPO_URL = "https://github.com/jonas9983/YOLO11-JDE.git"
BRANCH = "development"

# --- 3. REPO SETUP ---
%cd /content
if not os.path.exists("YOLO11-JDE"):
    !git clone {REPO_URL}
%cd YOLO11-JDE
!git checkout {BRANCH}
!git pull origin {BRANCH}

# Set PYTHONPATH
os.environ["PYTHONPATH"] = f"{os.getcwd()}:{os.environ.get('PYTHONPATH', '')}"

# --- 4. DEPENDENCIES ---
print("Installing dependencies...")
!pip install -r requirements.txt --quiet
!pip install --upgrade gdown mlflow --quiet

# --- 5. PREPARE WEIGHTS ---
print("Unzipping weights...")
!mkdir -p /content/test_weights
!unzip -qo {WEIGHTS_ZIP} -d /content/test_weights/

# Path to the actual .pt file inside the unzipped folder
# Based on our previous zip: YOLO11-JDE/ifbb_jde/prague_pro_final/weights/best.pt
MODEL_PATH = "/content/test_weights/YOLO11-JDE/ifbb_jde/prague_pro_final/weights/best.pt"

# --- 6. RUN TRACKING ---
print("\n--- STARTING TRACKING ---")
# Adjust confidence and tracking parameters here
!python track_bodybuilders.py --weights {MODEL_PATH} \
                             --source {TEST_VIDEO} \
                             --save-vid \
                             --conf 0.3 \
                             --imgsz 1280

# --- 7. SAVE OUTPUT BACK TO DRIVE ---
print("\n--- SAVING RESULT TO DRIVE ---")
import glob
output_vids = glob.glob("runs/track/*/weights/*.mp4") # Check where your script saves
if output_vids:
    !cp {output_vids[0]} /content/drive/MyDrive/YOLO11_Results/tracking_test_result.mp4
    print(f"Success! Video saved to Drive: YOLO11_Results/tracking_test_result.mp4")
else:
    print("Could not find output video. Check the 'runs' folder in Colab.")
