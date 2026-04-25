# === COLAB TRACKING SCRIPT (V2 - ROBUST PATHS) ===
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
# Path to your test video in Drive (Spaces are okay here)
TEST_VIDEO = "/content/drive/MyDrive/personal/Bodybuilding_Dataset/Videos/OPEN BODYBUILDING - EVLS PRAG PRO 2025 (FINALE IN 4K).mp4"
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
if not os.path.exists(WEIGHTS_ZIP):
    print(f"[ERROR] Weights zip not found at: {WEIGHTS_ZIP}")
    print("Please check your Drive path and make sure the file is there!")
else:
    print("Unzipping weights...")
    !mkdir -p /content/test_weights
    !unzip -qo "{WEIGHTS_ZIP}" -d /content/test_weights/

# Path to the actual .pt file inside the unzipped folder
MODEL_PATH = "/content/test_weights/YOLO11-JDE/ifbb_jde/prague_pro_final/weights/best.pt"

# --- 6. RUN TRACKING ---
if os.path.exists(MODEL_PATH):
    print("\n--- STARTING TRACKING ---")
    # Use --model instead of --weights to match track_bodybuilders.py
    !python track_bodybuilders.py --model "{MODEL_PATH}" \
                                 --source "{TEST_VIDEO}" \
                                 --output "tracked_result.mp4" \
                                 --conf 0.3 \
                                 --imgsz 1280
else:
    print(f"[ERROR] Model file not found at: {MODEL_PATH}")

# --- 7. SAVE OUTPUT BACK TO DRIVE ---
if os.path.exists("tracked_result.mp4"):
    print("\n--- SAVING RESULT TO DRIVE ---")
    !cp "tracked_result.mp4" "/content/drive/MyDrive/YOLO11_Results/tracking_test_result.mp4"
    print(f"Success! Video saved to Drive: YOLO11_Results/tracking_test_result.mp4")
else:
    print("Could not find output video. Tracking may have failed.")
