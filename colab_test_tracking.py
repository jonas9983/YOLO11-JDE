# === COLAB TRACKING SCRIPT (V4 - GPU & LIMITS) ===
# 1. Select 'T4 GPU' Accelerator
# 2. Mount your Google Drive
# 3. Paste, update PATHS, and run!

import os
from google.colab import drive

# --- 1. MOUNT DRIVE ---
if not os.path.exists("/content/drive"):
    drive.mount('/content/drive')

# --- 2. CONFIGURATION (UPDATE THESE!) ---
WEIGHTS_ZIP = "/content/drive/MyDrive/personal/Bodybuilding_Model_Training/results/2025_EVLS_PRAGUE_PRO/weights_results.zip"
TEST_VIDEO = "/content/drive/MyDrive/personal/Bodybuilding_Dataset/Videos/2025_EVLS_Prague/OPEN BODYBUILDING - EVLS PRAG PRO 2025 (FINALE IN 4K).mp4"
REPO_URL = "https://github.com/jonas9983/YOLO11-JDE.git"
BRANCH = "feat/multi-gpu-training" # Updated to your current branch!

# --- 3. REPO SETUP ---
%cd /content
if not os.path.exists("YOLO11-JDE"):
    !git clone {REPO_URL}
%cd YOLO11-JDE
!git checkout {BRANCH}
!git pull origin {BRANCH}

os.environ["PYTHONPATH"] = f"{os.getcwd()}:{os.environ.get('PYTHONPATH', '')}"

# --- 4. DEPENDENCIES ---
!pip install -r requirements.txt --quiet
!pip install --upgrade gdown mlflow --quiet

# --- 5. PREPARE WEIGHTS ---
print("Unzipping weights...")
!mkdir -p /content/test_weights
!unzip -qo "{WEIGHTS_ZIP}" -d /content/test_weights/

MODEL_PATH = "/content/test_weights/YOLO11-JDE/ifbb_jde/prague_pro_final2/weights/best.pt"

# --- 6. RUN TRACKING ---
print("\n--- STARTING TRACKING ---")
# Added --device 0 for GPU and --frames 500 for a quick test!
!python track_bodybuilders.py --model "{MODEL_PATH}" \
                             --source "{TEST_VIDEO}" \
                             --output "tracked_result.mp4" \
                             --conf 0.3 \
                             --imgsz 1280 \
                             --device 0 \
                             --frames 500

# --- 7. SAVE OUTPUT BACK TO DRIVE ---
print("\n--- SAVING RESULT TO DRIVE ---")
# Ensure the directory exists in Drive
!mkdir -p "/content/drive/MyDrive/YOLO11_Results/"

if os.path.exists("tracked_result.mp4"):
    !cp "tracked_result.mp4" "/content/drive/MyDrive/YOLO11_Results/tracking_test_result.mp4"
    print(f"Success! Video saved to Drive: YOLO11_Results/tracking_test_result.mp4")
else:
    print("Error: tracked_result.mp4 was not created. Check for errors in the tracking logs above.")
