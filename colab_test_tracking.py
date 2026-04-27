# === COLAB TRACKING SCRIPT (V7 - FULL VIDEO OR RANGE) ===
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
BRANCH = "feat/multi-gpu-training" 

# --- FRAME RANGE SELECTION ---
# Set both to 0 to process the WHOLE video!
# Example for a specific range: 1800 to 2400
START_FRAME = 0 
END_FRAME = 0   

# --- 3. REPO SETUP ---
%cd /content
if not os.path.exists("YOLO11-JDE"):
    !git clone {REPO_URL}
%cd YOLO11-JDE
!git checkout {BRANCH}
!git pull origin {BRANCH}

os.environ["PYTHONPATH"] = f"{os.getcwd()}:{os.environ.get('PYTHONPATH', '')}"

# --- 4. DEPENDENCIES ---
print("Installing dependencies...")
!pip install -r requirements.txt --quiet
!pip install --upgrade gdown mlflow --quiet

# --- 5. PREPARE WEIGHTS ---
print("Unzipping weights...")
!mkdir -p /content/test_weights
!unzip -qo "{WEIGHTS_ZIP}" -d /content/test_weights/
MODEL_PATH = "/content/test_weights/YOLO11-JDE/ifbb_jde/prague_pro_final2/weights/best.pt"

# --- 6. CODEC FIX & EXTRACTION (WITH RESIZING) ---
print("\n--- PREPARING VIDEO ---")
# Using scale=1920:1080 to make it MUCH faster to process and decode!
if START_FRAME == 0 and END_FRAME == 0:
    print("Mode: FULL VIDEO (Resizing to 1080p and converting to H.264...)")
    !ffmpeg -y -i "{TEST_VIDEO}" -vf "scale=1920:1080" -c:v libx264 -preset ultrafast -crf 23 test_segment.mp4
else:
    print(f"Mode: RANGE (Frames {START_FRAME} to {END_FRAME})")
    FPS = 60 
    START_SEC = START_FRAME / FPS
    DURATION_SEC = (END_FRAME - START_FRAME) / FPS
    !ffmpeg -y -ss {START_SEC} -i "{TEST_VIDEO}" -t {DURATION_SEC} -vf "scale=1920:1080" -c:v libx264 -preset ultrafast -crf 23 test_segment.mp4

# --- 7. RUN TRACKING ---
print("\n--- STARTING TRACKING ---")
# Running the tracking on the 1080p version
!python track_bodybuilders.py --model "{MODEL_PATH}" \
                             --source "test_segment.mp4" \
                             --output "tracked_result.mp4" \
                             --conf 0.3 \
                             --imgsz 1280 \
                             --device 0

# --- 8. SAVE OUTPUT BACK TO DRIVE ---
print("\n--- SAVING RESULT TO DRIVE ---")
!mkdir -p "/content/drive/MyDrive/YOLO11_Results/"
if os.path.exists("tracked_result.mp4"):
    !cp "tracked_result.mp4" "/content/drive/MyDrive/YOLO11_Results/tracking_test_result.mp4"
    print(f"Success! Video saved to Drive: YOLO11_Results/tracking_test_result.mp4")
else:
    print("Error: tracked_result.mp4 was not created.")
