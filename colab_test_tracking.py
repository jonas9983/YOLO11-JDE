# === COLAB TRACKING SCRIPT (V9.5 - EXPLICIT ERROR CAPTURE) ===
import os
import sys
import subprocess
from google.colab import drive

# --- 1. MOUNT DRIVE ---
if not os.path.exists("/content/drive"):
    drive.mount('/content/drive')

# --- 2. CONFIGURATION (UPDATE THESE!) ---
WEIGHTS_ZIP = "/content/drive/MyDrive/personal/Bodybuilding_Model_Training/results/2025_EVLS_PRAGUE_PRO/weights_results.zip"
TEST_VIDEO = "/content/drive/MyDrive/personal/Bodybuilding_Dataset/Videos/2025_EVLS_Prague/OPEN BODYBUILDING - EVLS PRAG PRO 2025 (FINALE IN 4K).mp4"
REPO_URL = "https://github.com/jonas9983/YOLO11-JDE.git"
BRANCH = "feat/multi-gpu-training" 

# THE DB FILE - DOUBLE CHECK THIS FILENAME!
DB_FILE = "/content/drive/MyDrive/personal/Bodybuilding_Model_Training/dataset_builder.db"
# THE TRAINING IMAGES
DATASET_IMAGES_ZIP = "/content/drive/MyDrive/personal/Bodybuilding_Model_Training/ifbb_jde_dataset/2025_IFBB_EVLS_Prague_Pro.zip" 

# --- FRAME RANGE SELECTION ---
START_FRAME = 2000 
END_FRAME = 15000

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
!pip install --upgrade gdown mlflow pytorch-metric-learning --quiet

# --- 5. PREPARE WEIGHTS, DB & DATASET ---
def run_step(cmd, msg):
    print(f"\n--- {msg} ---")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Print the standard output so we can see normal progress
    if result.stdout:
        print(result.stdout)
        
    if result.returncode != 0:
        print(f"\n[CRITICAL ERROR] {msg} failed!")
        print(f"Here is the exact error from the script:\n")
        print("================ ERROR DETAILS ================")
        print(result.stderr)
        print("===============================================")
        raise RuntimeError(f"{msg} failed. See error details above.")

run_step(f'mkdir -p /content/test_weights && unzip -qo "{WEIGHTS_ZIP}" -d /content/test_weights/', "Unzipping Weights")

# Verification before copy
if not os.path.exists(DB_FILE):
    print(f"\n[ERROR] DB_FILE not found at: {DB_FILE}")
    sys.exit(1)

run_step(f'mkdir -p /content/dataset/ifbb_jde && cp "{DB_FILE}" /content/dataset/ifbb_jde/dataset_builder.db', "Copying Database")

# Verification before unzip
if not os.path.exists(DATASET_IMAGES_ZIP):
    print(f"\n[ERROR] DATASET_IMAGES_ZIP not found at: {DATASET_IMAGES_ZIP}")
    sys.exit(1)

if not os.path.exists("/content/dataset/ifbb_jde/images"):
    run_step(f'unzip -qo "{DATASET_IMAGES_ZIP}" -d /content/dataset/ifbb_jde/', "Unzipping Training Images")

MODEL_PATH = "/content/test_weights/YOLO11-JDE/ifbb_jde/prague_pro_final2/weights/best.pt"

# --- 6. CREATE ATHLETE GALLERY ---
run_step(f'python ifbb/create_gallery.py --model "{MODEL_PATH}" --dataset "/content/dataset/ifbb_jde" --db "/content/dataset/ifbb_jde/dataset_builder.db" --output "athlete_gallery.pt"', "Creating Athlete Gallery")

# --- 7. CODEC FIX & EXTRACTION ---
print("\n--- PREPARING VIDEO ---")
FPS = 60 
START_SEC = START_FRAME / FPS
DURATION_SEC = (END_FRAME - START_FRAME) / FPS
run_step(f'ffmpeg -y -ss {START_SEC} -i "{TEST_VIDEO}" -t {DURATION_SEC} -vf "scale=1920:1080" -c:v libx264 -preset ultrafast -crf 23 test_segment.mp4', "Converting Video to 1080p")

# --- 8. RUN TRACKING ---
run_step(f'python track_bodybuilders.py --model "{MODEL_PATH}" --source "test_segment.mp4" --output "tracked_result.mp4" --conf 0.5 --imgsz 1280 --device 0 --gallery "athlete_gallery.pt"', "Running Tracking")

# --- 9. SAVE OUTPUT BACK TO DRIVE ---
run_step('mkdir -p "/content/drive/MyDrive/YOLO11_Results/" && cp "tracked_result.mp4" "/content/drive/MyDrive/YOLO11_Results/tracking_test_result.mp4"', "Saving to Drive")

print("\nDONE! Result is in your Drive: YOLO11_Results/tracking_test_result.mp4")
