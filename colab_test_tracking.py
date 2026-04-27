# === COLAB TRACKING SCRIPT (V12 - ROBUST & SKIP VIDEO PREP) ===
import os
import sys
import subprocess
import cv2
from google.colab import drive

# --- 1. MOUNT DRIVE ---
if not os.path.exists("/content/drive"):
    drive.mount('/content/drive')

# --- 2. CONFIGURATION (UPDATE THESE!) ---
WEIGHTS_ZIP = "/content/drive/MyDrive/personal/Bodybuilding_Model_Training/results/2025_EVLS_PRAGUE_PRO/weights_results.zip"
TEST_VIDEO = "/content/drive/MyDrive/personal/Bodybuilding_Dataset/Videos/2025_EVLS_Prague/OPEN BODYBUILDING - EVLS PRAG PRO 2025 (FINALE IN 4K).mp4"
REPO_URL = "https://github.com/jonas9983/YOLO11-JDE.git"
BRANCH = "feat/multi-gpu-training" 

# PATHS IN DRIVE
DB_FILE = "/content/drive/MyDrive/personal/Bodybuilding_Model_Training/dataset_builder.db"
DATASET_IMAGES_ZIP = "/content/drive/MyDrive/personal/Bodybuilding_Dataset/ifbb_jde_dataset/2025_IFBB_EVLS_Prague_Pro.zip" 
DRIVE_GALLERY_PATH = "/content/drive/MyDrive/personal/Bodybuilding_Model_Training/athlete_gallery.pt"

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

# --- 5. PREPARE WEIGHTS & DB ---
def run_step(cmd, msg):
    print(f"\n--- {msg} ---")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout: print(result.stdout)
    if result.returncode != 0:
        print(f"\n[CRITICAL ERROR] {msg} failed!\nDetails:\n{result.stderr}")
        raise RuntimeError(f"{msg} failed.")

run_step(f'mkdir -p /content/test_weights && unzip -qo "{WEIGHTS_ZIP}" -d /content/test_weights/', "Unzipping Weights")
run_step(f'mkdir -p /content/dataset/ifbb_jde && cp "{DB_FILE}" /content/dataset/ifbb_jde/dataset_builder.db', "Copying Database")

MODEL_PATH = "/content/test_weights/YOLO11-JDE/ifbb_jde/prague_pro_final2/weights/best.pt"

# --- 6. CREATE OR LOAD ATHLETE GALLERY ---
if os.path.exists(DRIVE_GALLERY_PATH):
    print(f"\n--- LOADING EXISTING GALLERY FROM DRIVE (INSTANT) ---")
    !cp "{DRIVE_GALLERY_PATH}" "athlete_gallery.pt"
    print("Gallery loaded! Skipping image processing.")
else:
    print(f"\n--- CREATING NEW ATHLETE GALLERY (GPU Accelerated) ---")
    if not os.path.exists("/content/dataset/ifbb_jde/images"):
        run_step(f'unzip -qo "{DATASET_IMAGES_ZIP}" -d /content/dataset/ifbb_jde/', "Unzipping Training Images")
    run_step(f'python ifbb/create_gallery.py --model "{MODEL_PATH}" --dataset "/content/dataset/ifbb_jde" --db "/content/dataset/ifbb_jde/dataset_builder.db" --output "athlete_gallery.pt" --device cuda', "Building Athlete Gallery")
    !cp "athlete_gallery.pt" "{DRIVE_GALLERY_PATH}"
    print(f"Gallery saved to Drive: {DRIVE_GALLERY_PATH}")

# --- 7. CODEC FIX & EXTRACTION (GPU ACCELERATED) ---
print("\n--- PREPARING VIDEO (NVENC GPU) ---")
if os.path.exists("test_segment.mp4"):
    print("test_segment.mp4 already exists! Skipping conversion.")
else:
    cap = cv2.VideoCapture(TEST_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps < 1: fps = 25 
    print(f"Detected Video FPS: {fps}")

    START_SEC = START_FRAME / fps
    DURATION_SEC = (END_FRAME - START_FRAME) / fps

    # Using h264_nvenc to encode the 1080p output using the GPU's hardware chip
    run_step(f'ffmpeg -y -ss {START_SEC} -i "{TEST_VIDEO}" -t {DURATION_SEC} -vf "scale=1920:1080" -c:v h264_nvenc -preset p1 test_segment.mp4', "GPU Video Conversion")

# --- 8. RUN TRACKING ---
run_step(f'python track_bodybuilders.py --model "{MODEL_PATH}" --source "test_segment.mp4" --output "tracked_result.mp4" --conf 0.5 --imgsz 1280 --device 0 --gallery "athlete_gallery.pt"', "Running Tracking")

# --- 9. SAVE OUTPUT BACK TO DRIVE ---
run_step('mkdir -p "/content/drive/MyDrive/YOLO11_Results/" && cp "tracked_result.mp4" "/content/drive/MyDrive/YOLO11_Results/tracking_test_result.mp4"', "Saving to Drive")

print("\nDONE! Result is in your Drive: YOLO11_Results/tracking_test_result.mp4")
