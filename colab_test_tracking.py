# === COLAB TRACKING SCRIPT (V16.1 - FIXED PATHS) ===
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
# Corrected path to match user's Drive structure
DATASET_IMAGES_ZIP = "/content/drive/MyDrive/personal/Bodybuilding_Model_Training/ifbb_jde_dataset/2025_IFBB_EVLS_Prague_Pro.zip" 

# !!! CONTEST FILTERING !!!
# limits the search to only these athletes to prevent gender-mismatch.
CONTEST_FILTER = "Prague_Pro" 

# Set this to True if you want to rebuild the gallery from scratch
FORCE_REBUILD_GALLERY = True

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
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None: break
        if output:
            print(output.strip())
            sys.stdout.flush()
    if process.returncode != 0:
        raise RuntimeError(f"{msg} failed.")

run_step(f'mkdir -p /content/test_weights && unzip -qo "{WEIGHTS_ZIP}" -d /content/test_weights/', "Unzipping Weights")
run_step(f'mkdir -p /content/dataset/ifbb_jde && cp "{DB_FILE}" /content/dataset/ifbb_jde/dataset_builder.db', "Copying Database")

MODEL_PATH = "/content/test_weights/YOLO11-JDE/ifbb_jde/prague_pro_final2/weights/best.pt"

# --- 6. CREATE OR LOAD ATHLETE GALLERY ---
GALLERY_FILE = "athlete_gallery.pt"
if os.path.exists(GALLERY_FILE) and not FORCE_REBUILD_GALLERY:
    print(f"\n--- REUSING LOCAL GALLERY ---")
else:
    print(f"\n--- BUILDING FILTERED ATHLETE GALLERY (Contest: {CONTEST_FILTER}) ---")
    if not os.path.exists("/content/dataset/ifbb_jde/images"):
        run_step(f'unzip -qo "{DATASET_IMAGES_ZIP}" -d /content/dataset/ifbb_jde/', "Unzipping Training Images")
    
    filter_cmd = f"--contest {CONTEST_FILTER}" if CONTEST_FILTER else ""
    run_step(f'python ifbb/create_gallery.py --model "{MODEL_PATH}" --dataset "/content/dataset/ifbb_jde" --db "/content/dataset/ifbb_jde/dataset_builder.db" --output "{GALLERY_FILE}" --device cuda {filter_cmd}', "Building Athlete Gallery")

# --- 7. CODEC FIX & EXTRACTION ---
LOCAL_VIDEO = f"segment_{START_FRAME}_{END_FRAME}.mp4"
if not os.path.exists(LOCAL_VIDEO):
    print(f"\n--- PREPARING VIDEO (NVENC GPU) ---")
    cap = cv2.VideoCapture(TEST_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps < 1: fps = 25 
    START_SEC = START_FRAME / fps
    DURATION_SEC = (END_FRAME - START_FRAME) / fps
    run_step(f'ffmpeg -y -ss {START_SEC} -i "{TEST_VIDEO}" -t {DURATION_SEC} -vf "scale=1920:1080" -c:v h264_nvenc -preset p1 {LOCAL_VIDEO}', "GPU Video Conversion")

# --- 8. RUN TRACKING ---
run_step(f'python track_bodybuilders.py --model "{MODEL_PATH}" --source "{LOCAL_VIDEO}" --output "tracked_result.mp4" --conf 0.5 --imgsz 1280 --device 0 --gallery "{GALLERY_FILE}" --start-frame 0', "Running Tracking")

# --- 9. SAVE OUTPUT BACK TO DRIVE ---
OUTPUT_NAME = f"tracking_{START_FRAME}_{END_FRAME}"
run_step(f'mkdir -p "/content/drive/MyDrive/YOLO11_Results/" && cp "tracked_result.mp4" "/content/drive/MyDrive/YOLO11_Results/{OUTPUT_NAME}.mp4"', "Saving Video to Drive")
if os.path.exists("tracked_result_debug_log.csv"):
    run_step(f'cp "tracked_result_debug_log.csv" "/content/drive/MyDrive/YOLO11_Results/{OUTPUT_NAME}_debug.csv"', "Saving Debug CSV to Drive")

print(f"\nDONE! Results saved as {OUTPUT_NAME} in YOLO11_Results folder.")
