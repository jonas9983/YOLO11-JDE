# === COLAB TRACKING SCRIPT (V18 - DIRECT WEIGHTS & REBUILD GALLERY) ===
import os
import sys
import subprocess
import cv2
from google.colab import drive

# --- 1. MOUNT DRIVE ---
if not os.path.exists("/content/drive"):
    drive.mount('/content/drive')

# --- 2. CONFIGURATION (UPDATE THESE!) ---
# Point directly to your best.pt file in Google Drive!
WEIGHTS_PT = "/content/drive/MyDrive/iron_insights/Bodybuilding_Model_Training/results/3_divisions_incomplete/weights/best.pt"
TEST_VIDEO = "/content/drive/MyDrive/iron_insights/Bodybuilding_Dataset/Videos/2025_EVLS_Prague/OPEN BODYBUILDING - EVLS PRAG PRO 2025 (FINALE IN 4K).mp4"
REPO_URL = "https://github.com/jonas9983/YOLO11-JDE.git"
BRANCH = "feat/recognition-debug" 

# PATHS IN DRIVE
DRIVE_BASE = "/content/drive/MyDrive/iron_insights/Bodybuilding_Model_Training"
DB_FILE = f"{DRIVE_BASE}/ifbb_jde_dataset_clean/dataset_id_mapping.db"
# You need the images zip of the competition to rebuild the gallery.
# Note: If you want to build the gallery using images from ALL competitions,
# you should unzip the complete dataset here instead of just one competition.
DATASET_IMAGES_ZIP = f"{DRIVE_BASE}/ifbb_jde_dataset_clean/2025_IFBB_EVLS_Prague_Pro.zip" 
DRIVE_GALLERY_PATH = f"{DRIVE_BASE}/athlete_gallery.pt"

# --- PERSISTENT SEGMENTS ---
# Store segments in Drive so they survive session restarts!
SEGMENT_DIR = "/content/drive/MyDrive/iron_insights/tracking_results/segments"
os.makedirs(SEGMENT_DIR, exist_ok=True)

# Set to empty string "" to build the gallery with athletes from ALL competitions
# in the dataset. Otherwise, it will only use images with this name in the path.
CONTEST_FILTER = "" 

# Set to True because your gallery is outdated and needs to be rebuilt with the new model
FORCE_REBUILD_GALLERY = True

# --- FRAME RANGE SELECTION ---
START_FRAME = 2000 
END_FRAME = 15000
IMGSZ = 960

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
def run_step(cmd, msg, quiet=False):
    print(f"\n--- {msg} ---")
    if quiet:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(f"{msg} failed.")
        return

    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None: break
        if output:
            # Only print if it's not a tqdm carriage return to keep it clean
            if not output.startswith('\r'):
                print(output.strip())
    if process.returncode != 0:
        raise RuntimeError(f"{msg} failed.")

run_step(f'mkdir -p /content/dataset/ifbb_jde && cp "{DB_FILE}" /content/dataset/ifbb_jde/dataset_builder.db', "Copying Database", quiet=True)

MODEL_PATH = WEIGHTS_PT

# --- 6. CREATE OR LOAD ATHLETE GALLERY ---
if os.path.exists(DRIVE_GALLERY_PATH) and not FORCE_REBUILD_GALLERY:
    print(f"\n--- LOADING PERSISTENT GALLERY FROM DRIVE (INSTANT) ---")
    !cp "{DRIVE_GALLERY_PATH}" "athlete_gallery.pt"
else:
    print(f"\n--- BUILDING FILTERED GALLERY ---")
    if not os.path.exists("/content/dataset/ifbb_jde/images"):
        run_step(f'unzip -qo "{DATASET_IMAGES_ZIP}" -d /content/dataset/ifbb_jde/', "Unzipping Training Images", quiet=True)
    
    filter_cmd = f"--contest '{CONTEST_FILTER}'" if CONTEST_FILTER else ""
    run_step(f'python ifbb/create_gallery.py --model "{MODEL_PATH}" --dataset "/content/dataset/ifbb_jde" --db "/content/dataset/ifbb_jde/dataset_builder.db" --output "athlete_gallery.pt" --device cuda --imgsz {IMGSZ} --division "MEN\'S BODYBUILDING" --max-poses 50 {filter_cmd}', "Creating Gallery")
    !cp "athlete_gallery.pt" "{DRIVE_GALLERY_PATH}"

# --- 7. PERSISTENT VIDEO EXTRACTION ---
LOCAL_VIDEO = f"{SEGMENT_DIR}/segment_{START_FRAME}_{END_FRAME}.mp4"

if os.path.exists(LOCAL_VIDEO):
    print(f"\n--- REUSING EXISTING SEGMENT FROM DRIVE: {os.path.basename(LOCAL_VIDEO)} ---")
else:
    print(f"\n--- PREPARING NEW VIDEO SEGMENT (NVENC GPU) ---")
    cap = cv2.VideoCapture(TEST_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps < 1: fps = 25 
    
    START_SEC = START_FRAME / fps
    DURATION_SEC = (END_FRAME - START_FRAME) / fps
    run_step(f'ffmpeg -y -ss {START_SEC} -i "{TEST_VIDEO}" -t {DURATION_SEC} -vf "scale=1920:1080" -c:v h264_nvenc -preset p1 "{LOCAL_VIDEO}"', "GPU Video Conversion")

# --- 8. RUN TRACKING ---
run_step(f'python ifbb/track_bodybuilders.py --model "{MODEL_PATH}" --source "{LOCAL_VIDEO}" --output "tracked_result.mp4" --conf 0.5 --imgsz {IMGSZ} --device 0 --gallery "athlete_gallery.pt" --start-frame 0 --half', "Running Tracking")

# --- 9. SAVE OUTPUT BACK TO DRIVE ---
OUTPUT_NAME = f"tracking_{START_FRAME}_{END_FRAME}"
FINAL_SAVE_DIR = f"{DRIVE_BASE}/tracking_results"
run_step(f'mkdir -p "{FINAL_SAVE_DIR}" && cp "tracked_result.mp4" "{FINAL_SAVE_DIR}/{OUTPUT_NAME}.mp4"', "Saving Result", quiet=True)
if os.path.exists("tracked_result_debug_log.csv"):
    !cp "tracked_result_debug_log.csv" "{FINAL_SAVE_DIR}/{OUTPUT_NAME}_debug.csv"

print(f"\nDONE! Results are in your Drive folder: {FINAL_SAVE_DIR}")
