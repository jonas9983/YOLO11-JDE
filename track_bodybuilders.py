import os
import re
import cv2
import argparse
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

def extract_id(link):
    match = re.search(r"(?:/d/|id=)([a-zA-Z0-9_-]+)", link)
    return match.group(1) if match else link

def run_tracking(model_path, source, output_path, imgsz=1280, conf=0.25, device=0, start_frame=0, end_frame=None):
    # 1. Handle Source (Drive vs Local)
    is_drive = "drive.google.com" in source or len(source) == 33 # Likely an ID
    
    if is_drive:
        video_id = extract_id(source)
        print(f"Downloading video from Drive (ID: {video_id})...")
        local_input = "input_video.mp4"
        if os.path.exists(local_input): os.remove(local_input)
        os.system(f"python -m gdown https://drive.google.com/uc?id={video_id} -O {local_input}")
    else:
        local_input = source
        if not os.path.exists(local_input):
            print(f"[ERROR] File not found: {local_input}")
            return

    # 2. Load Model
    print(f"Loading model: {model_path} on device: {device}")
    model = YOLO(model_path, task="jde")

    # 3. Process Video
    cap = cv2.VideoCapture(local_input)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Handle Frame Range
    if start_frame > 0:
        print(f"Seeking to frame {start_frame}...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    if end_frame is None or end_frame > total_video_frames:
        end_frame = total_video_frames
    
    num_to_process = end_frame - start_frame
    if num_to_process <= 0:
        print(f"[ERROR] Invalid range: start={start_frame}, end={end_frame}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Tracking bodybuilders from frame {start_frame} to {end_frame} ({num_to_process} frames)...")
    
    count = 0
    with tqdm(total=num_to_process) as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            if not success or (count >= num_to_process): 
                break

            # Track using the JDE embeddings
            results = model.track(
                source=frame, 
                imgsz=imgsz, 
                conf=conf, 
                persist=True, 
                tracker="jdetracker.yaml", 
                device=device,
                verbose=False
            )
            
            # This will plot class names (e.g., 'Cbum') + ID if the model was trained with them
            annotated_frame = results[0].plot(labels=True, conf=True) 
            out.write(annotated_frame)
            pbar.update(1)
            count += 1

    cap.release()
    out.release()
    print(f"Done! Saved result to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to weights (best.pt)")
    parser.add_argument("--source", type=str, required=True, help="Local path OR GDrive link")
    parser.add_argument("--output", type=str, default="tracked_result.mp4")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame to start from")
    parser.add_argument("--end-frame", type=int, default=None, help="Frame to end at")
    
    args = parser.parse_args()
    run_tracking(args.model, args.source, args.output, args.imgsz, args.conf, args.device, args.start_frame, args.end_frame)
