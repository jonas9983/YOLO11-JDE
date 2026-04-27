import os
import re
import cv2
import argparse
import numpy as np
import torch
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

def extract_id(link):
    match = re.search(r"(?:/d/|id=)([a-zA-Z0-9_-]+)", link)
    return match.group(1) if match else link

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def run_tracking(model_path, source, output_path, imgsz=1280, conf=0.25, device=0, start_frame=0, end_frame=None, gallery_path=None):
    # 1. Handle Source
    is_drive = "drive.google.com" in source or len(source) == 33
    if is_drive:
        video_id = extract_id(source)
        local_input = "input_video.mp4"
        if os.path.exists(local_input): os.remove(local_input)
        os.system(f"python -m gdown https://drive.google.com/uc?id={video_id} -O {local_input}")
    else:
        local_input = source

    # 2. Load Model and Gallery
    print(f"Loading model: {model_path} on device: {device}")
    model = YOLO(model_path, task="jde")
    
    gallery = None
    if gallery_path and os.path.exists(gallery_path):
        print(f"Loading athlete gallery: {gallery_path}")
        gallery = torch.load(gallery_path)
    
    track_to_athlete = {} # track_id -> athlete_name
    tracking_logs = [] # To save as CSV

    # 3. Process Video
    cap = cv2.VideoCapture(local_input)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    if end_frame is None or end_frame == 0 or end_frame > total_video_frames:
        end_frame = total_video_frames
    
    num_to_process = end_frame - start_frame
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Tracking bodybuilders...")
    
    count = 0
    # mininterval=2.0 prevents tqdm from flooding the Colab console
    with tqdm(total=num_to_process, mininterval=2.0) as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            if not success or (count >= num_to_process): 
                break

            results = model.track(
                source=frame, 
                imgsz=imgsz, 
                conf=conf, 
                persist=True, 
                tracker="jdetracker.yaml", 
                device=device,
                verbose=False
            )
            
            if len(results) > 0:
                result = results[0]
                frame_num = start_frame + count
                
                if hasattr(result, 'boxes') and result.boxes.id is not None:
                    ids = result.boxes.id.cpu().numpy().astype(int)
                    boxes = result.boxes.xyxy.cpu().numpy()
                    
                    # Fetch embeddings directly from the tracker's active state
                    tracker = getattr(model.predictor, 'trackers', [None])[0]
                    track_embeds = {}
                    if tracker and hasattr(tracker, 'tracked_stracks'):
                        for t in tracker.tracked_stracks + getattr(tracker, 'lost_stracks', []):
                            feat = getattr(t, 'smooth_feat', getattr(t, 'curr_feat', getattr(t, 'features', None)))
                            if feat is not None:
                                # Ensure it's a numpy array, sometimes it's a deque of features
                                if isinstance(feat, np.ndarray):
                                    track_embeds[t.track_id] = feat
                                elif isinstance(feat, list) and len(feat) > 0:
                                    track_embeds[t.track_id] = feat[-1]
                    
                    for i, track_id in enumerate(ids):
                        best_name = "Unknown"
                        best_sim = 0.0
                        
                        if track_id in track_embeds and gallery:
                            embed = track_embeds[track_id]
                            for athlete_name, ref_embed in gallery.items():
                                sim = cosine_similarity(embed, ref_embed)
                                if sim > best_sim:
                                    best_sim = sim
                                    best_name = athlete_name
                            
                            # Lowered threshold for labeling visibility
                            if best_sim > 0.25:
                                track_to_athlete[track_id] = f"{best_name}"
                            else:
                                if track_id not in track_to_athlete:
                                    track_to_athlete[track_id] = f"ID:{track_id}"
                            
                            # Log data for CSV
                            tracking_logs.append({
                                "frame": frame_num,
                                "track_id": track_id,
                                "best_match": best_name,
                                "similarity": round(float(best_sim), 4),
                                "bbox": boxes[i].tolist()
                            })
                        else:
                            # If no embedding found, just ensure the ID is shown
                            if track_id not in track_to_athlete:
                                track_to_athlete[track_id] = f"ID:{track_id}"

                # Annotation
                annotated_frame = frame.copy()
                if hasattr(result, 'boxes') and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    ids = result.boxes.id.cpu().numpy().astype(int)
                    
                    for i, box in enumerate(boxes):
                        track_id = ids[i]
                        name = track_to_athlete.get(track_id, f"ID:{track_id}")
                        label = f"{name} ({track_id})"
                        
                        cv2.rectangle(annotated_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, label, (int(box[0]), int(box[1]) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                out.write(annotated_frame)
            else:
                out.write(frame)
                
            pbar.update(1)
            count += 1

    cap.release()
    out.release()
    
    # Save logs to CSV
    log_path = output_path.replace(".mp4", "_debug_log.csv")
    if tracking_logs:
        df = pd.DataFrame(tracking_logs)
        df.to_csv(log_path, index=False)
        print(f"\n[DEBUG] Debug log saved to {log_path} with {len(tracking_logs)} entries.")
    else:
        # Create an empty CSV so we know the script tried to save it
        pd.DataFrame(columns=["frame", "track_id", "best_match", "similarity", "bbox"]).to_csv(log_path, index=False)
        print(f"\n[WARNING] No tracking logs were generated! Empty CSV saved to {log_path}.")
    
    print(f"Done! Saved result to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output", type=str, default="tracked_result.mp4")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--gallery", type=str, default=None)
    
    args = parser.parse_args()
    run_tracking(args.model, args.source, args.output, args.imgsz, args.conf, args.device, args.start_frame, args.end_frame, args.gallery)
