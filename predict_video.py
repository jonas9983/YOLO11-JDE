import os
import cv2
import argparse
from ultralytics import YOLO
from tqdm import tqdm

def run_tracking(model_path, video_path, output_path, imgsz=1280, conf=0.25):
    # 1. Load the trained JDE model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path, task="jde")

    # 2. Open Video
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 3. Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing {total_frames} frames...")
    
    # 4. Tracking Loop
    with tqdm(total=total_frames, desc="Tracking Bodybuilders") as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Run tracking
            # Using 'jdetracker.yaml' which is optimized for your trained JDE embeddings
            results = model.track(
                source=frame,
                imgsz=imgsz,
                conf=conf,
                persist=True,
                tracker="jdetracker.yaml",
                verbose=False
            )

            # Get annotated frame
            annotated_frame = results[0].plot()

            # Write to output file
            out.write(annotated_frame)
            pbar.update(1)

    cap.release()
    out.release()
    print(f"\n--- SUCCESS ---")
    print(f"Tracked video saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default="tracked_output.mp4")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    
    args = parser.parse_args()
    run_tracking(args.model, args.video, args.output, args.imgsz, args.conf)
