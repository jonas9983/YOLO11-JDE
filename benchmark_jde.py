import os
import re
import time
import torch
import argparse
import numpy as np
import cv2
from ultralytics import YOLO

def extract_id(link):
    match = re.search(r"(?:/d/|id=)([a-zA-Z0-9_-]+)", link)
    return match.group(1) if match else link

def run_benchmark(model_path, source, imgsz=1280, device='cpu'):
    print(f"\n--- BENCHMARKING WITH REAL FRAMES: {model_path} on {device.upper()} ---")
    
    # 1. Handle Source (Fetch real frames)
    if "drive.google.com" in source:
        video_id = extract_id(source)
        local_video = "bench_video.mp4"
        os.system(f"python -m gdown https://drive.google.com/uc?id={video_id} -O {local_video}")
    else:
        local_video = source

    cap = cv2.VideoCapture(local_video)
    frames = []
    print("Reading 50 frames for benchmark...")
    for _ in range(50):
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    
    if not frames:
        print("[ERROR] No frames found. Check your source!")
        return

    # 2. Load Model
    model = YOLO(model_path, task='jde')
    
    # 3. Export Formats to Test
    formats = ['pytorch', 'onnx', 'openvino']
    if device == 'cuda' or torch.cuda.is_available():
        formats.append('engine') # TensorRT
        
    results = {}

    for fmt in formats:
        print(f"\nTesting format: {fmt}...")
        try:
            if fmt != 'pytorch':
                # Export model
                export_path = model.export(format=fmt, imgsz=imgsz, device=device)
                current_model = YOLO(export_path, task='jde')
            else:
                current_model = model

            # Warmup
            print("  Warming up...")
            for f in frames[:5]:
                _ = current_model.predict(f, imgsz=imgsz, device=device, verbose=False)

            # Benchmark
            print("  Measuring latency...")
            times = []
            for f in frames:
                start = time.time()
                _ = current_model.predict(f, imgsz=imgsz, device=device, verbose=False)
                times.append(time.time() - start)

            avg_latency = np.mean(times) * 1000 # ms
            fps = 1 / np.mean(times)
            results[fmt] = {"latency": avg_latency, "fps": fps}
            print(f"  Result: {avg_latency:.2f} ms | {fps:.2f} FPS")

        except Exception as e:
            print(f"  [ERROR] Failed to benchmark {fmt}: {e}")

    # 4. Final Summary
    print("\n" + "="*40)
    print(f"{'Format':<15} | {'Latency (ms)':<15} | {'FPS':<8}")
    print("-" * 43)
    for fmt, data in results.items():
        print(f"{fmt:<15} | {data['latency']:<15.2f} | {data['fps']:<8.2f}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--source", type=str, required=True, help="Video path or GDrive link")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    run_benchmark(args.model, args.source, args.imgsz, args.device)
