import os
import sqlite3
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import cv2

def create_gallery(model_path, dataset_dir, db_path, output_path="athlete_gallery.pt", device=None, contest_name=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"--- Gallery Creation Debug ---")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Dataset Dir: {dataset_dir}")
    print(f"DB Path: {db_path}")
    print(f"Contest Filter: {contest_name if contest_name else 'ALL'}")
    
    if not os.path.exists(db_path):
        print(f"ERROR: Database file not found at {db_path}")
        return

    # Load Model on GPU if possible
    model = YOLO(model_path, task="jde").to(device)
    
    # Open DB in explicit Read-Only mode for safety
    db_uri = f"{Path(db_path).absolute().as_uri()}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    cursor = conn.cursor()
    
    # Get all athletes from DB
    cursor.execute("SELECT athlete_id, athlete_name FROM id_mapping")
    athletes = cursor.fetchall()
    id_to_name = {str(a[0]): a[1] for a in athletes}
    print(f"Found {len(id_to_name)} athlete mappings in DB.")
    
    gallery = {} # name -> list of embeddings
    
    # SMART SEARCH: Find the 'images' folder anywhere inside dataset_dir
    img_dir = None
    for root, dirs, files in os.walk(dataset_dir):
        if root.endswith("images") or root.endswith("train/images") or root.endswith("train\\images"):
            # Verify it has jpgs
            if any(f.endswith(".jpg") for f in files):
                img_dir = Path(root)
                break
            
    if not img_dir or not img_dir.exists():
        print(f"ERROR: Could not find an 'images' folder with .jpg files inside {dataset_dir}")
        return

    print(f"Found images at: {img_dir}")
    images = list(img_dir.glob("*.jpg"))
    print(f"Total images found: {len(images)}")
    
    # Process images and group by athlete ID (filename starts with ID_)
    for img_path in tqdm(images):
        filename = img_path.name
        athlete_id = filename.split('_')[0]
        
        if athlete_id not in id_to_name:
            continue
            
        athlete_name = id_to_name[athlete_id]

        # If we have a contest filter, only use images from that contest
        # We check if the image path contains the contest name (e.g., inside a folder named after the contest)
        if contest_name and contest_name.upper() not in img_path.as_posix().upper():
            continue
        
        # 1. Load the raw image
        img_cv = cv2.imread(str(img_path))
        if img_cv is None: continue
        
        # Run prediction on the FULL raw image (matches the tracker's perspective)
        results = model.predict(img_path, imgsz=1280, device=device, verbose=False, classes=[0])
        
        # Ensure we found at least one person and the model generated embeddings
        if len(results) > 0 and len(results[0].boxes) > 0 and hasattr(results[0], 'embeds') and results[0].embeds is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            # This is a list of ALL embeddings for every person on stage
            # Use .data to safely access raw tensor
            embeds = results[0].embeds.data.cpu().numpy() 
            
            # 2. Find the index of the LARGEST bounding box (the main athlete)
            areas = [(box[2]-box[0]) * (box[3]-box[1]) for box in boxes]
            largest_idx = np.argmax(areas)
            
            # 3. CRITICAL: Extract the exact embedding that matches that specific box!
            if largest_idx < len(embeds):
                embed = embeds[largest_idx]
                
                if athlete_name not in gallery:
                    gallery[athlete_name] = []
                gallery[athlete_name].append(embed)

    # Average embeddings for each athlete to create a robust reference
    final_gallery = {}
    for name, embeds in gallery.items():
        final_gallery[name] = np.mean(embeds, axis=0)
        
    print(f"Gallery created for {len(final_gallery)} athletes.")
    torch.save(final_gallery, output_path)
    print(f"Saved gallery to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--db", type=str, required=True)
    parser.add_argument("--output", type=str, default="athlete_gallery.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--contest", type=str, default=None, help="Filter images by contest name in path")
    args = parser.parse_args()
    
    create_gallery(args.model, args.dataset, args.db, args.output, args.device, args.contest)
