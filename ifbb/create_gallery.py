import os
import sqlite3
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import cv2

def create_gallery(model_path, dataset_dir, db_path, output_path="athlete_gallery.pt"):
    print(f"--- Gallery Creation Debug ---")
    print(f"Model: {model_path}")
    print(f"Dataset Dir: {dataset_dir}")
    print(f"DB Path: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"ERROR: Database file not found at {db_path}")
        return

    # Load Model
    model = YOLO(model_path, task="jde")
    
    # Open DB in explicit Read-Only mode for safety
    db_uri = f"{Path(db_path).absolute().as_uri()}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    cursor = conn.cursor()
    
    # Get all athletes from DB
    cursor.execute("SELECT id, name FROM id_mapping")
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
    for img_path in tqdm(images[:1000]): # Limit for speed
        filename = img_path.name
        athlete_id = filename.split('_')[0]
        
        if athlete_id not in id_to_name:
            continue
            
        athlete_name = id_to_name[athlete_id]
        
        # Run inference to get embedding
        results = model.predict(img_path, imgsz=1280, verbose=False)
        
        if len(results) > 0 and hasattr(results[0], 'embeds') and results[0].embeds is not None:
            embed = results[0].embeds[0].cpu().numpy()
            
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
    args = parser.parse_args()
    
    create_gallery(args.model, args.dataset, args.db, args.output)
