import os
import sqlite3
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import cv2

def create_gallery(model_path, dataset_dir, db_path, output_path="athlete_gallery.pt", device=None, contest_name=None, imgsz=960, division=None, max_poses=50, npc_db_path=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"--- Gallery Creation Debug ---")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Dataset Dir: {dataset_dir}")
    print(f"DB Path: {db_path}")
    print(f"NPC DB Path: {npc_db_path if npc_db_path else 'Not provided (division filter disabled)'}")
    print(f"Contest Filter: {contest_name if contest_name else 'ALL'}")
    print(f"Division Filter: {division if division else 'ALL'}")
    print(f"Image Size: {imgsz}")
    print(f"Max Poses per Athlete: {max_poses}")
    
    if not os.path.exists(db_path):
        print(f"ERROR: Database file not found at {db_path}")
        return

    # Load Model on GPU if possible
    model = YOLO(model_path, task="jde").to(device)
    
    # Open DB in explicit Read-Only mode for safety
    db_uri = f"{Path(db_path).absolute().as_uri()}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    cursor = conn.cursor()
    
    # --- DIVISION ALLOWLIST: Query npc_database.db if provided ---
    division_allowlist = None  # None means no filtering
    if division and npc_db_path and os.path.exists(npc_db_path):
        npc_uri = f"{Path(npc_db_path).absolute().as_uri()}?mode=ro"
        npc_conn = sqlite3.connect(npc_uri, uri=True)
        npc_cursor = npc_conn.cursor()
        # Get all athlete names in this division (and optional contest) from the rich NPC DB
        if contest_name:
            npc_cursor.execute(
                "SELECT DISTINCT athlete_name FROM athletes WHERE division LIKE ? AND contest_name LIKE ? "
                "AND athlete_name NOT LIKE '%COMPARISON%' AND athlete_name NOT LIKE '%AWARD%' AND athlete_name NOT LIKE '%OVERALL%'",
                (f"%{division}%", f"%{contest_name}%")
            )
        else:
            npc_cursor.execute(
                "SELECT DISTINCT athlete_name FROM athletes WHERE division LIKE ? "
                "AND athlete_name NOT LIKE '%COMPARISON%' AND athlete_name NOT LIKE '%AWARD%' AND athlete_name NOT LIKE '%OVERALL%'",
                (f"%{division}%",)
            )
        division_allowlist = {row[0].upper() for row in npc_cursor.fetchall()}
        npc_conn.close()
        print(f"Division allowlist loaded: {len(division_allowlist)} athletes from NPC DB for division '{division}'.")
    elif division and not npc_db_path:
        print(f"WARNING: --division set but --npc_db not provided. Division filter will be skipped.")
        print(f"         Pass --npc_db /path/to/npc_database.db to enable division filtering.")

    # Open dataset_builder.db for ID->name mapping
    db_uri = f"{Path(db_path).absolute().as_uri()}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    cursor = conn.cursor()
    cursor.execute("SELECT athlete_id, athlete_name FROM id_mapping")
    athletes = cursor.fetchall()
    conn.close()
    
    # Apply division allowlist filter if we have one
    id_to_name = {}
    for athlete_id, athlete_name in athletes:
        if division_allowlist is None or athlete_name.upper() in division_allowlist:
            id_to_name[str(athlete_id)] = athlete_name
    print(f"Found {len(id_to_name)} athlete mappings after filtering.")
    
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
    
    # --- 1. GROUP FILES FIRST (NO AI YET) ---
    import random
    athlete_files = {}
    
    for img_path in images:
        filename = img_path.name
        athlete_id = filename.split('_')[0]
        
        if athlete_id in id_to_name:
            athlete_name = id_to_name[athlete_id]
            
            # Contest filter
            if contest_name and contest_name.upper() not in img_path.as_posix().upper():
                continue
                
            if athlete_name not in athlete_files:
                athlete_files[athlete_name] = []
            athlete_files[athlete_name].append(img_path)

    # --- 2. FAST AI PROCESSING ---
    final_gallery = {}
    print(f"Processing {len(athlete_files)} athletes...")
    
    for name, paths in tqdm(athlete_files.items(), desc="Athletes"):
        # Shuffle to ensure we get diverse poses (front, back, side)
        random.shuffle(paths) 
        final_gallery[name] = []
        
        for img_path in paths:
            # Stop as soon as we get the required number of good embeddings
            if len(final_gallery[name]) >= max_poses:
                break
                
            # Load the raw image
            img_cv = cv2.imread(str(img_path))
            if img_cv is None: continue
            
            # Run prediction on the FULL raw image
            results = model.predict(img_path, imgsz=imgsz, device=device, verbose=False, classes=[0])
            
            # Ensure we found at least one person and the model generated embeddings
            if len(results) > 0 and len(results[0].boxes) > 0 and hasattr(results[0], 'embeds') and results[0].embeds is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                embeds = results[0].embeds.data.cpu().numpy() 
                
                # Find the index of the LARGEST bounding box (the main athlete)
                areas = [(box[2]-box[0]) * (box[3]-box[1]) for box in boxes]
                largest_idx = np.argmax(areas)
                
                if largest_idx < len(embeds):
                    embed = embeds[largest_idx]
                    # Normalize and save immediately
                    embed = embed / (np.linalg.norm(embed) + 1e-6)
                    final_gallery[name].append(embed)

    print(f"Gallery created for {len(final_gallery)} athletes.")
    torch.save(final_gallery, output_path)
    print(f"Saved multi-pose gallery to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--db", type=str, required=True)
    parser.add_argument("--npc_db", type=str, default=None, help="Path to npc_database.db for division-aware filtering")
    parser.add_argument("--output", type=str, default="athlete_gallery.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--contest", type=str, default=None, help="Filter images by contest name in path")
    parser.add_argument("--db-contest", type=str, default=None, help="Filter athletes by contest name in NPC DB")
    parser.add_argument("--division", type=str, default=None, help="Filter athletes by division name (requires --npc_db)")
    parser.add_argument("--imgsz", type=int, default=960, help="Image size for model inference")
    parser.add_argument("--max-poses", type=int, default=50, help="Maximum number of embeddings to extract per athlete")
    args = parser.parse_args()
    
    create_gallery(args.model, args.dataset, args.db, args.output, args.device, args.contest, args.imgsz, args.division, args.max_poses, args.npc_db, args.db_contest)

    parser.add_argument("--max-poses", type=int, default=50, help="Maximum number of embeddings to extract per athlete")
    args = parser.parse_args()
    
    create_gallery(args.model, args.dataset, args.db, args.output, args.device, args.contest, args.imgsz, args.division, args.max_poses, args.npc_db)
