import os
import shutil
import zipfile
import argparse
from pathlib import Path
from tqdm import tqdm

def bundle_dataset(input_dir, output_zip_path):
    input_dir = Path(input_dir)
    output_zip_path = Path(output_zip_path)
    
    # Use local /content or /tmp for high-speed extraction
    temp_extract_dir = Path("/content/temp_bundle_extraction") if Path("/content").exists() else Path("/tmp/temp_bundle_extraction")
    
    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)
    temp_extract_dir.mkdir(parents=True)
    
    print(f"\n--- DATASET BUNDLER ---")
    print(f"Input Directory: {input_dir.absolute()}")
    print(f"Temp Extraction: {temp_extract_dir.absolute()}")
    print(f"Final Master ZIP: {output_zip_path.absolute()}")
    print(f"-----------------------\n")
    
    zip_files = list(input_dir.glob("*.zip"))
    print(f"Found {len(zip_files)} contest ZIPs to bundle.")
    
    # 1. UNZIP ALL TO LOCAL STORAGE
    for z in tqdm(zip_files, desc="Extracting ZIPs locally"):
        try:
            with zipfile.ZipFile(z, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_dir)
        except Exception as e:
            print(f"  [ERROR] Failed to extract {z.name}: {e}")
            
    # 2. CREATE MASTER ZIP
    print(f"\nCreating Master ZIP (this may take a while)...")
    local_master_zip = Path("/tmp/bodybuilding_jde_master")
    # make_archive appends .zip automatically
    shutil.make_archive(str(local_master_zip), 'zip', temp_extract_dir)
    
    # 3. MOVE TO DRIVE
    print(f"Moving Master ZIP to Drive...")
    shutil.move(str(local_master_zip.with_suffix('.zip')), output_zip_path)
    
    # 4. CLEANUP
    shutil.rmtree(temp_extract_dir)
    print(f"\n[SUCCESS] Master ZIP created and saved to {output_zip_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder containing individual contest ZIPs")
    parser.add_argument("--output", required=True, help="Path to save the final master .zip file")
    args = parser.parse_args()
    
    bundle_dataset(args.input, args.output)
