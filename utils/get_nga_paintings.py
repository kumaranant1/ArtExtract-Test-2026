import os
import argparse
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import concurrent.futures
from tqdm import tqdm
from functools import partial

def download_image(row, output_dir):
    """Downloads a single image from an IIIF URL and saves it."""
    obj_id = row['objectid']
    iiif_url = row['iiifurl']
    
    if pd.isna(iiif_url):
        return False
        
    save_path = os.path.join(output_dir, f"{obj_id}.jpg")
    
    if os.path.exists(save_path):
        return True

    try:
        # NGA IIIF format: Append '/full/!512,512/0/default.jpg' for a smaller version of the painting 
        # this is to avoid downloading very large images which may not be necessary for us.
        fetch_url = f"{iiif_url}/full/!512,512/0/default.jpg" if not iiif_url.endswith('.jpg') else iiif_url
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(fetch_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img.save(save_path)
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to download {obj_id}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download paintings from the NGA Dataset.")
    parser.add_argument('--objects_csv', type=str, required=True, help="Path to objects.csv")
    parser.add_argument('--images_csv', type=str, required=True, help="Path to published_images.csv")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save downloaded images")
    parser.add_argument('--sample_size', type=int, default=10000, help="Number of random paintings to download")
    parser.add_argument('--max_threads', type=int, default=30, help="Maximum number of parallel downloads")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[INFO] Loading NGA Metadata...")
    try:
        df_objects = pd.read_csv(args.objects_csv, low_memory=False)
        df_images = pd.read_csv(args.images_csv, low_memory=False)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("Please verify the provided CSV paths.")
        return

    print("[INFO] Merging Object data with Image URLs...")
    df = pd.merge(df_objects, df_images, left_on='objectid', right_on='depictstmsobjectid')

    if 'classification' in df.columns:
        df = df[df['classification'].str.contains('painting', case=False, na=False)]

    df = df.dropna(subset=['iiifurl'])
    print(f"[INFO] Found {len(df)} total paintings with valid image URLs.")
    
    print(f"[INFO] Sampling {args.sample_size} random paintings...")
    sample_df = df.sample(n=min(args.sample_size, len(df)), random_state=42)

    rows_to_download = sample_df.to_dict('records')

    print(f"[INFO] Starting parallel download with {args.max_threads} threads...")
    
    download_func = partial(download_image, output_dir=args.output_dir)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_threads) as executor:
        results = list(tqdm(executor.map(download_func, rows_to_download), total=len(rows_to_download)))
        successful_downloads = sum(results)

    print(f"\n[SUCCESS] Downloaded {successful_downloads}/{len(rows_to_download)} images to '{args.output_dir}'")

if __name__ == "__main__":
    main()