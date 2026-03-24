import os

def clean_image_filenames(data_dir):
    print(f"--- Starting Image Filename Cleanup ---")
    print(f"Scanning directory: {data_dir}")
    
    renamed_count = 0
    
    # Traverse directories and rename files
    for root, _, files in os.walk(data_dir):
        for filename in files:
            if "'" in filename:
                old_path = os.path.join(root, filename)
                new_filename = filename.replace("'", "")
                new_path = os.path.join(root, new_filename)
                
                try:
                    # Rename the file on disk
                    os.rename(old_path, new_path)
                    renamed_count += 1
                except Exception as e:
                    print(f"Failed to rename {filename}. Error: {e}")
    
    print(f"\nCleanup Complete! Successfully renamed {renamed_count} image files.")