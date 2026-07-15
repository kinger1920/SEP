import os
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

def augment_dataset(csv_path, images_dir):
    """
    Reads the CSV labels, generates three augmented versions (inverted, flipped,
    and inverted+flipped) for each original image, and appends the new entries
    to the CSV.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    # Load existing labels
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} entries from {csv_path}")

    # To make the script idempotent, only process original images
    # (i.e., those that do not contain '_inverted' or '_flipped' in their filenames)
    original_mask = ~df['filename'].str.contains('_inverted|_flipped', case=False, na=False)
    original_df = df[original_mask]
    
    print(f"Found {len(original_df)} original images to process.")
    
    if len(original_df) == 0:
        print("No new original images to process. Dataset might already be augmented.")
        return

    new_rows = []
    
    # Process each original image
    for idx, row in tqdm(original_df.iterrows(), total=len(original_df), desc="Augmenting Images"):
        filename = row['filename']
        label = row['label']
        
        img_path = os.path.join(images_dir, filename)
        if not os.path.exists(img_path):
            # Skip if image does not exist
            continue
            
        try:
            with Image.open(img_path) as img:
                # Ensure image is RGB (required for inverting)
                img_rgb = img.convert('RGB')
                
                # Split filename and extension
                base, ext = os.path.splitext(filename)
                
                # 1. Invert
                img_invert = ImageOps.invert(img_rgb)
                invert_name = f"{base}_inverted{ext}"
                img_invert.save(os.path.join(images_dir, invert_name))
                new_rows.append({'filename': invert_name, 'label': label})
                
                # 2. Flip (horizontal)
                img_flip = img_rgb.transpose(Image.FLIP_LEFT_RIGHT)
                flip_name = f"{base}_flipped{ext}"
                img_flip.save(os.path.join(images_dir, flip_name))
                new_rows.append({'filename': flip_name, 'label': label})
                
                # 3. Invert and Flip
                img_invert_flip = img_invert.transpose(Image.FLIP_LEFT_RIGHT)
                inv_flip_name = f"{base}_inverted_flipped{ext}"
                img_invert_flip.save(os.path.join(images_dir, inv_flip_name))
                new_rows.append({'filename': inv_flip_name, 'label': label})
                
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            continue

    if new_rows:
        # Append new rows to the CSV
        new_df = pd.DataFrame(new_rows)
        updated_df = pd.concat([df, new_df], ignore_index=True)
        updated_df.to_csv(csv_path, index=False)
        print(f"\nAugmentation complete! Added {len(new_rows)} new images.")
        print(f"Total entries in CSV now: {len(updated_df)}")
    else:
        print("\nNo new images were created.")

if __name__ == "__main__":
    # Define paths
    csv_file_path = "images/labels.csv"
    images_directory = "images"
    
    augment_dataset(csv_file_path, images_directory)
