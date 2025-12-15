"""
Reorganize extracted ISL words for training
-------------------------------------------
Flattens the User_1, User_2, etc. structure into a simple class-based structure
that improved_dataset_processor.py expects:
    dataset/
        afraid/
            image1.jpg
            image2.jpg
            ...
        agree/
            image1.jpg
            ...
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Configuration
EXTRACTED_DIR = Path("extracted_isl_words/New folder")
REORGANIZED_DIR = Path("data")  # This matches what improved_dataset_processor.py expects

def reorganize_files():
    """Reorganize extracted files into proper structure for training"""
    
    print("=" * 60)
    print("ISL Words Reorganization")
    print("=" * 60)
    
    if not EXTRACTED_DIR.exists():
        print(f"Error: Extracted directory not found: {EXTRACTED_DIR}")
        return
    
    # Create output directory
    REORGANIZED_DIR.mkdir(exist_ok=True)
    print(f"Output directory: {REORGANIZED_DIR}")
    print()
    
    # Find all word directories
    word_dirs = [d for d in EXTRACTED_DIR.iterdir() if d.is_dir()]
    
    if not word_dirs:
        print("No word directories found!")
        return
    
    print(f"Found {len(word_dirs)} word directories")
    print()
    
    total_files = 0
    
    for word_dir in tqdm(word_dirs, desc="Processing words"):
        word_name = word_dir.name
        
        # Create output directory for this word
        output_word_dir = REORGANIZED_DIR / word_name
        output_word_dir.mkdir(exist_ok=True)
        
        # Find all nested directories (afraid/afraid/User_1, etc.)
        user_dirs = []
        for root, dirs, files in os.walk(word_dir):
            # Look for directories that contain JPG files
            if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
                user_dirs.append(Path(root))
        
        file_count = 0
        
        # Copy all images from all user directories to the word directory
        for user_dir in user_dirs:
            image_files = [f for f in user_dir.iterdir() 
                          if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            
            for img_file in image_files:
                # Create unique filename to avoid conflicts
                # Format: wordname_originalname.jpg
                new_name = f"{word_name}_{img_file.name}"
                dest_path = output_word_dir / new_name
                
                # Copy file
                shutil.copy2(img_file, dest_path)
                file_count += 1
        
        total_files += file_count
        print(f"  {word_name}: {file_count} images")
    
    print()
    print("=" * 60)
    print("REORGANIZATION SUMMARY")
    print("=" * 60)
    print(f"Total words processed: {len(word_dirs)}")
    print(f"Total images copied: {total_files}")
    print(f"Output directory: {REORGANIZED_DIR}")
    print()
    print("Directory structure:")
    print(f"  {REORGANIZED_DIR}/")
    for word_dir in sorted(REORGANIZED_DIR.iterdir()):
        if word_dir.is_dir():
            img_count = len(list(word_dir.glob("*.jpg"))) + len(list(word_dir.glob("*.jpeg"))) + len(list(word_dir.glob("*.png")))
            print(f"    ├── {word_dir.name}/ ({img_count} images)")
    print("=" * 60)
    print()
    print("✓ Files are now ready for training with improved_dataset_processor.py!")
    print(f"  Set DATASET_DIR = '{REORGANIZED_DIR}' in improved_dataset_processor.py")

if __name__ == "__main__":
    reorganize_files()

