"""
Extract all zip files from Indian sign Language-Real-life Words folder
---------------------------------------------------------------------
This script extracts all zip files found in the ISL Real-life Words directory
and organizes them into folders named after the zip file (without .zip extension)
"""

import os
import zipfile
import shutil
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent
ISL_WORDS_DIR = BASE_DIR / "Indian sign Language-Real-life Words"
EXTRACT_BASE_DIR = BASE_DIR / "extracted_isl_words"

def extract_zip_file(zip_path, extract_to):
    """Extract a single zip file to the specified directory"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True, None
    except zipfile.BadZipFile:
        return False, "Bad zip file"
    except Exception as e:
        return False, str(e)

def find_all_zip_files(directory):
    """Recursively find all zip files in a directory"""
    zip_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.zip'):
                zip_files.append(os.path.join(root, file))
    return zip_files

def main():
    print("=" * 60)
    print("ISL Real-life Words Extractor")
    print("=" * 60)
    
    # Check if source directory exists
    if not ISL_WORDS_DIR.exists():
        print(f"Error: Directory not found: {ISL_WORDS_DIR}")
        return
    
    # Create extraction directory
    EXTRACT_BASE_DIR.mkdir(exist_ok=True)
    print(f"Extraction directory: {EXTRACT_BASE_DIR}")
    print()
    
    # Find all zip files
    print("Searching for zip files...")
    zip_files = find_all_zip_files(ISL_WORDS_DIR)
    
    if not zip_files:
        print("No zip files found!")
        return
    
    print(f"Found {len(zip_files)} zip file(s)")
    print()
    
    # Extract each zip file
    success_count = 0
    error_count = 0
    
    for zip_path in zip_files:
        # Get the zip file name without extension
        zip_name = Path(zip_path).stem  # Gets filename without .zip
        relative_path = Path(zip_path).relative_to(ISL_WORDS_DIR)
        
        # Create extraction directory for this zip file
        # Preserve the folder structure from source
        if relative_path.parent != Path('.'):
            # If zip is in a subdirectory, create same structure in extract dir
            extract_dir = EXTRACT_BASE_DIR / relative_path.parent / zip_name
        else:
            extract_dir = EXTRACT_BASE_DIR / zip_name
        
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Extracting: {zip_name}")
        print(f"  From: {zip_path}")
        print(f"  To: {extract_dir}")
        
        success, error_msg = extract_zip_file(zip_path, extract_dir)
        
        if success:
            # Count extracted files
            extracted_count = sum(len(files) for _, _, files in os.walk(extract_dir))
            print(f"  ✓ Success! ({extracted_count} files extracted)")
            success_count += 1
        else:
            print(f"  ✗ Failed: {error_msg}")
            error_count += 1
        print()
    
    # Summary
    print("=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total zip files: {len(zip_files)}")
    print(f"Successfully extracted: {success_count}")
    print(f"Failed: {error_count}")
    print(f"Extracted files location: {EXTRACT_BASE_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()

