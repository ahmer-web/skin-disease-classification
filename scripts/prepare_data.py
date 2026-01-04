import os
import shutil
from pathlib import Path

RAW_DIR = Path('raw')
OUT_DIR = Path('data')
OUT_DIR.mkdir(exist_ok=True)

# Map class folders (AK, BCC, ...) to same folder names in data/
for cls_folder in RAW_DIR.iterdir():
    if cls_folder.is_dir():
        dest_folder = OUT_DIR / cls_folder.name
        dest_folder.mkdir(exist_ok=True)
        for img_file in cls_folder.glob('*.jpg'):
            shutil.copy(img_file, dest_folder / img_file.name)

print("All images copied to data/ folder organized by class.")
