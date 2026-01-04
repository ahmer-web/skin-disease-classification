import os
import shutil
import random

# Source dataset directory
SOURCE_DIR = "data"

# Output split directories
OUTPUT_DIR = "data_split"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

# Split ratios
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Ensure reproducibility
random.seed(42)

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_dataset():
    classes = os.listdir(SOURCE_DIR)
    classes = [c for c in classes if os.path.isdir(os.path.join(SOURCE_DIR, c))]

    for cls in classes:
        class_path = os.path.join(SOURCE_DIR, cls)
        images = os.listdir(class_path)

        random.shuffle(images)

        total = len(images)
        train_end = int(total * TRAIN_SPLIT)
        val_end = train_end + int(total * VAL_SPLIT)

        train_files = images[:train_end]
        val_files = images[train_end:val_end]
        test_files = images[val_end:]

        print(f"\nClass: {cls}")
        print(f" Total: {total}")
        print(f" Train: {len(train_files)}")
        print(f" Val:   {len(val_files)}")
        print(f" Test:  {len(test_files)}")

        # Create class subfolders
        for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
            create_dir(os.path.join(split_dir, cls))

        # Copy files
        for file_list, split_dir in [
            (train_files, TRAIN_DIR),
            (val_files, VAL_DIR),
            (test_files, TEST_DIR)
        ]:
            for file in file_list:
                src = os.path.join(class_path, file)
                dst = os.path.join(split_dir, cls, file)
                shutil.copy(src, dst)

    print("\n✅ Dataset split completed successfully!")
    print(f"✅ Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    split_dataset()