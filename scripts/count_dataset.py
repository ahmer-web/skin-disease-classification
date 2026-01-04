import os

# Base folder containing train/val/test
BASE_DIR = "data_split"

# The three splits we created
SPLITS = ["train", "val", "test"]

# Get class names from the 'train' folder (AK, BCC, BKL, DF, MEL, NV, SCC, VASC)
train_dir = os.path.join(BASE_DIR, "train")
class_names = sorted(os.listdir(train_dir))

# Prepare a dictionary to hold counts
counts = {}
for cls in class_names:
    counts[cls] = {"train": 0, "val": 0, "test": 0, "total": 0}

# Count images in each split and class
for split in SPLITS:
    split_dir = os.path.join(BASE_DIR, split)
    for cls in class_names:
        class_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(class_dir):
            continue

        # Count only image files
        n_images = len([
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        counts[cls][split] = n_images
        counts[cls]["total"] += n_images

# Print a nice text table
print("\nClass-wise image counts (train / val / test / total):\n")
print(f"{'Class':<10} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>7}")
print("-" * 40)

total_train = total_val = total_test = total_overall = 0

for cls in class_names:
    tr = counts[cls]["train"]
    va = counts[cls]["val"]
    te = counts[cls]["test"]
    tot = counts[cls]["total"]

    total_train += tr
    total_val += va
    total_test += te
    total_overall += tot

    print(f"{cls:<10} {tr:7d} {va:7d} {te:7d} {tot:7d}")

print("-" * 40)
print(f"{'TOTAL':<10} {total_train:7d} {total_val:7d} {total_test:7d} {total_overall:7d}")
print()