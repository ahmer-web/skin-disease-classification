import os
import matplotlib.pyplot as plt

train_dir = "data_split/train"

class_counts = {}

for cls in os.listdir(train_dir):
    cls_path = os.path.join(train_dir, cls)
    if os.path.isdir(cls_path):
        class_counts[cls] = len(os.listdir(cls_path))

print(class_counts)

plt.figure(figsize=(8,4))
plt.bar(class_counts.keys(), class_counts.values())
plt.xlabel("Disease Class")
plt.ylabel("Number of Training Images")
plt.title("Class Distribution in the Training Subset")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figure4_class_distribution.png")
plt.show()
