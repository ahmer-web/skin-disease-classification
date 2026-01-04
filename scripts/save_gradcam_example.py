import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ✅ Import Grad-CAM helper from the same folder
from gradcam import generate_gradcam

# ----------------------------------
# 1. Find one example image automatically from data_split/test
# ----------------------------------
test_root = Path("data_split/test")

if not test_root.exists():
    raise FileNotFoundError("❌ data_split/test folder not found. Make sure your split is created.")

image_path = None

# Look for the first JPG/PNG in any class folder
for cls_dir in sorted(test_root.iterdir()):
    if cls_dir.is_dir():
        for fname in cls_dir.iterdir():
            if fname.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                image_path = fname
                break
    if image_path is not None:
        break

if image_path is None:
    raise RuntimeError("❌ No image files (.jpg/.png) found under data_split/test/*/*")

print(f"✅ Using image for Grad-CAM: {image_path}")

# ----------------------------------
# 2. Model config
# ----------------------------------
classes = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
num_classes = len(classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load ResNet-18 exactly like in training
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)

state_dict = torch.load("skin_model.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Use a layer from the last block as target layer
target_layer = model.layer4[1]

# ----------------------------------
# 3. Preprocess image
# ----------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

img_pil = Image.open(image_path).convert("RGB")
img_tensor = transform(img_pil).unsqueeze(0).to(device)

# ----------------------------------
# 4. Generate Grad-CAM
# ----------------------------------
heatmap_img, overlay_img = generate_gradcam(img_tensor, model, target_layer)

# Convert to NumPy for plotting
orig_np = np.array(img_pil.resize((224, 224)))
heat_np = np.array(heatmap_img)
overlay_np = np.array(overlay_img)

# ----------------------------------
# 5. Create 3-panel figure
# ----------------------------------
plt.figure(figsize=(9, 3))

plt.subplot(1, 3, 1)
plt.imshow(orig_np)
plt.axis("off")
plt.title("Original")

plt.subplot(1, 3, 2)
plt.imshow(heat_np)
plt.axis("off")
plt.title("Grad-CAM heatmap")

plt.subplot(1, 3, 3)
plt.imshow(overlay_np)
plt.axis("off")
plt.title("Overlay")

plt.tight_layout()
plt.savefig("fig_gradcam_example.png", dpi=300)
plt.close()

print("✅ Saved Grad-CAM example as fig_gradcam_example.png")
