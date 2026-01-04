import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, models
import os

# ---------------------------
# Load Grad-CAM function
# ---------------------------
from gradcam import generate_gradcam

# ---------------------------
# CONFIGURATION
# ---------------------------
IMAGE_PATH = "data_split/test/AK/ISIC_0024763.jpg"   # UPDATE if needed
MODEL_PATH = "skin_model.pth"                        # Your final model
OUTPUT_PATH = "figure3_gradcam.png"

# ---------------------------
# Load Model
# ---------------------------
num_classes = 8
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

print("Model loaded successfully.")

# ---------------------------
# Load and preprocess image
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

orig = Image.open(IMAGE_PATH).convert("RGB")
img_tensor = transform(orig).unsqueeze(0)

# ---------------------------
# Generate Grad-CAM
# ---------------------------
heatmap, overlay = generate_gradcam(img_tensor, model, model.layer4[-1])

# ---------------------------
# Create 3-panel IEEE-style figure
# ---------------------------
fig, ax = plt.subplots(1, 3, figsize=(12, 4))

ax[0].imshow(orig)
ax[0].set_title("(a) Original Image")
ax[0].axis("off")

ax[1].imshow(heatmap, cmap="jet")
ax[1].set_title("(b) Grad-CAM Heatmap")
ax[1].axis("off")

ax[2].imshow(overlay)
ax[2].set_title("(c) Overlay")
ax[2].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)
plt.close()

print(f"✅ Saved Figure 3 as {OUTPUT_PATH}")
