import io
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # .../SkinDiseaseProject
MODEL_PATH = PROJECT_ROOT / "skin_model.pth"

# ----------------------------
# Classes (folder names)
# IMPORTANT: must match training folder names exactly
# ----------------------------
classes = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]

# ----------------------------
# Transform (must match training)
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Build model (ResNet-18)
# ----------------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))

# Load weights
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Target layer for Grad-CAM (last conv block)
target_layer = model.layer4[-1]

# ----------------------------
# Helpers
# ----------------------------
def _predict_from_pil(img_pil: Image.Image):
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)[0].detach().cpu()

    # list of (class, prob) sorted high->low
    results = sorted([(classes[i], float(probs[i])) for i in range(len(classes))],
                     key=lambda x: x[1], reverse=True)

    return img_pil, results, img_tensor, target_layer


# ----------------------------
# Exported functions (used by app.py)
# ----------------------------
def predict_image(uploaded_file):
    """
    For Streamlit uploader objects.
    Returns: (PIL image, results, img_tensor, target_layer)
    """
    img_pil = Image.open(uploaded_file).convert("RGB")
    return _predict_from_pil(img_pil)


def predict_image_bytes(uploaded_file):
    """
    For Streamlit multiple files (bytes).
    Returns: (PIL image, top_class, top_conf, results)
    """
    # Streamlit UploadedFile behaves like a file-like object.
    img_pil = Image.open(uploaded_file).convert("RGB")
    img_pil, results, _, _ = _predict_from_pil(img_pil)
    top_class, top_conf = results[0]
    return img_pil, top_class, top_conf, results

