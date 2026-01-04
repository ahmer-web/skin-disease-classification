import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# ---------------------------
# Paths
# ---------------------------
TEST_DIR = "data_split/test"
CHECKPOINT_PATH = "checkpoints/resnet18_continue_epoch_2.pth"   # BEST MODEL

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# Transforms
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------------------
# Load Test Dataset
# ---------------------------
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

class_names = test_dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)
print("Test images:", len(test_dataset))

# ---------------------------
# Load Model
# ---------------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.to(device)
model.eval()

# ---------------------------
# Evaluation
# ---------------------------
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# ---------------------------
# Accuracy
# ---------------------------
y_true_np = np.array(y_true)
y_pred_np = np.array(y_pred)

accuracy = (y_true_np == y_pred_np).sum() / len(y_true_np) * 100
print(f"\n🔥 Final Test Accuracy: {accuracy:.2f}%")

# ---------------------------
# Classification Report
# ---------------------------
report = classification_report(
    y_true_np, y_pred_np,
    target_names=class_names,
    digits=4
)

print("\n=== Classification Report ===\n")
print(report)

with open("classification_report.txt", "w") as f:
    f.write(report)

print("\n📄 Saved classification_report.txt")

# ---------------------------
# Confusion Matrix
# ---------------------------
cm = confusion_matrix(y_true_np, y_pred_np)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("📊 Saved confusion_matrix.png")
plt.close()