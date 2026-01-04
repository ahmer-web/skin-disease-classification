import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

DATA_DIR = "data"
MODEL_PATH = "skin_model.pth"

# Image transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load full dataset
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

# Recreate SAME validation split
train_idx, val_idx = train_test_split(
    list(range(len(dataset))),
    test_size=0.2,
    random_state=42,
    shuffle=True
)

from torch.utils.data import Subset
val_dataset = Subset(dataset, val_idx)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load model
model = models.resnet50()
num_classes = len(dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# Evaluation
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.numpy())
        all_preds.extend(predicted.numpy())

# Generate metrics
report = classification_report(all_labels, all_preds, target_names=dataset.classes)
print(report)

with open("classification_report.txt", "w") as f:
    f.write(report)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=dataset.classes,
            yticklabels=dataset.classes, cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig("confusion_matrix.png")

print("\n✅ Evaluation completed!")
print("✅ classification_report.txt saved")
print("✅ confusion_matrix.png saved")
