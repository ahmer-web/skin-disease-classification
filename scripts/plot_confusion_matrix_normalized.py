import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models

# ---- 1. Data (same transforms as test_evaluate.py) ----
DATA_DIR = "data_split/test"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class_names = test_dataset.classes

# ---- 2. Load your best model (ResNet-18) ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("skin_model.pth", map_location=device))
model.to(device)
model.eval()

# ---- 3. Collect predictions ----
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())

# ---- 4. Confusion matrix (normalized per row) ----
cm = confusion_matrix(all_labels, all_preds, normalize="true")
cm = np.around(cm, decimals=2)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    vmin=0,
    vmax=1
)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Normalized confusion matrix on the test set")
plt.tight_layout()
plt.savefig("fig_confusion_matrix_norm.png", dpi=300)
plt.close()
