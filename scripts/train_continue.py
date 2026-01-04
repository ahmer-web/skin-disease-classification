import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

TRAIN_DIR = "data_split/train"
VAL_DIR = "data_split/val"
CHECKPOINT_DIR = "checkpoints"

EPOCHS_TO_RUN = 3
BATCH_SIZE = 16

# ---------------------------------------------------
# 1) Transforms
# ---------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------------------------------------------
# 2) Load datasets
# ---------------------------------------------------
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

classes = train_dataset.classes
num_classes = len(classes)

# ---------------------------------------------------
# 3) Load the SAME model as before (ResNet18!)
# ---------------------------------------------------
print("Loading model: ResNet18")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

# ---------------------------------------------------
# 4) Load last trained checkpoint
# ---------------------------------------------------
latest = Path("checkpoints/resnet18_epoch_2.pth")

if latest.exists():
    print(f"\n✅ Loading checkpoint weights from {latest}")
    state_dict = torch.load(latest, map_location=device)
    model.load_state_dict(state_dict)
else:
    print("❌ No checkpoint found. Cannot continue training.")
    exit()

# ---------------------------------------------------
# 5) Continue Training
# ---------------------------------------------------
print("\n🚀 Continuing training for 3 more epochs...\n")

for epoch in range(EPOCHS_TO_RUN):
    print(f"\n===== Epoch {epoch+1}/{EPOCHS_TO_RUN} =====")

    model.train()
    running_loss = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if batch_idx % 200 == 0:
            print(f"[Batch {batch_idx}/{len(train_loader)}] "
                  f"Train loss: {loss.item():.4f}")

    # --------- Validation ----------
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total

    print(f"Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

    # Save new checkpoint
    save_path = f"checkpoints/resnet18_continue_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"💾 Saved: {save_path}")

print("\n🎉 Training continued successfully!")