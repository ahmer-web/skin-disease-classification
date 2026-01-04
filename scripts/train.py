import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# -----------------------------
# Paths and basic config
# -----------------------------
DATA_ROOT = "data_split"          # we will use data_split/train and data_split/val
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
CHECKPOINT_DIR = "checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

batch_size = 8          # small batch for CPU
num_epochs = 3          # keep it small so it finishes in reasonable time
learning_rate = 1e-4

# -----------------------------
# Transforms (with augmentation on train)
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# -----------------------------
# Datasets and loaders
# -----------------------------
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,      # 0 is safest on Windows
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)

print("✅ Dataset loaded")
print(f"  Train images: {len(train_dataset)}")
print(f"  Val images:   {len(val_dataset)}")
print(f"  Classes:      {train_dataset.classes}")
print(f"  Train batches per epoch: {len(train_loader)}")
print(f"  Val batches per epoch:   {len(val_loader)}")

# -----------------------------
# Model: ResNet-18 (lighter than ResNet-50)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

num_classes = len(train_dataset.classes)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(num_epochs):
    print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")

    # ---- Train ----
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Print progress every 50 batches
        if batch_idx % 50 == 0 or batch_idx == len(train_loader):
            avg_loss = running_loss / batch_idx
            print(f"  [Batch {batch_idx}/{len(train_loader)}] "
                  f"Train loss: {avg_loss:.4f}")

    # ---- Validation ----
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100.0 * correct / total
    print(f"  Val loss: {avg_val_loss:.4f}  |  Val accuracy: {val_acc:.2f}%")

    # Save checkpoint each epoch
    ckpt_path = Path(CHECKPOINT_DIR) / f"resnet18_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  💾 Saved checkpoint: {ckpt_path}")

# -----------------------------
# Save final model
# -----------------------------
final_model_path = "skin_model.pth"
torch.save(model.state_dict(), final_model_path)
print(f"\n✅ Training complete! Final model saved as {final_model_path}")