import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image, ImageFile
from tqdm import tqdm

# ============================================================
# FIX BROKEN IMAGES
# ============================================================

ImageFile.LOAD_TRUNCATED_IMAGES = True

def safe_loader(path):
    """Loads an image safely. Skips corrupted files."""
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        print(f"[WARNING] Corrupted image skipped: {path}")
        return None  # return None → handled automatically by DataLoader


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = r"C:\Users\vishn\Documents\HEBA\plus"
BATCH_SIZE = 64
NUM_CLASSES = 8
IMG_SIZE = 96
LR = 1e-4
EPOCHS = 40
WEIGHT_DECAY = 1e-4
CHECKPOINT_PATH = "best_emotion_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device = {device}")

# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ============================================================
# DATASET CLASS OVERRIDE (TO SKIP CORRUPTED IMAGES)
# ============================================================

class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = safe_loader(path)

        if img is None:
            # corrupted image → skip by loading next valid item
            new_index = (index + 1) % len(self.samples)
            return self.__getitem__(new_index)

        if self.transform:
            img = self.transform(img)
        return img, target


# ============================================================
# LOAD DATASETS
# ============================================================

train_ds = SafeImageFolder(os.path.join(DATASET_ROOT, "train"), transform=train_transform)
val_ds = SafeImageFolder(os.path.join(DATASET_ROOT, "validation"), transform=val_test_transform)
test_ds = SafeImageFolder(os.path.join(DATASET_ROOT, "test"), transform=val_test_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Classes:", train_ds.classes)

# ============================================================
# MODEL
# ============================================================

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


# ============================================================
# TRAINING LOOP
# ============================================================

def train_one_epoch():
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    progress = tqdm(train_loader, desc="Train", leave=False)

    for img, lbl in progress:
        img, lbl = img.to(device), lbl.to(device)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, lbl)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img.size(0)
        _, pred = torch.max(out, 1)
        correct += (pred == lbl).sum().item()
        total += lbl.size(0)

        progress.set_postfix(loss=loss.item())

    return running_loss / total, correct / total


def validate():
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        progress = tqdm(val_loader, desc="Validation", leave=False)
        for img, lbl in progress:
            img, lbl = img.to(device), lbl.to(device)
            out = model(img)
            loss = criterion(out, lbl)

            running_loss += loss.item() * img.size(0)
            _, pred = torch.max(out, 1)
            correct += (pred == lbl).sum().item()
            total += lbl.size(0)

    return running_loss / total, correct / total


# ============================================================
# MAIN TRAINING
# ============================================================

best_acc = 0

print("\n[INFO] Starting training...\n")

for epoch in range(EPOCHS):
    print(f"\n========== EPOCH {epoch+1}/{EPOCHS} ==========")

    t_loss, t_acc = train_one_epoch()
    v_loss, v_acc = validate()

    print(f"Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.4f}")
    print(f"Val   Loss: {v_loss:.4f} | Val   Acc: {v_acc:.4f}")

    if v_acc > best_acc:
        best_acc = v_acc
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"[INFO] Best model saved! (Val Acc = {best_acc:.4f})")

print(f"\n[INFO] Training Complete. Best Validation Accuracy = {best_acc:.4f}")


# ============================================================
# TESTING
# ============================================================

model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for img, lbl in test_loader:
        img, lbl = img.to(device), lbl.to(device)
        out = model(img)
        _, pred = torch.max(out, 1)
        correct += (pred == lbl).sum().item()
        total += lbl.size(0)

print(f"\n[TEST] Accuracy = {correct/total:.4f}")
