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
