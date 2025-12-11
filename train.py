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

