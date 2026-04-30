"""
src/dataset.py — PyTorch Dataset with augmentations for skin lesion classification.

Fixes applied:
  - IMG_SIZE = 260 for EfficientNetB2
  - RandomErasing — removes ruler marks, hair, artifacts from dermoscopy images
  - CenterCrop for val/test (more stable than simple resize)
  - WeightedRandomSampler — oversamples rare classes (MEL, DF, VASC) each epoch
  - sqrt inverse frequency class weights (gentler than raw inverse)
  - persistent_workers=True for faster DataLoader on Windows
"""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from pathlib import Path

# ── IMAGE SIZE ────────────────────────────────────────────────────────────────
IMG_SIZE = 260   # EfficientNetB2 expects 260x260

# ── TRANSFORMS ───────────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    # Removes dermoscopy artifacts: ruler marks, hair, calibration patches
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def get_dataloaders(data_dir: str, batch_size: int = 16, num_workers: int = 2):
    """
    Loads train/val/test splits from ImageFolder structure.

    Args:
        data_dir:    path to data/split/
        batch_size:  16 safe for RTX 3050 4GB with EfficientNetB2
        num_workers: 2 on Windows

    Returns:
        dataloaders:   dict with 'train', 'val', 'test' keys
        class_names:   list of class folder names (sorted alphabetically)
        class_weights: tensor for FocalLoss (handles class imbalance)
    """
    data_path = Path(data_dir)

    image_datasets = {
        "train": datasets.ImageFolder(data_path / "train", transform=train_transforms),
        "val":   datasets.ImageFolder(data_path / "val",   transform=val_transforms),
        "test":  datasets.ImageFolder(data_path / "test",  transform=val_transforms),
    }

    class_names = image_datasets["train"].classes
    NUM_CLASSES = len(class_names)

    # ── CLASS WEIGHTS (sqrt inverse frequency) ───────────────────────────────
    # Gentler than raw inverse — handles extreme imbalance (NV=67%) without
    # crushing the NV gradient signal entirely.
    counts = torch.zeros(NUM_CLASSES)
    for _, label in image_datasets["train"].samples:
        counts[label] += 1

    class_weights = 1.0 / torch.sqrt(counts)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES

    # ── WEIGHTED RANDOM SAMPLER ───────────────────────────────────────────────
    # Each epoch: rare classes (MEL, DF, VASC) appear proportionally more often.
    # This ensures the model sees hard minority classes every epoch.
    sample_weights = torch.zeros(len(image_datasets["train"]))
    for idx, (_, label) in enumerate(image_datasets["train"].samples):
        sample_weights[idx] = class_weights[label]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # ── DATALOADERS ───────────────────────────────────────────────────────────
    dataloaders = {
        "train": DataLoader(
            image_datasets["train"],
            batch_size=batch_size,
            sampler=sampler,             # replaces shuffle=True
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        ),
        "val": DataLoader(
            image_datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        ),
        "test": DataLoader(
            image_datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        ),
    }

    print(f"Classes ({NUM_CLASSES}): {class_names}")
    print(f"Train: {len(image_datasets['train'])} | "
          f"Val: {len(image_datasets['val'])} | "
          f"Test: {len(image_datasets['test'])}")
    print(f"Class counts (train):")
    for i, c in enumerate(class_names):
        print(f"  {c:30s} {int(counts[i]):5d}  weight={class_weights[i]:.4f}")

    return dataloaders, class_names, class_weights
