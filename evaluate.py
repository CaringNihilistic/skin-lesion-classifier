"""
src/evaluate.py — Full evaluation metrics for skin lesion classification.
Run: python src/evaluate.py
"""

import torch
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    balanced_accuracy_score,
)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from model import build_model

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR   = "C:/skin-lesion/data/split"
MODEL_PATH = "models/best_model.pth"
IMG_SIZE   = 260
BATCH_SIZE = 32
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_test_loader():
    """Separate loader with num_workers=0 — avoids Windows multiprocessing error."""
    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_ds = datasets.ImageFolder(
        Path(DATA_DIR) / "test",
        transform=val_transforms
    )
    return DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,        # ← fixes the Windows multiprocessing crash
        pin_memory=True,
    ), test_ds.classes


def run_evaluation():
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load model ────────────────────────────────────────────────────────────
    checkpoint  = torch.load(MODEL_PATH, weights_only=False)
    class_names = checkpoint["class_names"]
    model       = build_model(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded — best val_acc: {checkpoint['val_acc']:.4f}")

    # ── Run inference ─────────────────────────────────────────────────────────
    test_loader, _ = get_test_loader()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs    = imgs.to(DEVICE)
            outputs = model(imgs)
            probs   = torch.softmax(outputs, dim=1)
            preds   = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=3
    ))

    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    auc     = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    print(f"Balanced Accuracy : {bal_acc:.4f} ({bal_acc*100:.2f}%)")
    print(f"Macro AUC-ROC     : {auc:.4f}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("CONFUSION MATRIX  (rows = actual, cols = predicted)")
    print("="*60)
    cm     = confusion_matrix(all_labels, all_preds)
    short  = [c[:6] for c in class_names]
    header = "".join(f"{s:>8}" for s in short)
    print(f"{'':>10}{header}")
    for i, row in enumerate(cm):
        print(f"{short[i]:>10}" + "".join(f"{v:>8}" for v in row))

    # ── Per-class AUC ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PER-CLASS AUC-ROC")
    print("="*60)
    for i, name in enumerate(class_names):
        binary_labels = (all_labels == i).astype(int)
        class_probs   = all_probs[:, i]
        from sklearn.metrics import roc_auc_score as ras
        class_auc = ras(binary_labels, class_probs)
        bar = "█" * int(class_auc * 20)
        print(f"  {name:<30} AUC: {class_auc:.3f}  {bar}")


# ── ENTRY POINT — required on Windows ─────────────────────────────────────────
if __name__ == "__main__":
    run_evaluation()
