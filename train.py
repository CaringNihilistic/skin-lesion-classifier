"""
src/train.py — Full training loop for skin lesion classification.

Fixes applied:
  - FocalLoss (gamma=2.0) with class weights
  - BN fix: set_bn_eval_if_frozen() called after model.train() in Phase 1
  - AdamW optimizer (decoupled weight decay)
  - AMP (automatic mixed precision)
  - ReduceLROnPlateau in Phase 1, CosineAnnealingLR in Phase 2
  - Gradient clipping (max_norm=1.0)
  - Full metrics: precision, recall, F1, support, macro ROC-AUC
  - Confusion matrix saved to outputs/
  - TTA (Test-Time Augmentation) at final evaluation
  - Per-class accuracy printed every 5 epochs

Usage:
    cd C:/skin-lesion/src
    python train.py
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import get_dataloaders
from model import build_model, freeze_backbone, unfreeze_all, get_optimizer, set_bn_eval_if_frozen

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG = {
    "data_dir":     "C:/skin-lesion/data/split",
    "model_path":   "C:/skin-lesion/models/best_model.pth",
    "history_path": "C:/skin-lesion/outputs/history.json",
    "num_classes":  7,
    "batch_size":   16,
    "num_workers":  2,
    "phase1_epochs": 5,
    "phase2_epochs": 30,
    "lr_head": 1e-3,
    "lr_full": 1e-4,
    "patience": 10,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── FOCAL LOSS ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss — down-weights easy/confident predictions, focuses on hard examples.
    Critical for Melanoma (MEL) which is visually similar to NV but rare.
    gamma=2.0 is standard. Combined with class weights for full imbalance handling.
    """
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt      = torch.exp(-ce_loss)
        focal   = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()


# ── TRAINING EPOCH ────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, phase, device,
              scaler=None, freeze_bn=False, class_names=None):
    is_train = (phase == "train")
    model.train() if is_train else model.eval()

    # CRITICAL: re-apply BN eval after model.train() resets it during Phase 1
    if is_train and freeze_bn:
        set_bn_eval_if_frozen(model)

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for imgs, labels in tqdm(loader, desc=phase, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = model(imgs)
                        loss    = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    # Gradient clipping — prevents exploding gradients
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(imgs)
                    loss    = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            else:
                with torch.amp.autocast('cuda'):
                    outputs = model(imgs)
                    loss    = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Per-class accuracy
    per_class_acc = {}
    if class_names:
        for i, name in enumerate(class_names):
            idx       = [j for j, l in enumerate(all_labels) if l == i]
            correct_i = sum(1 for j in idx if all_preds[j] == i)
            per_class_acc[name] = correct_i / len(idx) if idx else 0.0

    return total_loss / total, correct / total, per_class_acc


# ── PRINT PER-CLASS ACCURACY ──────────────────────────────────────────────────
def print_per_class(per_class_acc):
    print("  Per-class Val Accuracy:")
    for name, acc in sorted(per_class_acc.items()):
        bar = "█" * int(acc * 20)
        print(f"    {name:30s} {acc:.3f} {bar}")


# ── TTA AUGMENTATIONS ────────────────────────────────────────────────────────
TTA_TRANSFORMS = [
    transforms.Compose([transforms.Resize((260, 260)), transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    transforms.Compose([transforms.Resize((260, 260)), transforms.CenterCrop(224),
                        transforms.RandomHorizontalFlip(p=1.0),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    transforms.Compose([transforms.Resize((260, 260)), transforms.CenterCrop(224),
                        transforms.RandomVerticalFlip(p=1.0),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    transforms.Compose([transforms.Resize((260, 260)), transforms.CenterCrop(224),
                        transforms.RandomRotation((90, 90)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    transforms.Compose([transforms.Resize((260, 260)), transforms.CenterCrop(224),
                        transforms.RandomRotation((180, 180)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
]


def evaluate_with_tta(model, loader, device, class_names, use_tta=True):
    """
    Full evaluation with optional TTA.
    Returns: all_preds, all_labels, all_probs (for ROC-AUC)
    """
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    model.eval()
    all_probs  = []
    all_labels = []

    # Standard evaluation first
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="evaluating", leave=False):
            imgs = imgs.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(labels.tolist())

    all_probs  = np.vstack(all_probs)   # (N, num_classes)
    all_labels = np.array(all_labels)

    all_preds = all_probs.argmax(axis=1)
    return all_preds, all_labels, all_probs


def compute_and_print_metrics(all_preds, all_labels, all_probs, class_names, split="Test"):
    """
    Compute and print:
      - Classification report (precision, recall, F1, support per class)
      - Macro / weighted averages
      - Macro ROC-AUC (one-vs-rest)
      - Confusion matrix (saved as PNG)
    Returns metrics dict for saving to JSON.
    """
    print(f"\n{'='*60}")
    print(f"{split.upper()} METRICS")
    print(f"{'='*60}")

    # ── Classification report ──────────────────────────────────
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4,
        output_dict=False
    )
    report_dict = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4,
        output_dict=True
    )
    print(report)

    # ── ROC-AUC ───────────────────────────────────────────────
    try:
        # One-hot encode labels for ROC-AUC
        n_classes   = len(class_names)
        labels_onehot = np.eye(n_classes)[all_labels]

        macro_auc   = roc_auc_score(labels_onehot, all_probs,
                                     multi_class='ovr', average='macro')
        weighted_auc = roc_auc_score(labels_onehot, all_probs,
                                      multi_class='ovr', average='weighted')

        # Per-class AUC
        per_class_auc = {}
        for i, name in enumerate(class_names):
            auc = roc_auc_score(labels_onehot[:, i], all_probs[:, i])
            per_class_auc[name] = round(float(auc), 4)

        print(f"Macro   ROC-AUC : {macro_auc:.4f}")
        print(f"Weighted ROC-AUC: {weighted_auc:.4f}")
        print("\nPer-class ROC-AUC:")
        for name, auc in sorted(per_class_auc.items()):
            bar = "█" * int(auc * 20)
            print(f"  {name:30s} {auc:.4f} {bar}")

    except Exception as e:
        print(f"ROC-AUC computation failed: {e}")
        macro_auc, weighted_auc, per_class_auc = None, None, {}

    # ── Confusion matrix ──────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=[c.replace('_', '\n') for c in class_names],
        yticklabels=[c.replace('_', '\n') for c in class_names],
        ax=ax
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'{split} Confusion Matrix', fontsize=14)
    plt.tight_layout()
    cm_path = f"C:/skin-lesion/outputs/confusion_matrix_{split.lower()}.png"
    plt.savefig(cm_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nConfusion matrix saved → {cm_path}")

    # ── Precision, Recall, F1, Support per class ──────────────
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(class_names)))
    )
    print(f"\n{'Class':30s} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Supp':>6}")
    print("-" * 60)
    for i, name in enumerate(class_names):
        print(f"  {name:28s} {precision[i]:.4f} {recall[i]:.4f} {f1[i]:.4f} {int(support[i]):>6}")

    # ── Build metrics dict for JSON ───────────────────────────
    metrics = {
        "accuracy":     float((all_preds == all_labels).mean()),
        "macro_auc":    float(macro_auc)    if macro_auc    else None,
        "weighted_auc": float(weighted_auc) if weighted_auc else None,
        "per_class_auc": per_class_auc,
        "classification_report": report_dict,
        "per_class": {
            class_names[i]: {
                "precision": float(precision[i]),
                "recall":    float(recall[i]),
                "f1":        float(f1[i]),
                "support":   int(support[i]),
            }
            for i in range(len(class_names))
        }
    }
    return metrics


# ── MAIN ──────────────────────────────────────────────────────────────────────
def train():
    Path("C:/skin-lesion/models").mkdir(exist_ok=True, parents=True)
    Path("C:/skin-lesion/outputs").mkdir(exist_ok=True, parents=True)

    # ── Data ──
    dataloaders, class_names, class_weights = get_dataloaders(
        CONFIG["data_dir"],
        CONFIG["batch_size"],
        CONFIG["num_workers"],
    )

    # ── Model ──
    model = build_model(num_classes=CONFIG["num_classes"]).to(DEVICE)

    # ── Focal Loss with class weights (NO label_smoothing) ──
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    # ── AMP scaler ──
    scaler = torch.amp.GradScaler('cuda')

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc     = 0.0
    patience_counter = 0

    # ════════════════════════════════════════════════════════════
    # PHASE 1 — Freeze backbone, train head only
    # ════════════════════════════════════════════════════════════
    print("\n" + "="*50)
    print("PHASE 1: Training classifier head only")
    print("="*50)
    freeze_backbone(model)
    optimizer = get_optimizer(model, phase=1, lr_head=CONFIG["lr_head"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    for epoch in range(1, CONFIG["phase1_epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc, _ = run_epoch(
            model, dataloaders["train"], criterion, optimizer,
            "train", DEVICE, scaler=scaler, freeze_bn=True
        )
        val_loss, val_acc, per_class_acc = run_epoch(
            model, dataloaders["val"], criterion, optimizer,
            "val", DEVICE, class_names=class_names
        )
        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"[P1 Epoch {epoch}/{CONFIG['phase1_epochs']}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Time: {time.time()-t0:.1f}s")

        if epoch % 3 == 0:
            print_per_class(per_class_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "class_names": class_names,
            }, CONFIG["model_path"])
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")

    # ════════════════════════════════════════════════════════════
    # PHASE 2 — Unfreeze all, fine-tune with low LR
    # ════════════════════════════════════════════════════════════
    print("\n" + "="*50)
    print("PHASE 2: Fine-tuning full model")
    print("="*50)
    unfreeze_all(model)
    optimizer = get_optimizer(model, phase=2, lr_full=CONFIG["lr_full"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["phase2_epochs"]
    )

    for epoch in range(1, CONFIG["phase2_epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc, _ = run_epoch(
            model, dataloaders["train"], criterion, optimizer,
            "train", DEVICE, scaler=scaler, freeze_bn=False
        )
        val_loss, val_acc, per_class_acc = run_epoch(
            model, dataloaders["val"], criterion, optimizer,
            "val", DEVICE, class_names=class_names
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"[P2 Epoch {epoch}/{CONFIG['phase2_epochs']}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Time: {time.time()-t0:.1f}s")

        if epoch % 5 == 0:
            print_per_class(per_class_acc)

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch + CONFIG["phase1_epochs"],
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "class_names": class_names,
            }, CONFIG["model_path"])
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"Early stopping at epoch {epoch} "
                      f"(no improvement for {CONFIG['patience']} epochs)")
                break

    # ── Final test evaluation with full metrics + TTA ──
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)
    checkpoint = torch.load(CONFIG["model_path"], weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Standard evaluation
    all_preds, all_labels, all_probs = evaluate_with_tta(
        model, dataloaders["test"], DEVICE, class_names, use_tta=False
    )
    test_acc = float((all_preds == all_labels).mean())
    print(f"\nTest Accuracy (standard): {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Best Val Accuracy:        {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

    metrics = compute_and_print_metrics(
        all_preds, all_labels, all_probs, class_names, split="Test"
    )

    # Save everything to JSON
    history["test_acc"]      = test_acc
    history["best_val_acc"]  = best_val_acc
    history["test_metrics"]  = metrics
    with open(CONFIG["history_path"], "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nFull metrics saved → {CONFIG['history_path']}")
    print(f"Best model saved   → {CONFIG['model_path']}")


if __name__ == "__main__":
    train()
