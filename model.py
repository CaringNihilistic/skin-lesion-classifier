"""
src/model.py — EfficientNetB0 fine-tuned for 7-class skin lesion classification.

CHANGES FROM PREVIOUS VERSION:
- REVERTED: Removed drop_rate=0.3 override. EfficientNetB0's default dropout (0.2)
  is well-tuned. Increasing it to 0.3 was adding unnecessary noise early in training.
- KEPT: BatchNorm eval fix (the critical bug fix)
- KEPT: AdamW with weight_decay=1e-4 (better than plain Adam)
- KEPT: lr_full/5 backbone LR in Phase 2 (less aggressive than lr_full/10)
"""

import torch
import torch.nn as nn
import timm


def build_model(num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """
    Build EfficientNetB0 with custom classification head.
    Uses default drop_rate=0.2 (EfficientNetB0's original tuned value).
    """
    model = timm.create_model(
        "efficientnet_b2",
        pretrained=pretrained,
        num_classes=num_classes,
        # No drop_rate override — use EfficientNetB0's default (0.2)
    )
    return model


def freeze_backbone(model: nn.Module):
    """
    Phase 1: Freeze all layers except the classifier head.

    CRITICAL FIX: After freezing params, explicitly put ALL backbone
    BatchNorm layers into eval() mode. Without this, model.train() in
    run_epoch() flips them back to train mode and they start updating
    their running_mean/running_var using skin lesion images, destroying
    the ImageNet pretrained statistics. This was the primary bug causing ~57% accuracy.
    """
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    _set_backbone_bn_eval(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen backbone. Trainable params: {trainable:,}")


def _set_backbone_bn_eval(model: nn.Module):
    """
    Put all BatchNorm layers in the backbone (anything not named 'classifier')
    into eval mode so their running statistics are not updated.
    """
    for name, module in model.named_modules():
        if "classifier" in name:
            continue
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            module.eval()


def set_bn_eval_if_frozen(model: nn.Module):
    """
    Public helper called by train.py at the start of every Phase 1 training epoch,
    immediately after model.train() — which would otherwise reset all BN layers
    to train mode, undoing the freeze.
    """
    _set_backbone_bn_eval(model)


def unfreeze_all(model: nn.Module):
    """
    Phase 2: Unfreeze everything for full fine-tuning.
    In Phase 2, all layers (including BN) run in train mode — correct.
    """
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"All layers unfrozen. Trainable params: {trainable:,}")


def get_optimizer(model: nn.Module, phase: int, lr_head: float = 1e-3, lr_full: float = 1e-4):
    """
    AdamW (not Adam) — properly decouples weight decay from gradient updates,
    which gives consistent L2 regularization. Helps prevent overfitting on
    small medical datasets (~7k training images).

    Phase 1: High LR, only head params.
    Phase 2: Low LR, all params. Backbone gets lr/5 (less aggressive than original lr/10).
    """
    if phase == 1:
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr_head, weight_decay=1e-4)
    else:
        backbone_params = [p for name, p in model.named_parameters()
                           if "classifier" not in name]
        head_params     = [p for name, p in model.named_parameters()
                           if "classifier" in name]
        return torch.optim.AdamW([
            {"params": backbone_params, "lr": lr_full / 5},
            {"params": head_params,     "lr": lr_full},
        ], weight_decay=1e-4)
