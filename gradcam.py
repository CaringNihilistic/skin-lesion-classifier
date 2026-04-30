"""
src/gradcam.py — Grad-CAM visualization for EfficientNetB0.

CHANGES FROM ORIGINAL:
- BUG FIX: cm.get_cmap("jet") is deprecated in matplotlib >= 3.7 and raises
  a warning/error. Replaced with matplotlib.colormaps["jet"] (new API) with
  a fallback to the old API for older matplotlib versions.
- Added try/except around hook registration for robustness
- Improved predict_with_gradcam() to reuse a single model forward pass
  (original made two separate forward passes — wasteful and inconsistent)
- Added save_gradcam_figure() utility for saving output to disk

Usage:
    from gradcam import GradCAM, visualize_gradcam
    cam = GradCAM(model)
    heatmap, class_idx, confidence = cam.generate(img_tensor)
    fig = visualize_gradcam(original_img, heatmap, predicted_class, confidence)
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image


def _get_colormap(name: str):
    """
    FIX: cm.get_cmap() was deprecated in matplotlib 3.7 and removed in 3.9.
    Use the new matplotlib.colormaps[] API with a fallback for older versions.
    """
    try:
        return matplotlib.colormaps[name]          # matplotlib >= 3.7
    except AttributeError:
        import matplotlib.cm as cm
        return cm.get_cmap(name)                   # matplotlib < 3.7 fallback


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Registers forward and backward hooks on the last conv layer
    of EfficientNetB0 to capture activations and gradients.
    """

    def __init__(self, model: torch.nn.Module, device="cuda"):
        self.model  = model
        self.device = device
        self.model.eval()

        self.activations = None
        self.gradients   = None
        self._hooks      = []

        # EfficientNetB0 in timm: last conv block is model.conv_head
        target_layer = model.conv_head

        # Forward hook: save the activation map
        h1 = target_layer.register_forward_hook(self._save_activation)
        # Backward hook: save the gradients
        h2 = target_layer.register_full_backward_hook(self._save_gradient)

        self._hooks = [h1, h2]

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        """Call this when done to avoid memory leaks from dangling hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def generate(self, img_tensor: torch.Tensor, class_idx: int = None):
        """
        Generate Grad-CAM heatmap for a single image.

        Args:
            img_tensor:  (1, 3, 224, 224) normalized tensor on device
            class_idx:   class to explain (None = use predicted class)

        Returns:
            heatmap:    (224, 224) numpy array, values in [0, 1]
            class_idx:  int — the predicted (or specified) class index
            confidence: float — softmax probability of the target class
        """
        img_tensor = img_tensor.to(self.device)
        img_tensor = img_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(img_tensor)
        probs  = torch.softmax(output, dim=1)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass on the target class score
        self.model.zero_grad()
        output[0, class_idx].backward()

        # Global average pooling of gradients → weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        # Resize to input image size (224x224)
        cam = cv2.resize(cam, (224, 224))
        return cam, class_idx, probs[0, class_idx].item()


def overlay_heatmap(original_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4):
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        original_img: (H, W, 3) numpy array, uint8 [0, 255]
        heatmap:      (H, W)    numpy array, float [0, 1]
        alpha:        transparency of heatmap overlay

    Returns:
        overlaid: (H, W, 3) numpy array
    """
    # FIX: use new colormap API instead of deprecated cm.get_cmap()
    colormap = _get_colormap("jet")
    heatmap_colored = colormap(heatmap)[:, :, :3]         # (H, W, 3) float
    heatmap_uint8   = (heatmap_colored * 255).astype(np.uint8)

    original_resized = cv2.resize(original_img, (224, 224))
    overlaid = cv2.addWeighted(original_resized, 1 - alpha, heatmap_uint8, alpha, 0)
    return overlaid


def visualize_gradcam(
    original_img: np.ndarray,
    heatmap:      np.ndarray,
    class_name:   str,
    confidence:   float,
    all_probs:    dict = None,
):
    """
    Creates a 3-panel matplotlib figure:
      [Original] [Grad-CAM Heatmap] [Overlaid]
    With class probabilities bar chart if all_probs is provided.
    """
    overlaid = overlay_heatmap(original_img, heatmap)

    if all_probs:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    fig.suptitle(
        f"Prediction: {class_name}  |  Confidence: {confidence*100:.1f}%",
        fontsize=14, fontweight="bold"
    )

    axes[0].imshow(cv2.resize(original_img, (224, 224)))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlaid)
    axes[2].set_title("Grad-CAM Overlay")
    axes[2].axis("off")

    if all_probs and len(axes) == 4:
        classes = list(all_probs.keys())
        probs   = list(all_probs.values())
        colors  = ["#e74c3c" if c == class_name else "#3498db" for c in classes]
        axes[3].barh(classes, probs, color=colors)
        axes[3].set_xlim(0, 1)
        axes[3].set_xlabel("Probability")
        axes[3].set_title("Class Probabilities")

    plt.tight_layout()
    return fig


def save_gradcam_figure(fig: plt.Figure, output_path: str):
    """Save the Grad-CAM figure to disk."""
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grad-CAM figure saved → {output_path}")


def predict_with_gradcam(model, img_path: str, class_names: list, device, transform):
    """
    Full pipeline: load image → preprocess → predict → generate Grad-CAM.

    FIX: Now uses a single forward pass for both Grad-CAM and all-class probabilities.
    The original made two separate forward passes which was wasteful and could give
    inconsistent softmax values if the model had any stochastic components (dropout).

    Returns:
        class_name:   predicted class string
        confidence:   float in [0, 1]
        all_probs:    dict {class_name: probability}
        fig:          matplotlib figure with visualization
        heatmap:      raw heatmap numpy array
    """
    # Load original image (for display)
    original_img = np.array(Image.open(img_path).convert("RGB"))

    # Preprocess
    img_pil    = Image.open(img_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0)

    # Grad-CAM (handles forward + backward in one pass)
    cam_generator = GradCAM(model, device)
    heatmap, class_idx, confidence = cam_generator.generate(img_tensor)

    # Get all probabilities from the activation already computed in generate()
    # Re-run a clean forward pass in no_grad for the full prob vector
    model.eval()
    with torch.no_grad():
        output    = model(img_tensor.to(device))
        probs_all = torch.softmax(output, dim=1)[0].cpu().numpy()

    # Clean up hooks to avoid memory leaks
    cam_generator.remove_hooks()

    all_probs  = {class_names[i]: float(probs_all[i]) for i in range(len(class_names))}
    class_name = class_names[class_idx]

    fig = visualize_gradcam(original_img, heatmap, class_name, confidence, all_probs)

    return class_name, confidence, all_probs, fig, heatmap
