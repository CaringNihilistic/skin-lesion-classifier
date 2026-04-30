"""
app_hf.py — Hugging Face Spaces deployment (Gradio interface).

This replaces the FastAPI+HTML frontend for HF Spaces deployment.
HF Spaces runs Gradio apps natively — no server setup needed.

Deploy:
  1. Create new Space on huggingface.co (Gradio SDK)
  2. Upload: app_hf.py, models/best_model.pth, src/model.py, src/gradcam.py
  3. Add requirements.txt
  4. Space auto-deploys
"""

import gradio as gr
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pathlib import Path
import sys

from model import build_model
from gradcam import GradCAM, overlay_heatmap

# ── CONFIG ────────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/best_model.pth"

CLASS_INFO = {
    "Actinic_Keratosis":    "Rough, scaly patch. Pre-cancerous. Sun exposure related.",
    "Basal_Cell_Carcinoma": "Most common skin cancer. Rarely spreads, needs treatment.",
    "Benign_Keratosis":     "Non-cancerous growth. Usually harmless.",
    "Dermatofibroma":       "Harmless fibrous nodule. Benign.",
    "Melanocytic_Nevi":     "Common mole. Usually benign — monitor for changes.",
    "Melanoma":             "⚠️ Most dangerous skin cancer. Early detection critical.",
    "Vascular_Lesion":      "Blood vessel abnormality in skin. Usually benign.",
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
checkpoint  = torch.load(MODEL_PATH, map_location=DEVICE)
CLASS_NAMES = checkpoint["class_names"]
model       = build_model(num_classes=len(CLASS_NAMES), pretrained=False).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# ── PREDICT FUNCTION ──────────────────────────────────────────────────────────
def predict(pil_img):
    if pil_img is None:
        return None, "Please upload an image.", None

    original_np = np.array(pil_img.convert("RGB"))
    img_tensor  = transform(pil_img.convert("RGB")).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        output = model(img_tensor.to(DEVICE))
        probs  = torch.softmax(output, dim=1)[0].cpu().numpy()

    class_idx  = int(probs.argmax())
    class_name = CLASS_NAMES[class_idx]
    confidence = float(probs[class_idx])

    # Grad-CAM
    cam_gen  = GradCAM(model, DEVICE)
    heatmap, _, _ = cam_gen.generate(img_tensor)
    overlaid = overlay_heatmap(original_np, heatmap, alpha=0.4)

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Prediction: {class_name.replace('_',' ')}  |  Confidence: {confidence*100:.1f}%",
                 fontsize=13, fontweight="bold")
    axes[0].imshow(original_np); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
    axes[2].imshow(overlaid); axes[2].set_title("Overlay"); axes[2].axis("off")
    plt.tight_layout()

    # Probabilities for label output
    label_output = f"**{class_name.replace('_', ' ')}** — {confidence*100:.1f}% confidence\n\n"
    label_output += CLASS_INFO.get(class_name, "") + "\n\n---\n\n"
    label_output += "**All probabilities:**\n"
    sorted_probs = sorted(zip(CLASS_NAMES, probs), key=lambda x: x[1], reverse=True)
    for cls, prob in sorted_probs:
        bar = "█" * int(prob * 20)
        label_output += f"`{cls.replace('_',' '):25s}` {bar} {prob*100:.1f}%\n"

    label_output += "\n\n⚠️ *For research purposes only. Not a medical diagnostic tool.*"

    return fig, label_output


# ── GRADIO UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Skin Lesion Classifier") as demo:
    gr.Markdown("""
    # 🔬 Skin Lesion Classifier
    **EfficientNetB0 + Grad-CAM** · Trained on HAM10000 (ISIC 2018) · 7 skin condition classes

    Upload a dermoscopy image to get a prediction with Grad-CAM explainability visualization.

    > ⚠️ **Disclaimer:** For educational and research purposes only. Not a substitute for professional medical advice.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Upload Dermoscopy Image")
            submit_btn = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=2):
            result_text = gr.Markdown(label="Result")
            gradcam_out = gr.Plot(label="Grad-CAM Visualization")

    submit_btn.click(fn=predict, inputs=img_input, outputs=[gradcam_out, result_text])

    gr.Examples(
        examples=[],  # Add example images here after training
        inputs=img_input
    )

demo.launch()
