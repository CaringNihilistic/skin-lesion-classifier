"""
app/main.py — FastAPI backend for skin lesion classification.

Endpoints:
  GET  /           — Serve frontend UI
  POST /predict    — Upload image, get prediction + Grad-CAM heatmap
  GET  /health     — Health check
  GET  /classes    — List all 7 classes

Run locally:
  cd C:/skin-lesion/app
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import io
import json
import base64
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from torchvision import transforms

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model import build_model
from gradcam import GradCAM, overlay_heatmap, visualize_gradcam

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / "models" / "best_model.pth"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 260    # EfficientNetB2

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH}...")
checkpoint  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
CLASS_NAMES = checkpoint["class_names"]
NUM_CLASSES = len(CLASS_NAMES)

model = build_model(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Model loaded on {DEVICE}. Classes: {CLASS_NAMES}")

# ── TRANSFORMS ────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── CLASS DESCRIPTIONS ────────────────────────────────────────────────────────
CLASS_INFO = {
    "Actinic_Keratosis":    "Rough, scaly patch caused by years of sun exposure. Pre-cancerous.",
    "Basal_Cell_Carcinoma": "Most common form of skin cancer. Rarely spreads but needs treatment.",
    "Benign_Keratosis":     "Non-cancerous skin growth. Often appears with age.",
    "Dermatofibroma":       "Harmless fibrous nodule, usually on legs. Benign.",
    "Melanocytic_Nevi":     "Common mole. Usually benign but monitor for changes.",
    "Melanoma":             "Most dangerous form of skin cancer. Early detection is critical.",
    "Vascular_Lesion":      "Abnormality of blood vessels in the skin. Usually benign.",
}

# ── FASTAPI APP ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Skin Lesion Classifier",
    description="EfficientNetB2 + Grad-CAM — 7-class dermoscopy image classification",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── RESPONSE SCHEMA ───────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    predicted_class:   str
    confidence:        float
    description:       str
    all_probabilities: dict
    gradcam_image:     str  # base64 encoded PNG


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":      "ok",
        "device":      str(DEVICE),
        "model":       "EfficientNetB2",
        "num_classes": NUM_CLASSES,
        "classes":     CLASS_NAMES,
    }


@app.get("/classes")
def get_classes():
    return {
        "classes": [
            {"name": name, "description": CLASS_INFO.get(name, "")}
            for name in CLASS_NAMES
        ]
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # ── Validate ──
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 10MB).")

    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image.")

    # ── Preprocess ──
    original_np = np.array(pil_img)
    img_tensor  = transform(pil_img).unsqueeze(0)

    # ── Predict ──
    with torch.no_grad():
        output = model(img_tensor.to(DEVICE))
        probs  = torch.softmax(output, dim=1)[0].cpu().numpy()

    class_idx  = int(probs.argmax())
    class_name = CLASS_NAMES[class_idx]
    confidence = float(probs[class_idx])
    all_probs  = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)}

    # ── Grad-CAM ──
    gradcam_b64 = ""
    try:
        cam_gen  = GradCAM(model, DEVICE)
        heatmap, _, _ = cam_gen.generate(img_tensor)

        fig = visualize_gradcam(original_np, heatmap, class_name, confidence, all_probs)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        gradcam_b64 = base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        print(f"Grad-CAM failed: {e}")

    return PredictionResponse(
        predicted_class=class_name,
        confidence=round(confidence, 4),
        description=CLASS_INFO.get(class_name, ""),
        all_probabilities=all_probs,
        gradcam_image=gradcam_b64,
    )


# ── SERVE FRONTEND ────────────────────────────────────────────────────────────
@app.get("/")
def serve_frontend():
    return FileResponse(Path(__file__).parent / "index.html")
