import io
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b2

MODEL_DIR = Path(__file__).parent / "models"
CATEGORIES = ["bottle", "screw", "carpet"]
DEVICE = "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def build_model():
    m = efficientnet_b2(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(m.classifier[1].in_features, 2),
    )
    return m

models = {}
for cat in CATEGORIES:
    m = build_model()
    m.load_state_dict(torch.load(MODEL_DIR / f"effnet_{cat}_v2.pth",
                                 map_location=DEVICE))
    m.eval()
    models[cat] = m

app = FastAPI(title="Industrial Anomaly Detection API", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok", "categories": CATEGORIES}

@app.post("/predict/{category}")
async def predict(category: str, file: UploadFile = File(...)):
    if category not in models:
        raise HTTPException(404, f"category must be one of {CATEGORIES}")
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "could not read the file as an image")

    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(models[category](x), dim=1)[0]
    p_defect = float(probs[1])

    result = {
        "category": category,
        "prediction": "defective" if p_defect >= 0.5 else "normal",
        "defect_probability": round(p_defect, 4),
    }
    if category == "screw":
        result["warning"] = ("Screw model missed many defects in testing "
                             "(recall 0.60). Demo only.")
    return result
