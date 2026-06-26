from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms
app = FastAPI(title="CerviVision AI API")
# ==========================
# Model
# ==========================
model = models.mobilenet_v3_large(weights=None)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    5
)
model.load_state_dict(
    torch.load("model.pth", map_location="cpu")
)
model.eval()
# ==========================
# Classes
# ==========================
class_names = [
    "ASC-US",
    "HSIL",
    "LSIL",
    "NILM",
    "SCC"
]
# ==========================
# Transform
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
# ==========================
# Home
# ==========================
@app.get("/")
def home():
    return {
        "message": "CerviVision AI API is running"
    }
# ==========================
# Prediction Endpoint
# ==========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0]

    result = {
        class_names[i]: float(probs[i])
        for i in range(5)
    }

    prediction = class_names[
        torch.argmax(probs).item()
    ]

    return {
        "prediction": prediction,
        "probabilities": result
    }
