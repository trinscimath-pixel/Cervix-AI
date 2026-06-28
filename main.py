from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
from torchvision import models, transforms
import torch.nn as nn

app = FastAPI(title="CerviVision AI API")

# ==========================
# Model
# ==========================
model = models.efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(1280, 5)

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
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
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

    with torch.inference_mode():
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
