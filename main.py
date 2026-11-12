
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import os

app = FastAPI(title="MALARIASCAN AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "best_v2.pt")
model = YOLO(MODEL_PATH)

@app.get("/")
def home():
    return {
        "message": "MALARIASCAN AI API is running",
        "model": "YOLO26m-best_v2"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in [
        "image/jpeg", "image/png", "image/webp"
    ]:
        raise HTTPException(400, "Format gambar tidak didukung")

    try:
        image_bytes = await file.read()
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        results = model.predict(
            source=image,
            conf=0.15,
            iou=0.5,
            imgsz=640
        )

        detections = []

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class_id": class_id,
                "class_name": model.names[class_id],
                "confidence": round(confidence, 4),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                }
            })

        return {
            "model_name": "YOLO26m-best_v2",
            "total_detections": len(detections),
            "detections": detections
        }

    except Exception as e:
        raise HTTPException(500, str(e))
