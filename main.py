import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from asl_detector.models.mobilenetv2 import ASLMobilenetv2
from asl_detector.constants import CLASSES, IMAGE_SIZE, MODEL_WEIGHTS_PATH

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load model once at startup
model = ASLMobilenetv2(num_classes=len(CLASSES), dropout_rate=0.2).model
model.load_weights(MODEL_WEIGHTS_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 5):
    """Predict ASL gesture from image. Pass ?top_k=X in URL to get top X results."""
    # Process image
    img = Image.open(BytesIO(await file.read())).convert("RGB").resize(IMAGE_SIZE)
    batch = preprocess_input(np.expand_dims(np.array(img, dtype=np.float32), axis=0))
    
    # Predict
    preds = model.predict(batch, verbose=0)[0]
    
    # Get top_k results
    top_indices = np.argsort(preds)[-top_k:][::-1]
    return [{"class": CLASSES[i], "confidence": float(preds[i])} for i in top_indices]