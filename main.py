import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from asl_detector.models.mobilenetv2 import ASLMobilenetv2
from asl_detector.constants import CLASSES, HF_MODEL_FILENAME, HF_MODEL_REPO_ID, IMAGE_SIZE, MODEL_WEIGHTS_PATH

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load model once at startup
model = ASLMobilenetv2(num_classes=len(CLASSES), dropout_rate=0.2).model
if HF_MODEL_REPO_ID:
    weights_path = hf_hub_download(
        repo_id=HF_MODEL_REPO_ID,
        filename=HF_MODEL_FILENAME,
    )
else:
    weights_path = MODEL_WEIGHTS_PATH
model.load_weights(weights_path)

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
