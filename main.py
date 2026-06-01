import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from typing import List

from asl_detector.models.mobilenetv2 import ASLMobilenetv2
from asl_detector.constants import CLASSES, HF_MODEL_FILENAME, HF_MODEL_REPO_ID, IMAGE_SIZE, MODEL_WEIGHTS_PATH


class PredictionItem(BaseModel):
    label: str = Field(..., examples=["F"])
    confidence: float = Field(..., examples=[0.832])


class PredictionResponse(BaseModel):
    filename: str = Field(..., examples=["F80.jpg"])
    top_k: int = Field(..., examples=[5])
    predictions: List[PredictionItem]


app = FastAPI(
    title="ASL Alphabet Gesture Prediction API",
    description="API for predicting ASL alphabet gesture classes from uploaded images.",
    version="1.0.0",
)
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

@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict ASL alphabet gesture from image",
    description="Upload an ASL alphabet gesture image and return the top-k predicted classes with confidence scores.",
)
async def predict(
    file: UploadFile = File(..., description="Image file containing an ASL alphabet gesture."),
    top_k: int = Query(
        5,
        description="Number of top predictions to return.",
        ge=1,
        le=len(CLASSES),
    ),
):
    """Predict ASL alphabet gesture from image. Pass ?top_k=X in URL to get top X results."""
    # Process image
    img = Image.open(BytesIO(await file.read())).convert("RGB").resize(IMAGE_SIZE)
    batch = preprocess_input(np.expand_dims(np.array(img, dtype=np.float32), axis=0))
    
    # Predict
    preds = model.predict(batch, verbose=0)[0]
    
    # Get top_k results
    top_indices = np.argsort(preds)[-top_k:][::-1]
    predictions = [
        {"label": CLASSES[i], "confidence": float(preds[i])}
        for i in top_indices
    ]

    return {
        "filename": file.filename,
        "top_k": top_k,
        "predictions": predictions,
    }
