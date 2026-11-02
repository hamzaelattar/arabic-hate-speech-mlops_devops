
from pathlib import Path
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "tfidf_logreg.joblib"

model = None


# Load the model once when the API starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = joblib.load(MODEL_PATH)
    yield
    model = None


# Create FastAPI application
app = FastAPI(
    title="Arabic Hate Speech Detection API",
    description="Hate speech classification using TF-IDF and Logistic Regression",
    version="1.0.0",
    lifespan=lifespan
)


# Request schema
class TweetRequest(BaseModel):
    text: str = Field(min_length=1)


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Arabic Hate Speech Detection API"
    }


# Health check endpoint
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


# Prediction endpoint
@app.post("/predict")
def predict(request: TweetRequest):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    text = request.text

    # Predict class
    prediction = model.predict([text])[0]

    # Predict class probabilities
    probabilities = model.predict_proba([text])[0]

    class_probabilities = {
        str(label): float(probability)
        for label, probability in zip(
            model.classes_,
            probabilities
        )
    }

    return {
        "prediction": prediction.item()
            if hasattr(prediction, "item")
            else prediction,
        "probabilities": class_probabilities
    }
