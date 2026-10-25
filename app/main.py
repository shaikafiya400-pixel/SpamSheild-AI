
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model.joblib"
HTML_PATH = BASE_DIR / "app" / "index.html"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="SpamShield AI",
    description="Machine learning API for spam detection",
    version="1.0.0"
)


class MessageInput(BaseModel):
    text: str = Field(min_length=1, max_length=10000)


@app.get("/")
def home():
    if HTML_PATH.exists():
        return FileResponse(HTML_PATH)
    return {"message": "SpamShield AI API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(data: MessageInput):
    message = data.text.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )

    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]
    classes = list(model.classes_)

    spam_probability = float(
        probabilities[classes.index("spam")]
    )

    return {
        "prediction": prediction,
        "is_spam": prediction == "spam",
        "spam_probability": round(spam_probability, 4)
    }
