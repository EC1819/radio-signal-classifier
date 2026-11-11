import os

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, field_validator

from src.features import extract_all_features

MODEL_PATH = "models/model.joblib"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"No model found at {MODEL_PATH}. Run `python src/train.py` first to train and save it."
    )

app = FastAPI()
model = joblib.load(MODEL_PATH)


class Signal(BaseModel):
    i: list[float]
    q: list[float]

    @field_validator("i", "q")
    @classmethod
    def check_length(cls, value):
        if len(value) != 128:
            raise ValueError("must have exactly 128 samples")
        return value


@app.post("/classify")
def classify(signal: Signal):
    X = np.array([[signal.i, signal.q]])
    features = extract_all_features(X)

    probabilities = model.predict_proba(features)[0]
    top3 = np.argsort(probabilities)[::-1][:3]

    return {
        "predictions": [
            {"modulation": str(model.classes_[i]), "probability": float(probabilities[i])}
            for i in top3
        ]
    }
