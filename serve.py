from __future__ import annotations
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from src.utils import read_text
import yaml

app = FastAPI(title="MLOps Starter Model API", version="1.0.0")

class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., description="Sepal length in cm")
    sepal_width: float = Field(..., description="Sepal width in cm")
    petal_length: float = Field(..., description="Petal length in cm")
    petal_width: float = Field(..., description="Petal width in cm")

# Lazy model loader
_model = None
_feature_order = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]


def load_model():
    global _model
    if _model is None:
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        model_path = read_text(cfg["paths"]["latest_ptr"])  # absolute path
        _model = joblib.load(model_path)
    return _model

@app.get("/")
async def root():
    return {"status": "ok", "message": "Use POST /predict"}

@app.post("/predict")
async def predict(items: List[IrisFeatures]):
    model = load_model()
    import numpy as np
    X = np.array([[
        it.sepal_length, it.sepal_width, it.petal_length, it.petal_width
    ] for it in items])
    preds = model.predict(X)
    return {"predictions": preds.tolist()}