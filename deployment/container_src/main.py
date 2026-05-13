"""FastAPI inference server for the fraud-detector model.

Loads `fraud-detector-v1.onnx` at startup and serves POST /predict on port 8080.
Returns the same probabilities as the local sklearn XGBClassifier within 1e-4
(verified by scripts/export_onnx.py).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/app/fraud-detector-v1.onnx"))
MODEL_VERSION = "v1"
THRESHOLD = 0.9274014830589294
N_FEATURES = 29

app = FastAPI(title="Fraud Detector", version=MODEL_VERSION)

session: ort.InferenceSession | None = None
input_name: str | None = None
proba_output_idx: int | None = None


@app.on_event("startup")
def _load_model() -> None:
    global session, input_name, proba_output_idx
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.get_outputs()
    for i, out in enumerate(outputs):
        if len(out.shape) == 2 and out.shape[-1] == 2:
            proba_output_idx = i
            break
    if proba_output_idx is None:
        proba_output_idx = len(outputs) - 1


class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=N_FEATURES, max_length=N_FEATURES)


class PredictResponse(BaseModel):
    probability: float
    prediction: int
    threshold: float
    model_version: str


@app.get("/")
def root() -> dict:
    return {
        "service": "fraud-detector",
        "model_version": MODEL_VERSION,
        "threshold": THRESHOLD,
        "n_features": N_FEATURES,
        "endpoints": ["POST /predict", "GET /health"],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if session is not None else "loading"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if session is None:
        raise HTTPException(503, "model not loaded")
    arr = np.asarray(req.features, dtype=np.float32).reshape(1, N_FEATURES)
    if not np.isfinite(arr).all():
        raise HTTPException(400, "features must all be finite numbers")
    outputs = session.run(None, {input_name: arr})
    proba = float(outputs[proba_output_idx][0, 1])
    return PredictResponse(
        probability=proba,
        prediction=1 if proba >= THRESHOLD else 0,
        threshold=THRESHOLD,
        model_version=MODEL_VERSION,
    )
