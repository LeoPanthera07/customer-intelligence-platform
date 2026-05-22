from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.data_pipeline.features import bin_age, compute_contact_features, encode_categoricals, get_feature_names, scale_numerics
from src.serving.model_loader import load_model, load_scaler, warm_up
from src.serving.schemas import (
    BatchScoreItem,
    BatchScoreRequest,
    BatchScoreResponse,
    CustomerFeatures,
    MetricsResponse,
    PredictionResponse,
)

app = FastAPI(title="Customer Intelligence Platform - ML API", version="0.1.0")

REQUEST_LOG: List[Dict[str, Any]] = []
SERVICE_STATE: Dict[str, Any] = {"ready": False, "model_version": "unknown", "loaded_at": None, "scaler": None, "model": None}


def _to_dataframe(customer: CustomerFeatures) -> pd.DataFrame:
    return pd.DataFrame([customer.dict()])


def _feature_frame(df: pd.DataFrame, scaler=None) -> np.ndarray:
    df2 = encode_categoricals(df.copy())
    df2 = bin_age(df2)
    df2 = compute_contact_features(df2)
    df2, scaler_used = scale_numerics(df2, scaler=scaler)
    return df2[get_feature_names()].to_numpy(dtype=float), scaler_used


def _band(prob: float) -> str:
    if prob < 0.3:
        return "low"
    if prob <= 0.6:
        return "medium"
    return "high"


@app.on_event("startup")
def startup_event() -> None:
    run_id = None
    model, run_id, loaded_at = load_model(None)
    scaler = load_scaler(run_id)
    warm_up(model, scaler)
    SERVICE_STATE.update({"ready": True, "model_version": run_id, "loaded_at": loaded_at, "model": model, "scaler": scaler})


@app.middleware("http")
async def log_requests(request, call_next):
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        REQUEST_LOG.append({"method": request.method, "path": request.url.path, "status": status, "latency_ms": latency_ms})


@app.get("/health")
def health():
    if not SERVICE_STATE["ready"]:
        return JSONResponse(status_code=503, content={"status": "starting", "model_version": "unknown", "index_version": "unknown", "uptime_seconds": 0.0})
    return {"status": "ok", "model_version": SERVICE_STATE["model_version"], "index_version": "unknown", "uptime_seconds": 0.0}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    try:
        X, _ = _feature_frame(_to_dataframe(customer), scaler=SERVICE_STATE["scaler"])
        model = SERVICE_STATE["model"]
        t0 = time.perf_counter()
        prob = float(model.predict_proba(X)[0, 1])
        latency_ms = (time.perf_counter() - t0) * 1000.0
        pred = int(prob >= 0.4)
        return PredictionResponse(prediction=pred, probability=prob, threshold_decision=("subscribe" if pred else "not_subscribe"), model_version=SERVICE_STATE["model_version"], latency_ms=latency_ms)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/batch-score", response_model=BatchScoreResponse)
def batch_score(req: BatchScoreRequest):
    customers = req.customers or []
    if not customers:
        raise HTTPException(status_code=400, detail="Provide customers list.")
    df = pd.DataFrame([c.dict() for c in customers])
    X, _ = _feature_frame(df, scaler=SERVICE_STATE["scaler"])
    probs = SERVICE_STATE["model"].predict_proba(X)[:, 1]
    results = [BatchScoreItem(id=i, conversion_band=_band(float(p)), probability=float(p)) for i, p in enumerate(probs)]
    return BatchScoreResponse(results=results)


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    lats = [r["latency_ms"] for r in REQUEST_LOG] or [0.0]
    preds = Counter(["ok" for _ in REQUEST_LOG])
    return MetricsResponse(latency_p50=float(np.percentile(lats, 50)), latency_p99=float(np.percentile(lats, 99)), request_count=len(REQUEST_LOG), error_count=sum(1 for r in REQUEST_LOG if r["status"] >= 400), prediction_distribution=dict(preds))
