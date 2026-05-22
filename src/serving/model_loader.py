from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

import joblib
import mlflow


def _load_local_fallback() -> Any | None:
    for p in [
        Path("models/improved_model.joblib"),
        Path("models/baseline_model.joblib"),
        Path("models/model.joblib"),
    ]:
        if p.exists():
            return joblib.load(p)
    return None


def load_model(run_id: str | None = None) -> Tuple[Any, str, str]:
    model = _load_local_fallback()
    if model is None:
        raise RuntimeError("No local fallback model found in models/*.joblib")
    loaded_at = datetime.now(timezone.utc).isoformat()
    return model, run_id or os.getenv("MODEL_RUN_ID") or "local-fallback", loaded_at


def load_scaler(run_id: str) -> Any:
    for p in [
        Path("models/improved_scaler.joblib"),
        Path("models/baseline_scaler.joblib"),
        Path("models/scaler.joblib"),
    ]:
        if p.exists():
            return joblib.load(p)
    return None


def warm_up(model: Any, scaler: Any) -> None:
    import numpy as np
    n_features = getattr(model, "n_features_in_", 18)
    dummy = np.zeros((1, n_features), dtype=float)
    _ = model.predict_proba(dummy)
