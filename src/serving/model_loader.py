"""
Model Loader — retrieves the latest promoted model run and its preprocessors
from the central MLflow tracking store.
"""

import sys
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.mlflow_config import configure_mlflow, get_latest_run_id

# ── Configuration ────────────────────────────────────────────────
configure_mlflow()


def load_promoted_model() -> Tuple[Any, Dict[str, LabelEncoder], StandardScaler, str]:
    """Retrieve the latest promoted model and its preprocessors from MLflow.

    Returns
    -------
    model : The loaded scikit-learn or XGBoost model.
    encoders : Dict of fitted LabelEncoders.
    scaler : Fitted StandardScaler.
    run_id : The MLflow run ID of the model.
    """
    # Find the latest run with pr_auc logged (our promotion metric)
    run_id = get_latest_run_id(metric_key="pr_auc")
    if not run_id:
        raise RuntimeError("ERROR: No promoted runs found in MLflow. Please run training first.")
        
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    model_type = run.data.tags.get("model_type", "sklearn")
    
    # 1. Load the model natively
    try:
        if model_type == "xgboost":
            model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
            # Repair deserialized XGBClassifier missing attributes
            if not hasattr(model, "n_classes_"):
                if hasattr(model, "classes_"):
                    model.n_classes_ = len(model.classes_)
                else:
                    model.n_classes_ = 2
        else:
            model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    except Exception as exc:
        raise RuntimeError(f"Failed to load native model from runs:/{run_id}/model: {exc}")
        
    # 2. Download and load the preprocessors artifact
    try:
        local_dir = client.download_artifacts(run_id, "preprocessor")
        preproc_path = Path(local_dir) / "preprocessor.pkl"
        
        with open(preproc_path, "rb") as f:
            preprocessors = pickle.load(f)
            
        encoders = preprocessors["encoders"]
        scaler = preprocessors["scaler"]
    except Exception as exc:
        raise RuntimeError(f"Failed to load preprocessors from MLflow artifacts for run {run_id}: {exc}")
        
    return model, encoders, scaler, run_id
