"""
Model training for the Customer Intelligence Platform.

- Loads UCI Bank Marketing data from data/uci_bank_marketing.csv.[web:84]
- Applies feature engineering from src.data_pipeline.features.
- Trains two models:
  * Baseline: LogisticRegression (C=1.0, max_iter=1000)
  * Improved: XGBClassifier (n_estimators=200, max_depth=5,
      learning_rate=0.05, scale_pos_weight=auto)
- Logs both runs to MLflow:
  - run name: "baseline" and "improved"
  - dataset hash
  - all params
  - metrics including:
      ROC-AUC, PR-AUC, F1 at threshold=0.4,
      precision/recall at thresholds 0.3, 0.4, 0.5
  - artifacts:
      fitted model, scaler, validation predictions CSV,
      feature importance plot (improved),
      calibration curve plot (improved).
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure project root is on sys.path so `import src.*` works when running this file directly.
ROOT = _Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import os
import time
from pathlib import Path
from typing import Dict, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow import sklearn as mlflow_sklearn
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data_pipeline.features import (
    compute_contact_features,
    encode_categoricals,
    get_feature_names,
    bin_age,
    scale_numerics,
)
from src.mlflow_config import setup_mlflow


DATA_PATH = Path("data") / "uci_bank_marketing.csv"
UCI_HASH_PATH = Path("data") / "uci_hash.txt"

THRESHOLD_MAIN = 0.4
THRESHOLDS_ANALYSIS = (0.3, 0.4, 0.5)


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run src/data_pipeline/ingest.py first."
        )
    df = pd.read_csv(DATA_PATH, sep=";")
    y = (df["y"] == "yes").astype(int)
    return df, y


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, object]:
    df_fe = df.copy()
    df_fe = encode_categoricals(df_fe)
    df_fe = bin_age(df_fe)
    df_fe = compute_contact_features(df_fe)
    df_fe, scaler = scale_numerics(df_fe)

    feature_names = get_feature_names()
    X = df_fe[feature_names].to_numpy(dtype=float)
    return df_fe, X, scaler


def dataset_hash() -> str:
    if UCI_HASH_PATH.exists():
        return UCI_HASH_PATH.read_text().strip()
    return "unknown"


def train_baseline(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> Dict[str, float]:
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        n_jobs=-1,
        solver="lbfgs",
        random_state=42,
    )
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    val_proba = model.predict_proba(X_val)[:, 1]

    metrics = compute_metrics(y_val, val_proba, THRESHOLD_MAIN)
    metrics["fit_time_sec"] = fit_time

    return {"model": model, "metrics": metrics, "val_proba": val_proba}


def auto_scale_pos_weight(y: np.ndarray) -> float:
    # scale_pos_weight = (negative / positive)
    pos = y.sum()
    neg = len(y) - pos
    return float(neg / pos) if pos > 0 else 1.0


def train_improved(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> Dict[str, float]:
    spw = auto_scale_pos_weight(y_train)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=spw,
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
    )
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    val_proba = model.predict_proba(X_val)[:, 1]

    metrics = compute_metrics(y_val, val_proba, THRESHOLD_MAIN)
    metrics["fit_time_sec"] = fit_time
    metrics["scale_pos_weight"] = spw

    return {"model": model, "metrics": metrics, "val_proba": val_proba}


def compute_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float
) -> Dict[str, float]:
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    preds_main = (y_proba >= threshold).astype(int)
    f1_main = f1_score(y_true, preds_main)
    brier = brier_score_loss(y_true, y_proba)

    metrics: Dict[str, float] = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1_at_threshold": f1_main,
        "brier_score": brier,
    }

    for thr in THRESHOLDS_ANALYSIS:
        preds_thr = (y_proba >= thr).astype(int)
        precision = precision_score(y_true, preds_thr, zero_division=0)
        recall = recall_score(y_true, preds_thr, zero_division=0)
        metrics[f"precision_at_{thr}"] = precision
        metrics[f"recall_at_{thr}"] = recall

    return metrics


def log_threshold_analysis(metrics: Dict[str, float]) -> None:
    for thr in THRESHOLDS_ANALYSIS:
        mlflow.log_metric(f"precision_at_{thr}", metrics[f"precision_at_{thr}"])
        mlflow.log_metric(f"recall_at_{thr}", metrics[f"recall_at_{thr}"])


def save_predictions_artifact(
    y_val: np.ndarray, val_proba: np.ndarray, filename: str
) -> None:
    df_pred = pd.DataFrame({"y_true": y_val, "y_proba": val_proba})
    out_dir = Path("artifacts") / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    df_pred.to_csv(out_path, index=False)
    mlflow.log_artifact(str(out_path), artifact_path="predictions")


def save_scaler_artifact(scaler: object, filename: str) -> None:
    out_dir = Path("artifacts") / "scalers"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    joblib.dump(scaler, out_path)
    mlflow.log_artifact(str(out_path), artifact_path="scaler")


def save_feature_importance_plot(
    model: XGBClassifier, feature_names: list[str], filename: str
) -> None:
    import matplotlib.pyplot as plt

    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in order]
    sorted_importances = importances[order]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_importances)), sorted_importances)
    plt.xticks(range(len(sorted_importances)), sorted_names, rotation=90)
    plt.tight_layout()

    out_dir = Path("artifacts") / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    plt.savefig(out_path)
    plt.close()
    mlflow.log_artifact(str(out_path), artifact_path="plots")


def save_calibration_curve_plot(
    y_true: np.ndarray, y_proba: np.ndarray, filename: str
) -> None:
    import matplotlib.pyplot as plt

    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="uniform")

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.legend()
    plt.tight_layout()

    out_dir = Path("artifacts") / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    plt.savefig(out_path)
    plt.close()
    mlflow.log_artifact(str(out_path), artifact_path="plots")


def main() -> None:
    setup_mlflow()

    df, y = load_data()
    df_fe, X, scaler = engineer_features(df)

    feature_names = get_feature_names()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base_params = {
        "dataset_hash": dataset_hash(),
        "features_count": len(feature_names),
        "threshold_main": THRESHOLD_MAIN,
    }

    # Baseline run
    with mlflow.start_run(run_name="baseline") as run:
        mlflow.log_params(base_params)
        mlflow.log_params(
            {
                "model_type": "logistic_regression",
                "C": 1.0,
                "max_iter": 1000,
                "solver": "lbfgs",
            }
        )

        result = train_baseline(X_train, y_train, X_val, y_val)
        metrics = result["metrics"]
        val_proba = result["val_proba"]
        model = result["model"]

        mlflow.log_metric("roc_auc", metrics["roc_auc"])
        mlflow.log_metric("pr_auc", metrics["pr_auc"])
        mlflow.log_metric("f1_at_threshold", metrics["f1_at_threshold"])
        mlflow.log_metric("brier_score", metrics["brier_score"])
        mlflow.log_metric("fit_time_sec", metrics["fit_time_sec"])
        log_threshold_analysis(metrics)

        save_predictions_artifact(y_val, val_proba, "baseline_val_predictions.csv")
        save_scaler_artifact(scaler, "baseline_scaler.joblib")

        mlflow_sklearn.log_model(model, artifact_path="model")
        print(f"Baseline run_id: {run.info.run_id}")

    # Improved run
    with mlflow.start_run(run_name="improved") as run:
        mlflow.log_params(base_params)
        improved_result = train_improved(X_train, y_train, X_val, y_val)
        metrics = improved_result["metrics"]
        val_proba = improved_result["val_proba"]
        model = improved_result["model"]

        mlflow.log_params(
            {
                "model_type": "xgboost",
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.05,
                "scale_pos_weight": metrics["scale_pos_weight"],
            }
        )

        mlflow.log_metric("roc_auc", metrics["roc_auc"])
        mlflow.log_metric("pr_auc", metrics["pr_auc"])
        mlflow.log_metric("f1_at_threshold", metrics["f1_at_threshold"])
        mlflow.log_metric("brier_score", metrics["brier_score"])
        mlflow.log_metric("fit_time_sec", metrics["fit_time_sec"])
        log_threshold_analysis(metrics)

        save_predictions_artifact(y_val, val_proba, "improved_val_predictions.csv")
        save_scaler_artifact(scaler, "improved_scaler.joblib")
        save_feature_importance_plot(
            model, feature_names, "improved_feature_importance.png"
        )
        save_calibration_curve_plot(
            y_val, val_proba, "improved_calibration_curve.png"
        )

        mlflow_sklearn.log_model(model, artifact_path="model")
        print(f"Improved run_id: {run.info.run_id}")


if __name__ == "__main__":
    main()
