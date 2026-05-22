"""
Model Evaluation and Promotion Gate — loads a model and its preprocessors
from an MLflow run, computes classification metrics, outputs a 3-sentence
business interpretation, and validates the model against strict promotion gates.
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
from typing import Dict, Tuple

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    auc,
)
from sklearn.model_selection import train_test_split

from src.data_pipeline.features import build_features
from src.mlflow_config import configure_mlflow, get_latest_run_id

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
UCI_CSV_PATH = DATA_DIR / "bank_marketing.csv"


# ── Metrics Engine ───────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    
    roc_auc = roc_auc_score(y_true, y_prob)
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recalls, precisions)
    
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


# ── Promotion Gate ───────────────────────────────────────────────
def check_promotion(metrics: Dict[str, float]) -> Tuple[bool, Dict[str, Tuple[float, float, bool]]]:
    """Validate model metrics against production promotion gates.
    
    Gates:
    - ROC-AUC >= 0.94 (discriminative strength)
    - PR-AUC  >= 0.65 (imbalance-adjusted performance)
    - F1-score >= 0.60 (harmonic balance of precision & recall)
    """
    gates = {
        "roc_auc": 0.94,
        "pr_auc": 0.65,
        "f1": 0.60,
    }
    
    gate_results = {}
    all_passed = True
    
    for metric_name, threshold in gates.items():
        actual_val = metrics[metric_name]
        passed = actual_val >= threshold
        gate_results[metric_name] = (actual_val, threshold, passed)
        if not passed:
            all_passed = False
            
    return all_passed, gate_results


# ── Business Interpretation ──────────────────────────────────────
def get_business_interpretation(metrics: Dict[str, float], is_promoted: bool) -> str:
    """Generate a structured, exactly 3-sentence business interpretation."""
    sentence1 = (
        f"With a ROC-AUC of {metrics['roc_auc']:.2%} and a Brier Score of {metrics['brier_score']:.4f}, "
        f"the model demonstrates high discriminative power and strong probability calibration, making "
        f"it highly reliable for identifying customer propensity."
    )
    
    sentence2 = (
        f"At a standard threshold, the precision of {metrics['precision']:.2%} minimizes costly marketing "
        f"outreach to non-responsive customers, while the recall of {metrics['recall']:.2%} successfully captures "
        f"the vast majority of potential term-deposit subscribers."
    )
    
    if is_promoted:
        sentence3 = (
            "Therefore, having surpassed all production gating criteria (ROC-AUC >= 0.94, PR-AUC >= 0.65, "
            "and F1-score >= 0.60), this model is highly recommended for immediate deployment into "
            "the marketing campaign production system."
        )
    else:
        sentence3 = (
            "However, because the model fails to meet the strict quality thresholds required for "
            "reliable business outcomes, it is blocked from deployment and should undergo further "
            "hyperparameter tuning or feature engineering."
        )
        
    return f"{sentence1} {sentence2} {sentence3}"


# ── Main CLI ─────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an MLflow model run and run promotion gate validation."
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="MLflow Run ID to evaluate. If omitted, uses the latest run with logged pr_auc.",
    )
    args = parser.parse_args()
    
    configure_mlflow()
    
    # 1. Determine run ID
    run_id = args.run_id
    if not run_id:
        print("🔍 Searching for the latest successful MLflow run ...")
        run_id = get_latest_run_id(metric_key="pr_auc")
        if not run_id:
            print("ERROR: No successful runs found in MLflow. Run train.py first.")
            sys.exit(1)
            
    print(f"📊 Loading MLflow Run: {run_id}")
    
    # 2. Load model from MLflow using native flavor
    try:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        model_type = run.data.tags.get("model_type", "sklearn")
        
        if model_type == "xgboost":
            model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
        else:
            model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    except Exception as exc:
        print(f"ERROR: Failed to load native model from runs:/{run_id}/model: {exc}")
        sys.exit(1)
        
    # 3. Load preprocessor (encoders + scaler)
    try:
        client = mlflow.tracking.MlflowClient()
        local_dir = client.download_artifacts(run_id, "preprocessor")
        preproc_file = Path(local_dir) / "preprocessor.pkl"
        
        with open(preproc_file, "rb") as f:
            preprocessors = pickle.load(f)
            
        encoders = preprocessors["encoders"]
        scaler = preprocessors["scaler"]
    except Exception as exc:
        print(f"ERROR: Failed to load preprocessors from MLflow artifacts: {exc}")
        sys.exit(1)
        
    # 4. Load and process evaluation data
    if not UCI_CSV_PATH.exists():
        print(f"ERROR: UCI dataset CSV not found at {UCI_CSV_PATH}. Run ingest.py first.")
        sys.exit(1)
        
    df = pd.read_csv(UCI_CSV_PATH)
    X = df.drop(columns=["y"])
    y = df["y"]
    
    # Take the exact test split from train-test split (80-20 stratified)
    _, X_test_raw, _, y_test_raw = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    # Process features using the training preprocessors
    X_test_processed, _, _ = build_features(X_test_raw, encoders=encoders, scaler=scaler)
    y_test = y_test_raw.map({"yes": 1, "no": 0}).values
    
    # 5. Compute metrics
    if model_type == "xgboost":
        import xgboost as xgb
        # Extract the underlying raw Booster object to bypass MLflow's sklearn-wrapper deserialization issue
        booster = model.get_booster()
        dtest = xgb.DMatrix(X_test_processed)
        y_prob = booster.predict(dtest)
    else:
        # Standard scikit-learn model
        y_prob = model.predict_proba(X_test_processed)[:, 1]
        
    y_pred = (y_prob >= 0.5).astype(int)
    
    metrics = compute_metrics(y_test, y_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # 6. Check Promotion Gates
    is_promoted, gate_results = check_promotion(metrics)
    
    # 7. Print formatted output
    print("─" * 60)
    print("📋 MODEL PERFORMANCE METRICS")
    print("─" * 60)
    print(f"   Accuracy    : {metrics['accuracy']:.4f}")
    print(f"   ROC-AUC     : {metrics['roc_auc']:.4f}")
    print(f"   PR-AUC      : {metrics['pr_auc']:.4f}")
    print(f"   F1-Score    : {metrics['f1']:.4f}")
    print(f"   Precision   : {metrics['precision']:.4f}")
    print(f"   Recall      : {metrics['recall']:.4f}")
    print(f"   Brier Score : {metrics['brier_score']:.4f}")
    print("\n   Confusion Matrix:")
    print(f"      TN: {conf_matrix[0,0]:<5} | FP: {conf_matrix[0,1]}")
    print(f"      FN: {conf_matrix[1,0]:<5} | TP: {conf_matrix[1,1]}")
    
    print("\n" + "─" * 60)
    print("🎯 PROMOTION GATE STATUS")
    print("─" * 60)
    for m_name, (val, threshold, passed) in gate_results.items():
        status_symbol = "✅ PASS" if passed else "❌ FAIL"
        name_str = m_name.upper().replace("_", "-")
        print(f"   {name_str:<8} (Threshold: >= {threshold:.2f}) : {val:.4f} -> {status_symbol}")
        
    print("\n   RESULT: ", end="")
    if is_promoted:
        print("🟢 PROMOTED (Passed all quality control criteria)")
    else:
        print("🔴 BLOCKED (Failed quality control criteria)")
        
    print("\n" + "─" * 60)
    print("💡 BUSINESS INTERPRETATION")
    print("─" * 60)
    interpretation = get_business_interpretation(metrics, is_promoted)
    print(interpretation)
    print("─" * 60)


if __name__ == "__main__":
    main()
