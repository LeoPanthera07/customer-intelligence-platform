"""
Model Training — trains a baseline LogisticRegression and an improved XGBoost model.
Logs hyper-parameters, test/train metrics, calibration curves, threshold plots,
feature importances, and serialized preprocessors to MLflow.
"""

import os
import pickle
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from src.data_pipeline.features import build_features, get_feature_names
from src.mlflow_config import configure_mlflow

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
UCI_CSV_PATH = DATA_DIR / "bank_marketing.csv"
UCI_HASH_PATH = DATA_DIR / "uci_hash.txt"
PLOT_DIR = PROJECT_ROOT / "data" / "temp_plots"


# ── Metric Calculation ───────────────────────────────────────────
def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Calculate comprehensive performance metrics for a model."""
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate Area Under curves
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


# ── Rich Visualization Functions ─────────────────────────────────
def plot_feature_importance(model: Any, feature_names: List[str], save_path: Path, is_xgb: bool = False) -> None:
    """Generate and save a premium horizontal feature importance chart."""
    plt.figure(figsize=(10, 6), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    if is_xgb:
        importances = model.feature_importances_
    else:
        # For Logistic Regression, use absolute coefficients
        importances = np.abs(model.coef_[0])
        # Normalise to sum to 1 for visual equivalence
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
            
    indices = np.argsort(importances)
    
    # Elegant blue gradient aesthetic
    colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(importances)))
    
    plt.barh(range(len(importances)), importances[indices], color=colors, edgecolor="#2c3e50", height=0.7)
    plt.yticks(range(len(importances)), [feature_names[i] for i in indices], fontsize=9)
    plt.xlabel("Relative Importance / Influence Scale", fontsize=10, fontweight="bold", labelpad=8)
    plt.title("Feature Contribution Analysis", fontsize=12, fontweight="bold", pad=12, color="#2c3e50")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, model_name: str, save_path: Path) -> None:
    """Generate and save an elegant calibration curve diagram."""
    plt.figure(figsize=(7, 7), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    # Plot perfect calibration reference line
    plt.plot([0, 1], [0, 1], linestyle="--", color="#7f8c8d", label="Perfectly Calibrated", alpha=0.8)
    
    # Plot model's calibration curve
    plt.plot(prob_pred, prob_true, marker="o", linewidth=2.5, color="#1abc9c", label=model_name)
    
    # Style the chart premium
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Mean Predicted Probability", fontsize=10, fontweight="bold", labelpad=8)
    plt.ylabel("Fraction of True Positive Outcomes", fontsize=10, fontweight="bold", labelpad=8)
    plt.title(f"Probability Calibration Analysis — {model_name}", fontsize=12, fontweight="bold", pad=12, color="#2c3e50")
    plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#bdc3c7")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_threshold_analysis(y_true: np.ndarray, y_prob: np.ndarray, model_name: str, save_path: Path) -> None:
    """Generate and save a premium threshold trade-off analysis curve."""
    plt.figure(figsize=(9, 6), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Avoid zero division when calculating f1 metrics along thresholds
    f1_scores = np.zeros_like(thresholds)
    for idx, t in enumerate(thresholds):
        y_pred_t = (y_prob >= t).astype(int)
        f1_scores[idx] = f1_score(y_true, y_pred_t, zero_division=0)
        
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    # Plot nice curves
    plt.plot(thresholds, precisions[:-1], color="#3498db", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], color="#e74c3c", label="Recall", linewidth=2)
    plt.plot(thresholds, f1_scores, color="#2ecc71", label="F1-Score", linewidth=2.5)
    
    # Highlight optimal F1 threshold
    plt.axvline(x=best_threshold, color="#8e44ad", linestyle=":", linewidth=2, 
                label=f"Optimum F1 ({best_f1:.3f} @ t={best_threshold:.2f})")
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("Decision Probability Threshold", fontsize=10, fontweight="bold", labelpad=8)
    plt.ylabel("Score Value", fontsize=10, fontweight="bold", labelpad=8)
    plt.title(f"Classification Boundary Optimization — {model_name}", fontsize=12, fontweight="bold", pad=12, color="#2c3e50")
    plt.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#bdc3c7")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# ── Training and Logging Engine ──────────────────────────────────
def train_and_log_run(
    run_name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    encoders: Dict[str, LabelEncoder],
    scaler: StandardScaler,
    dataset_hash: str,
    is_xgb: bool = False,
) -> str:
    """Train the model, evaluate it, generate visualizations, and log everything to MLflow."""
    print(f"\n🚀 Training model run: {run_name} ...")
    
    # Ensure plots output directory exists
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        
        # 1. Fit the model
        model.fit(X_train, y_train)
        
        # 2. Get predictions
        y_train_prob = model.predict_proba(X_train)[:, 1]
        y_test_prob = model.predict_proba(X_test)[:, 1]
        
        # 3. Calculate metrics
        train_metrics = calculate_metrics(y_train, y_train_prob)
        test_metrics = calculate_metrics(y_test, y_test_prob)
        
        # 4. Log hyperparameters
        if is_xgb:
            params = model.get_params()
            # Log only essential/configured parameters to avoid clutter
            logged_keys = ["n_estimators", "max_depth", "learning_rate", "scale_pos_weight", "random_state", "eval_metric"]
            mlflow.log_params({k: params[k] for k in logged_keys if k in params})
        else:
            params = model.get_params()
            logged_keys = ["C", "max_iter", "random_state", "class_weight"]
            mlflow.log_params({k: params[k] for k in logged_keys if k in params})
            
        # 5. Log tags
        mlflow.set_tag("dataset_sha256", dataset_hash)
        mlflow.set_tag("model_type", "xgboost" if is_xgb else "logistic_regression")
        
        # 6. Log metrics
        for k, v in train_metrics.items():
            mlflow.log_metric(f"train_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        # Main target metric for search runs
        mlflow.log_metric("pr_auc", test_metrics["pr_auc"])
        
        # 7. Log model artifact
        if is_xgb:
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")
            
        # 8. Custom package and log preprocessing artefacts (encoders + scaler)
        preprocessors = {
            "encoders": encoders,
            "scaler": scaler,
        }
        preproc_path = PLOT_DIR / "preprocessor.pkl"
        with open(preproc_path, "wb") as f:
            pickle.dump(preprocessors, f)
        mlflow.log_artifact(str(preproc_path), artifact_path="preprocessor")
        
        # 9. Generate and log beautiful diagnostic plots
        feature_names = get_feature_names()
        
        feat_img = PLOT_DIR / f"{run_name}_feature_importance.png"
        plot_feature_importance(model, feature_names, feat_img, is_xgb=is_xgb)
        mlflow.log_artifact(str(feat_img), artifact_path="diagnostic_plots")
        
        cal_img = PLOT_DIR / f"{run_name}_calibration.png"
        plot_calibration(y_test, y_test_prob, run_name, cal_img)
        mlflow.log_artifact(str(cal_img), artifact_path="diagnostic_plots")
        
        thresh_img = PLOT_DIR / f"{run_name}_threshold_analysis.png"
        plot_threshold_analysis(y_test, y_test_prob, run_name, thresh_img)
        mlflow.log_artifact(str(thresh_img), artifact_path="diagnostic_plots")
        
        # Clean up local temporary files
        for path in [preproc_path, feat_img, cal_img, thresh_img]:
            if path.exists():
                path.unlink()
                
        print(f"✅ Success! Logged to MLflow Run ID: {run_id}")
        print(f"   Test ROC-AUC: {test_metrics['roc_auc']:.4f} | Test PR-AUC: {test_metrics['pr_auc']:.4f} | Test F1: {test_metrics['f1']:.4f}")
        return run_id


# ── CLI ──────────────────────────────────────────────────────────
def main() -> None:
    print("─" * 60)
    print("UCI Bank Marketing Model Training Pipeline starting …")
    
    # Verify dataset exists
    if not UCI_CSV_PATH.exists():
        print(f"ERROR: Dataset CSV not found at {UCI_CSV_PATH}. Ingest data first.")
        sys.exit(1)
        
    # Read raw dataset
    df = pd.read_csv(UCI_CSV_PATH)
    
    # Fetch dataset hash for traceability
    dataset_hash = "unknown"
    if UCI_HASH_PATH.exists():
        dataset_hash = UCI_HASH_PATH.read_text(encoding="utf-8").strip()
        
    # Split into features (X) and label (y)
    X = df.drop(columns=["y"])
    y = df["y"]
    
    # 80-20 Stratified Train/Test split
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    # Build feature preprocessing pipeline
    X_train_processed, encoders, scaler = build_features(X_train_raw, encoders=None, scaler=None)
    X_test_processed, _, _ = build_features(X_test_raw, encoders=encoders, scaler=scaler)
    
    # Map raw targets ("yes", "no") to (1, 0)
    y_train = y_train_raw.map({"yes": 1, "no": 0}).values
    y_test = y_test_raw.map({"yes": 1, "no": 0}).values
    
    # Centralised MLflow tracking setup
    configure_mlflow()
    
    # --- MODEL 1: Baseline Logistic Regression ---
    lr_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    )
    train_and_log_run(
        run_name="logistic_regression_baseline",
        model=lr_model,
        X_train=X_train_processed,
        y_train=y_train,
        X_test=X_test_processed,
        y_test=y_test,
        encoders=encoders,
        scaler=scaler,
        dataset_hash=dataset_hash,
        is_xgb=False
    )
    
    # --- MODEL 2: Improved XGBoost Classifier ---
    # Calculate dynamic scale_pos_weight for imbalance handling
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    scale_pos_weight = float(num_neg / num_pos)
    
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )
    
    train_and_log_run(
        run_name="xgboost_improved",
        model=xgb_model,
        X_train=X_train_processed,
        y_train=y_train,
        X_test=X_test_processed,
        y_test=y_test,
        encoders=encoders,
        scaler=scaler,
        dataset_hash=dataset_hash,
        is_xgb=True
    )
    
    print("─" * 60)
    print("✅ Model training complete. All runs tracked in MLflow.")


if __name__ == "__main__":
    main()
