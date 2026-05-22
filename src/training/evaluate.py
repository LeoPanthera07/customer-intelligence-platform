from __future__ import annotations

import sys
from pathlib import Path as _Path

ROOT = _Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
from pathlib import Path
from typing import Dict, Tuple

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data_pipeline.features import compute_contact_features, encode_categoricals, get_feature_names, bin_age, scale_numerics
from src.mlflow_config import setup_mlflow

DATA_PATH = Path("data") / "uci_bank_marketing.csv"
THRESHOLD_MAIN = 0.4
LATENCY_THRESHOLD_MS = 200.0


def load_data() -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(DATA_PATH, sep=";")
    y = (df["y"] == "yes").astype(int).to_numpy()
    return df, y


def engineer_features(df: pd.DataFrame) -> np.ndarray:
    df_fe = df.copy()
    df_fe = encode_categoricals(df_fe)
    df_fe = bin_age(df_fe)
    df_fe = compute_contact_features(df_fe)
    df_fe, _ = scale_numerics(df_fe)
    return df_fe[get_feature_names()].to_numpy(dtype=float)


def latest_run_id_by_name(run_name: str) -> str:
    client = MlflowClient()
    exp = client.get_experiment_by_name("customer-intel-experiments")
    if exp is None:
        raise RuntimeError("Could not find experiment customer-intel-experiments.")
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string='tags.mlflow.runName = "{}"'.format(run_name),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError(f"No MLflow run found with name {run_name!r}.")
    return runs.iloc[0]["run_id"]


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "f1_at_threshold": f1_score(y_true, preds),
        "brier_score": brier_score_loss(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, preds),
    }


def measure_latency_ms(model, X_sample: np.ndarray, repeats: int = 50) -> float:
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = model.predict_proba(X_sample)[:, 1]
    return ((time.perf_counter() - t0) / repeats) * 1000.0


def check_promotion(baseline_metrics: Dict[str, float], improved_metrics: Dict[str, float], latency_ms: float) -> Tuple[bool, str]:
    pr_auc_delta = improved_metrics["pr_auc"] - baseline_metrics["pr_auc"]
    f1_delta = improved_metrics["f1_at_threshold"] - baseline_metrics["f1_at_threshold"]
    reasons = []
    if pr_auc_delta < 0.03:
        reasons.append(f"PR-AUC improvement too small: Δ={pr_auc_delta:.4f} (< 0.03).")
    if f1_delta < -0.02:
        reasons.append(f"F1 dropped too much: Δ={f1_delta:.4f} (< -0.02 allowed).")
    if latency_ms > LATENCY_THRESHOLD_MS:
        reasons.append(f"Latency too high: {latency_ms:.2f}ms (> {LATENCY_THRESHOLD_MS:.0f}ms).")
    if reasons:
        return False, " ".join(reasons)
    return True, f"PR-AUC Δ={pr_auc_delta:.4f}, F1 Δ={f1_delta:.4f}, latency={latency_ms:.2f}ms."


def business_interpretation(baseline_metrics: Dict[str, float], improved_metrics: Dict[str, float]) -> None:
    print("\n=== Business interpretation for campaign ROI ===")
    print(f"- ROC-AUC improved from {baseline_metrics['roc_auc']:.3f} to {improved_metrics['roc_auc']:.3f}, so ranking is better.")
    print(f"- PR-AUC improved from {baseline_metrics['pr_auc']:.3f} to {improved_metrics['pr_auc']:.3f}, so top-scored leads are richer in true converters.")
    print(f"- F1 at threshold {THRESHOLD_MAIN:.2f} changed from {baseline_metrics['f1_at_threshold']:.3f} to {improved_metrics['f1_at_threshold']:.3f}, showing the precision/recall trade-off.")


def demonstrate_blocked_case(baseline_metrics: Dict[str, float], X: np.ndarray, y: np.ndarray) -> None:
    print("\n=== Demonstrating BLOCKED gate with degraded model (n_estimators=5) ===")
    split = int(0.8 * len(y))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    pos = y_train.sum()
    neg = len(y_train) - pos
    spw = float(neg / pos) if pos > 0 else 1.0
    degraded_model = XGBClassifier(
        n_estimators=5,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=spw,
        n_jobs=-1,
        random_state=123,
        tree_method="hist",
    )
    degraded_model.fit(X_train, y_train)
    y_proba = degraded_model.predict_proba(X_val)[:, 1]
    degraded_metrics = compute_metrics(y_val, y_proba, THRESHOLD_MAIN)
    latency_ms = measure_latency_ms(degraded_model, X_val[:1])
    promoted, reason = check_promotion(baseline_metrics, degraded_metrics, latency_ms)
    if promoted:
        print("⚠️ Degraded model unexpectedly passed gate; adjust thresholds or seed.")
    else:
        print(f"🚫 BLOCKED — reason: {reason}")


def main() -> None:
    setup_mlflow()

    baseline_run_id = latest_run_id_by_name("baseline")
    improved_run_id = latest_run_id_by_name("improved")
    print(f"Baseline run_id: {baseline_run_id}")
    print(f"Improved run_id: {improved_run_id}")

    df, y = load_data()
    X = engineer_features(df)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    baseline_model = LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1, solver="lbfgs", random_state=42)
    baseline_model.fit(X_train, y_train)

    pos = y_train.sum()
    neg = len(y_train) - pos
    spw = float(neg / pos) if pos > 0 else 1.0
    improved_model = XGBClassifier(
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
    improved_model.fit(X_train, y_train)

    p_base = baseline_model.predict_proba(X_val)[:, 1]
    p_impr = improved_model.predict_proba(X_val)[:, 1]

    baseline_metrics = compute_metrics(y_val, p_base, THRESHOLD_MAIN)
    improved_metrics = compute_metrics(y_val, p_impr, THRESHOLD_MAIN)

    print("\n=== Baseline metrics ===")
    print(f"ROC-AUC: {baseline_metrics['roc_auc']:.3f}")
    print(f"PR-AUC: {baseline_metrics['pr_auc']:.3f}")
    print(f"F1@{THRESHOLD_MAIN:.2f}: {baseline_metrics['f1_at_threshold']:.3f}")
    print(f"Brier score: {baseline_metrics['brier_score']:.4f}")
    print("Confusion matrix:")
    print(baseline_metrics["confusion_matrix"])

    print("\n=== Improved metrics ===")
    print(f"ROC-AUC: {improved_metrics['roc_auc']:.3f}")
    print(f"PR-AUC: {improved_metrics['pr_auc']:.3f}")
    print(f"F1@{THRESHOLD_MAIN:.2f}: {improved_metrics['f1_at_threshold']:.3f}")
    print(f"Brier score: {improved_metrics['brier_score']:.4f}")
    print("Confusion matrix:")
    print(improved_metrics["confusion_matrix"])

    business_interpretation(baseline_metrics, improved_metrics)

    latency_ms = measure_latency_ms(improved_model, X_val[:1])
    print(f"\nInference latency (1 row): {latency_ms:.2f} ms")

    promoted, reason = check_promotion(baseline_metrics, improved_metrics, latency_ms)
    if promoted:
        print(f"\n✅ PROMOTED — {reason}")
    else:
        print(f"\n🚫 BLOCKED — reason: {reason}")

    demonstrate_blocked_case(baseline_metrics, X, y)


if __name__ == "__main__":
    main()
