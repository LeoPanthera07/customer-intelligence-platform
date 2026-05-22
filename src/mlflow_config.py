"""
MLflow configuration — centralises experiment name, tracking URI,
and a helper to retrieve the active run ID.

Every training / evaluation script imports from here so that
the experiment name and artifact root are never hard-coded
in multiple places.
"""

import os

import mlflow
from dotenv import load_dotenv

# Load .env so local development picks up MLFLOW_TRACKING_URI etc.
load_dotenv()

# ── Configuration ────────────────────────────────────────────────
EXPERIMENT_NAME: str = os.getenv(
    "MLFLOW_EXPERIMENT_NAME", "customer-intel-experiments"
)
TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"
)
ARTIFACT_ROOT: str = os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlruns")


def configure_mlflow() -> str:
    """Set the MLflow tracking URI and experiment.

    Returns the experiment ID so callers can reference it in logs.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    return experiment.experiment_id


def get_active_run_id() -> str | None:
    """Return the run_id of the currently active MLflow run, or None."""
    active_run = mlflow.active_run()
    if active_run is None:
        return None
    return active_run.info.run_id


def get_latest_run_id(metric_key: str = "pr_auc") -> str | None:
    """Return the run_id of the latest finished run that logged *metric_key*.

    Useful for model_loader.py when MODEL_RUN_ID is not set — it falls
    back to the most recent successful training run.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.{metric_key} > 0",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        return None
    return str(runs.iloc[0]["run_id"])
