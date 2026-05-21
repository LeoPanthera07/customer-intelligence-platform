import os
from typing import Optional

import mlflow


def setup_mlflow() -> None:
    """
    Configure MLflow tracking and experiment.

    - Uses MLFLOW_TRACKING_URI if set, otherwise defaults to local ./mlruns.
    - Uses MLFLOW_EXPERIMENT_NAME and MLFLOW_ARTIFACT_ROOT from env with sensible defaults.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "customer-intel-experiments")
    artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlruns")

    mlflow.set_tracking_uri(tracking_uri)

    # Ensure experiment exists with the configured artifact location
    mlflow.set_experiment(
        experiment_name=experiment_name,
        artifact_location=artifact_root,
    )


def get_active_run_id(fallback_env_var: str = "MODEL_RUN_ID") -> Optional[str]:
    """
    Return the currently active MLflow run_id if a run is active.

    If no run is active, fall back to the environment variable given by
    `fallback_env_var` (default: MODEL_RUN_ID). Returns None if neither is set.
    """
    active_run = mlflow.active_run()
    if active_run is not None:
        return active_run.info.run_id

    env_run_id = os.getenv(fallback_env_var)
    return env_run_id or None
