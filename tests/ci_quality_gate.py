"""
CI Quality Gate — programmatically evaluates model evaluation metrics from MLflow
and data drift status from Evidently to enforce production deployment gates.
Fails the CI build (exit code 1) if criteria are not satisfied.
"""

import sys
import json
from pathlib import Path
import mlflow

from src.mlflow_config import configure_mlflow, get_latest_run_id

# ── Performance Gates ────────────────────────────────────────────
ROC_AUC_THRESHOLD = 0.80
PR_AUC_THRESHOLD = 0.35
MAX_DRIFTED_SHARE = 0.85

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIFT_JSON_PATH = PROJECT_ROOT / "monitoring" / "reports" / "drift_report.json"


def evaluate_quality_gates() -> None:
    """Check performance and drift metrics against strict quality gates."""
    print("═" * 60)
    print("🔍 RUNNING AUTOMATED PRODUCTION QUALITY GATE CHECK ...")
    print("═" * 60)
    
    configure_mlflow()
    client = mlflow.tracking.MlflowClient()
    
    # 1. Fetch latest promoted model run ID
    run_id = get_latest_run_id(metric_key="pr_auc")
    if not run_id:
        print("❌ FAILED: No promoted model runs found in MLflow.")
        sys.exit(1)
        
    run = client.get_run(run_id)
    metrics = run.data.metrics
    model_type = run.data.tags.get("model_type", "unknown")
    
    test_roc_auc = metrics.get("test_roc_auc")
    test_pr_auc = metrics.get("test_pr_auc")
    
    print(f"📋 Model Info: {model_type} (Run ID: {run_id})")
    print(f"   - Test ROC-AUC: {test_roc_auc}")
    print(f"   - Test PR-AUC:  {test_pr_auc}")
    
    # 2. Assert ML metrics satisfy gates
    failed = False
    
    if test_roc_auc is None:
        print("❌ FAILED: 'test_roc_auc' metric is missing in MLflow run.")
        failed = True
    elif test_roc_auc < ROC_AUC_THRESHOLD:
        print(f"❌ FAILED: Test ROC-AUC {test_roc_auc:.4f} is below threshold of {ROC_AUC_THRESHOLD}")
        failed = True
    else:
        print(f"✅ PASSED: Test ROC-AUC satisfies gate (>={ROC_AUC_THRESHOLD})")
        
    if test_pr_auc is None:
        print("❌ FAILED: 'test_pr_auc' metric is missing in MLflow run.")
        failed = True
    elif test_pr_auc < PR_AUC_THRESHOLD:
        print(f"❌ FAILED: Test PR-AUC {test_pr_auc:.4f} is below threshold of {PR_AUC_THRESHOLD}")
        failed = True
    else:
        print(f"✅ PASSED: Test PR-AUC satisfies gate (>={PR_AUC_THRESHOLD})")

    # 3. Check Data Drift status from Evidently report
    print("\n🔍 Checking Data Drift status ...")
    if not DRIFT_JSON_PATH.exists():
        print(f"⚠️ WARNING: Evidently drift JSON report not found at {DRIFT_JSON_PATH}. Skipping drift check.")
    else:
        try:
            with open(DRIFT_JSON_PATH, "r", encoding="utf-8") as f:
                report_dict = json.load(f)
                
            metrics_list = report_dict.get("metrics", [])
            drift_share = None
            
            for m in metrics_list:
                if m.get("metric") == "DatasetDriftMetric":
                    drift_share = m.get("result", {}).get("share_of_drifted_columns")
                    break
            
            if drift_share is not None:
                print(f"   - Share of drifted columns: {drift_share:.2%}")
                if drift_share > MAX_DRIFTED_SHARE:
                    print(f"❌ FAILED: Data drift share {drift_share:.2%} exceeds maximum allowed limit of {MAX_DRIFTED_SHARE:.2%}")
                    failed = True
                else:
                    print(f"✅ PASSED: Data drift share within acceptable boundaries (<={MAX_DRIFTED_SHARE:.2%})")
            else:
                print("⚠️ WARNING: Could not parse DatasetDriftMetric from Evidently report.")
        except Exception as exc:
            print(f"⚠️ WARNING: Failed to parse Evidently report: {exc}")
            
    print("═" * 60)
    if failed:
        print("🛑 SYSTEM STATUS: DEPLOYMENT BLOCKED — Quality gate criteria not met.")
        print("═" * 60)
        sys.exit(1)
    else:
        print("🚀 SYSTEM STATUS: READY TO DEPLOY — All quality gate checks passed!")
        print("═" * 60)
        sys.exit(0)


if __name__ == "__main__":
    evaluate_quality_gates()
