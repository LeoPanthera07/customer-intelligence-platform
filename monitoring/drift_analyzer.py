"""
Evidently Drift Analyzer — monitors data and target drift for the Customer
Intelligence Platform by comparing reference and current (serving) datasets.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
UCI_CSV_PATH = DATA_DIR / "bank_marketing.csv"
REPORTS_DIR = PROJECT_ROOT / "monitoring" / "reports"


def run_drift_analysis() -> None:
    """Run Evidently data drift and target drift analysis and save HTML + JSON reports."""
    print("─" * 60)
    print("🚀 Running Evidently Drift and Data Quality Analyzer ...")

    # 1. Verify dataset exists
    if not UCI_CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset CSV not found at {UCI_CSV_PATH}. Please run ingestion first.")

    # 2. Load dataset
    df = pd.read_csv(UCI_CSV_PATH)
    print(f"📊 Loaded {len(df)} rows from {UCI_CSV_PATH.name}")

    # 3. Simulate Reference vs Current datasets
    # We split 50/50. To make it interesting and realistic, we introduce synthetic drift
    # into the 'current' dataset to demonstrate Evidently's detection capabilities.
    half = len(df) // 2
    reference_df = df.iloc[:half].copy()
    current_df = df.iloc[half:].copy()

    print("⚠️ Injecting synthetic drift into the 'current' dataset ...")
    # Shift numerical distributions slightly to trigger drift
    current_df["age"] = current_df["age"] + np.random.randint(5, 12, size=len(current_df))
    current_df["duration"] = current_df["duration"] * 1.3  # Longer calls in current period
    current_df["campaign"] = current_df["campaign"] + 2    # More touches per customer

    # 4. Configure and run Evidently Report
    # We include DataDriftPreset for input features and TargetDriftPreset for the target column
    print("⏳ Computing Evidently metrics (Data Drift & Target Drift) ...")
    drift_report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
    ])

    drift_report.run(reference_data=reference_df, current_data=current_df)

    # 5. Create reports directory
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 6. Save Reports (HTML & JSON)
    html_path = REPORTS_DIR / "drift_report.html"
    json_path = REPORTS_DIR / "drift_report.json"

    drift_report.save_html(str(html_path))
    
    # Save JSON report
    report_dict = drift_report.as_dict()
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    print("═" * 60)
    print("💡 DRIFT MONITORING REPORT GENERATED")
    print("═" * 60)
    print(f"✅ HTML report: {html_path}")
    print(f"✅ JSON report: {json_path}")
    
    # Programmatic check for drift metrics
    metrics = report_dict["metrics"]
    
    # Find dataset drift metric summary
    data_drift_summary = {}
    for metric_info in metrics:
        if metric_info.get("metric") == "DatasetDriftMetric":
            result = metric_info.get("result", {})
            data_drift_summary = {
                "number_of_features": result.get("number_of_columns"),
                "drifted_features": result.get("number_of_drifted_columns"),
                "share_of_drifted_features": result.get("share_of_drifted_columns"),
                "dataset_drift": result.get("dataset_drift"),
            }
            break
            
    print(f"\n📈 Data Drift Summary:")
    print(f"   - Number of features: {data_drift_summary.get('number_of_features')}")
    print(f"   - Drifted features: {data_drift_summary.get('drifted_features')}")
    print(f"   - Share of drifted features: {data_drift_summary.get('share_of_drifted_features', 0):.2%}")
    print(f"   - Overall Dataset Drift Detected: {data_drift_summary.get('dataset_drift')}")
    print("─" * 60)


if __name__ == "__main__":
    run_drift_analysis()
