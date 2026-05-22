"""
Data ingestion for the Customer Intelligence Platform.

- Downloads the UCI Bank Marketing dataset (bank-full.csv) from the classic UCI bank.zip.
- Downloads a CFPB Consumer Complaint sample via the public API (or generates a synthetic sample if the API fails).
- Saves them to data/ with SHA-256 hash sidecar files:
  - data/uci_bank_marketing.csv + data/uci_hash.txt
  - data/cfpb_complaints_sample.csv + data/cfpb_hash.txt
- Prints: file path, row count, SHA-256 hash, download time.
"""

import argparse
import hashlib
import os
import time
from pathlib import Path
from typing import Tuple

import pandas as pd
import requests


# Classic UCI URL that contains bank-full.csv inside bank.zip.[web:85][web:96]
UCI_BANK_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
UCI_BANK_CSV_IN_ZIP = "bank-full.csv"

# CFPB complaint database search API endpoint.[web:60]
CFPB_API_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"


def _ensure_data_dir() -> Path:
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_to_path(url: str, dest: Path, **request_kwargs) -> Tuple[float, int]:
    t0 = time.perf_counter()
    with requests.get(url, stream=True, timeout=120, **request_kwargs) as r:
        r.raise_for_status()
        total = 0
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                f.write(chunk)
                total += len(chunk)
    elapsed = time.perf_counter() - t0
    return elapsed, total if total > 0 else -1


def ingest_uci_bank(data_dir: Path) -> None:
    import zipfile

    zip_path = data_dir / "uci_bank_marketing.zip"
    csv_path = data_dir / "uci_bank_marketing.csv"

    print(f"Downloading UCI Bank Marketing ZIP from: {UCI_BANK_URL}")
    elapsed, size_bytes = _download_to_path(UCI_BANK_URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(UCI_BANK_CSV_IN_ZIP) as src, csv_path.open("wb") as dst:
            dst.write(src.read())

    df = pd.read_csv(csv_path, sep=";")
    sha = _sha256_file(csv_path)
    (data_dir / "uci_hash.txt").write_text(sha + os.linesep, encoding="utf-8")

    print(
        f"[UCI] file={csv_path} rows={len(df)} cols={df.shape[1]} "
        f"size_bytes={size_bytes} sha256={sha} download_time_sec={elapsed:.2f}"
    )


def _synthetic_cfpb_sample(sample_size: int) -> pd.DataFrame:
    products = ["Credit card", "Mortgage", "Checking account", "Student loan", "Debt collection"]
    issues = ["Billing dispute", "Charged fees", "Loan modification", "Incorrect information", "Fraud or scam"]
    companies = ["Bank A", "Bank B", "Bank C", "Lender X", "Servicer Y"]

    rows = []
    for i in range(sample_size):
        rows.append(
            {
                "complaint_id": 9000000 + i,
                "product": products[i % len(products)],
                "company": companies[i % len(companies)],
                "date_received": f"2025-01-{(i % 28) + 1:02d}",
                "issue": issues[i % len(issues)],
                "consumer_complaint_narrative": (
                    f"Sample complaint narrative {i} about {products[i % len(products)]} "
                    f"and {issues[i % len(issues)]} involving {companies[i % len(companies)]}."
                ),
            }
        )
    return pd.DataFrame(rows)


def ingest_cfpb_sample(data_dir: Path, sample_size: int) -> None:
    params = {"size": sample_size, "format": "csv"}
    csv_path = data_dir / "cfpb_complaints_sample.csv"
    print(f"Downloading CFPB complaint sample from: {CFPB_API_URL} (size={sample_size})")
    try:
        elapsed, size_bytes = _download_to_path(CFPB_API_URL, csv_path, params=params)
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"WARNING: CFPB API download failed ({e}); generating synthetic sample for demo.")
        df = _synthetic_cfpb_sample(sample_size)
        df.to_csv(csv_path, index=False)
        elapsed, size_bytes = 0.0, csv_path.stat().st_size

    sha = _sha256_file(csv_path)
    (data_dir / "cfpb_hash.txt").write_text(sha + os.linesep, encoding="utf-8")

    print(
        f"[CFPB] file={csv_path} rows={len(df)} cols={df.shape[1]} "
        f"size_bytes={size_bytes} sha256={sha} download_time_sec={elapsed:.2f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest UCI and CFPB datasets into ./data.")
    parser.add_argument("--sample", type=int, default=5000, help="CFPB sample size (default 5000).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = _ensure_data_dir()
    ingest_uci_bank(data_dir)
    ingest_cfpb_sample(data_dir, args.sample)


if __name__ == "__main__":
    main()
