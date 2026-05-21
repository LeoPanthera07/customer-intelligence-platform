"""
Data ingestion for the Customer Intelligence Platform.

- Downloads the UCI Bank Marketing dataset (bank-full.csv).
- Downloads a CFPB Consumer Complaint sample via the public API.
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


# UCI: Bank Marketing dataset (older bank-full.csv that includes "balance")
# Reference: UCI ML Repository Bank Marketing dataset.[web:68]
UCI_BANK_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-full.csv"
)

# CFPB: Consumer Complaint Database sample via public search API.
# We request CSV with a configurable size parameter.[web:60][web:69]
CFPB_BASE_URL = (
    "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
)


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
    """
    Stream a file from `url` to `dest`.

    Returns:
        elapsed_seconds, content_length_bytes (if known, else -1)
    """
    t0 = time.perf_counter()
    with requests.get(url, stream=True, timeout=60, **request_kwargs) as r:
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
    csv_path = data_dir / "uci_bank_marketing.csv"
    print(f"Downloading UCI Bank Marketing data from: {UCI_BANK_URL}")
    elapsed, size_bytes = _download_to_path(UCI_BANK_URL, csv_path)

    # UCI bank-full.csv uses ";" as separator
    df = pd.read_csv(csv_path, sep=";")
    sha = _sha256_file(csv_path)
    (data_dir / "uci_hash.txt").write_text(sha + os.linesep, encoding="utf-8")

    print(
        f"[UCI] file={csv_path} rows={len(df)} cols={df.shape[1]} "
        f"size_bytes={size_bytes} sha256={sha} download_time_sec={elapsed:.2f}"
    )


def ingest_cfpb_sample(data_dir: Path, sample_size: int) -> None:
    """
    Download a CFPB complaint sample as CSV using the public search API.

    Note: The API supports a size parameter for limiting results.[web:69]
    """
    params = {
        "size": sample_size,
        "format": "csv",
    }
    csv_path = data_dir / "cfpb_complaints_sample.csv"
    print(
        f"Downloading CFPB complaint sample from: {CFPB_BASE_URL} "
        f"(size={sample_size})"
    )
    elapsed, size_bytes = _download_to_path(CFPB_BASE_URL, csv_path, params=params)

    df = pd.read_csv(csv_path)
    sha = _sha256_file(csv_path)
    (data_dir / "cfpb_hash.txt").write_text(sha + os.linesep, encoding="utf-8")

    print(
        f"[CFPB] file={csv_path} rows={len(df)} cols={df.shape[1]} "
        f"size_bytes={size_bytes} sha256={sha} download_time_sec={elapsed:.2f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest UCI Bank Marketing and CFPB complaint datasets into ./data "
            "with SHA-256 hash sidecars."
        )
    )
    parser.add_argument(
        "--uci-url",
        type=str,
        default=UCI_BANK_URL,
        help=f"Override URL for UCI bank marketing CSV (default: {UCI_BANK_URL})",
    )
    parser.add_argument(
        "--cfpb-size",
        "--sample",
        dest="cfpb_size",
        type=int,
        default=5000,
        help="Number of CFPB complaint rows to request via API (default: 5000).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = _ensure_data_dir()

    # Allow overriding UCI URL via CLI while keeping default constant
    global UCI_BANK_URL
    UCI_BANK_URL = args.uci_url

    ingest_uci_bank(data_dir)
    ingest_cfpb_sample(data_dir, args.cfpb_size)


if __name__ == "__main__":
    main()
