"""
Data validation for the UCI Bank Marketing dataset using Pandera.

- Enforces a strict schema for bank-full.csv (as saved by ingest.py).
- Applies business rules:
  * age between 18 and 95
  * balance not null
  * duration greater than 0
  * pdays is -1 or a positive integer
  * campaign is a positive integer
- On failure: prints failing rows and exits with code 1.
- On success: prints "Validation passed — N rows, M columns.".
"""

import sys
from pathlib import Path

import pandas as pd
import pandera as pa
from pandera import Column, Check
from pandera.errors import SchemaError


DATA_PATH = Path("data") / "uci_bank_marketing.csv"


def build_schema() -> pa.DataFrameSchema:
    """
    Build a strict Pandera schema for the UCI bank-full.csv data.[web:68]
    """
    return pa.DataFrameSchema(
        {
            "age": Column(int, Check.in_range(18, 95), nullable=False),
            "job": Column(str, nullable=False),
            "marital": Column(str, nullable=False),
            "education": Column(str, nullable=False),
            "default": Column(str, nullable=False),
            "balance": Column(int, Check.not_null(), nullable=False),
            "housing": Column(str, nullable=False),
            "loan": Column(str, nullable=False),
            "contact": Column(str, nullable=False),
            "day": Column(int, Check.in_range(1, 31), nullable=False),
            "month": Column(str, nullable=False),
            "duration": Column(int, Check.gt(0), nullable=False),
            "campaign": Column(int, Check.ge(1), nullable=False),
            "pdays": Column(
                int,
                Check(lambda s: ((s == -1) | (s >= 0)), element_wise=True),
                nullable=False,
            ),
            "previous": Column(int, Check.ge(0), nullable=False),
            "poutcome": Column(str, nullable=False),
            "y": Column(str, Check.isin(["yes", "no"]), nullable=False),
        },
        strict=True,
        name="uci_bank_marketing_schema",
    )


def validate() -> None:
    if not DATA_PATH.exists():
        print(f"ERROR: Expected data file not found at {DATA_PATH}.")
        print("Run: python src/data_pipeline/ingest.py")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH, sep=";")

    schema = build_schema()

    try:
        validated = schema.validate(df, lazy=True)
    except SchemaError as err:
        print("❌ Validation failed.")
        print("Schema error details:")
        print(err)
        if hasattr(err, "failure_cases"):
            print("\nFailing rows (first 20):")
            print(err.failure_cases.head(20))
        sys.exit(1)

    print(
        f"Validation passed — {len(validated)} rows, {validated.shape[1]} columns."
    )


if __name__ == "__main__":
    validate()
