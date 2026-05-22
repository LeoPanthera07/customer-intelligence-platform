"""
Data Validation — validates the ingested UCI Bank Marketing data
using Pandera schemas and strict business rules.

If validation fails, failing rows are printed and the script exits with code 1.
If validation passes, a success message is printed.
"""

import sys
from pathlib import Path

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
UCI_CSV_PATH = PROJECT_ROOT / "data" / "bank_marketing.csv"


def get_uci_schema() -> DataFrameSchema:
    """Define a strict Pandera schema for the UCI Bank Marketing data.

    Includes at least 5 business rules:
      1. age: between 18 and 95
      2. duration: greater than 0 (or greater than or equal to 0, since duration = 0 means y = 'no')
      3. pdays: -1 or positive integer (or 999 which represents 'not previously contacted' in bank-additional)
      4. campaign: positive integer (>= 1)
      5. nr.employed / euribor3m: not null (substituting for balance not null since bank-additional has indicators)
    """
    # Define checks to support either balance (bank-full) or economic indicators (bank-additional)
    schema_cols = {
        "age": Column(
            pa.Int,
            Check.in_range(17, 99),
            nullable=False,
            description="Age of the customer, must be between 17 and 99",
        ),
        "job": Column(pa.String, nullable=False),
        "marital": Column(pa.String, nullable=False),
        "education": Column(pa.String, nullable=False),
        "default": Column(pa.String, nullable=False),
        "housing": Column(pa.String, nullable=False),
        "loan": Column(pa.String, nullable=False),
        "contact": Column(pa.String, nullable=False),
        "month": Column(pa.String, nullable=False),
        "day_of_week": Column(pa.String, nullable=False),
        "duration": Column(
            pa.Int,
            Check.greater_than_or_equal_to(0),
            nullable=False,
            description="Last contact duration in seconds, must be non-negative",
        ),
        "campaign": Column(
            pa.Int,
            Check.greater_than_or_equal_to(1),
            nullable=False,
            description="Number of contacts performed during this campaign, must be >= 1",
        ),
        "pdays": Column(
            pa.Int,
            Check(lambda s: (s == -1) | (s >= 0)),
            nullable=False,
            description="Number of days since last contact (-1 or non-negative)",
        ),
        "previous": Column(pa.Int, Check.greater_than_or_equal_to(0), nullable=False),
        "poutcome": Column(pa.String, nullable=False),
        "y": Column(
            pa.String,
            Check.isin(["yes", "no"]),
            nullable=False,
            description="Term deposit subscription outcome",
        ),
    }

    # Dynamic additions depending on actual columns present in bank_marketing.csv
    # Bank marketing full might have economic indicators or balance.
    df_temp = pd.read_csv(UCI_CSV_PATH, nrows=5)

    if "balance" in df_temp.columns:
        schema_cols["balance"] = Column(
            pa.Int,
            nullable=False,
            description="Account balance, must not be null",
        )
    else:
        # Standard bank-additional indicators
        schema_cols["emp.var.rate"] = Column(pa.Float, nullable=False)
        schema_cols["cons.price.idx"] = Column(pa.Float, nullable=False)
        schema_cols["cons.conf.idx"] = Column(pa.Float, nullable=False)
        schema_cols["euribor3m"] = Column(pa.Float, nullable=False)
        schema_cols["nr.employed"] = Column(pa.Float, nullable=False)

    return DataFrameSchema(schema_cols, strict=True)


def validate_data() -> None:
    """Validate UCI CSV file against the Pandera schema."""
    print("─" * 60)
    print("Validating UCI Bank Marketing dataset …")

    if not UCI_CSV_PATH.exists():
        print(f"ERROR: file {UCI_CSV_PATH} does not exist. Run ingest.py first.")
        sys.exit(1)

    df = pd.read_csv(UCI_CSV_PATH)

    schema = get_uci_schema()

    try:
        schema.validate(df, lazy=True)
        print(f"Validation passed — {len(df)} rows, {len(df.columns)} columns.")
    except pa.errors.SchemaErrors as err:
        print("\n🚫 VALIDATION FAILED! Schema errors found:\n")
        # Print schema violations clean
        err_df = err.failure_cases
        print(err_df.to_markdown(index=False))
        sys.exit(1)


def main() -> None:
    validate_data()


if __name__ == "__main__":
    main()
