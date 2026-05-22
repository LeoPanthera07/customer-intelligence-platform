"""
Feature Engineering — reusable, tested feature transformation functions
for the UCI Bank Marketing dataset.

CRITICAL: scale_numerics() accepts a pre-fitted scaler so that training
and serving use the exact same transformation. At train time, pass
scaler=None to fit a new scaler. At serve time, pass the saved scaler.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ── Categorical encoding ────────────────────────────────────────
# Columns that need label encoding.
CATEGORICAL_COLS: list[str] = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month",
    "day_of_week", "poutcome",
]


def encode_categoricals(
    df: pd.DataFrame,
    encoders: Optional[dict[str, LabelEncoder]] = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode all categorical columns.

    Parameters
    ----------
    df : DataFrame with raw categorical strings.
    encoders : Optional dict mapping column name → fitted LabelEncoder.
               If None, new encoders are fitted (training mode).
               If provided, they are used to transform (serving mode).

    Returns
    -------
    df : DataFrame with encoded integers replacing the original strings.
    encoders : The dict of fitted LabelEncoders (save this for serving).
    """
    df = df.copy()
    if encoders is None:
        encoders = {}
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
    else:
        for col in CATEGORICAL_COLS:
            if col in df.columns and col in encoders:
                le = encoders[col]
                # Handle unseen labels gracefully at serve time.
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x, _k=known, _le=le: (
                        _le.transform([x])[0] if x in _k else -1
                    )
                )
    return df, encoders


# ── Age binning ──────────────────────────────────────────────────
def bin_age(df: pd.DataFrame) -> pd.DataFrame:
    """Create an ``age_group`` column: young (≤30), mid (31–55), senior (>55)."""
    df = df.copy()
    conditions = [
        df["age"] <= 30,
        (df["age"] > 30) & (df["age"] <= 55),
        df["age"] > 55,
    ]
    labels = ["young", "mid", "senior"]
    df["age_group"] = np.select(conditions, labels, default="mid")
    # Convert to numeric ordinal for modelling.
    age_map = {"young": 0, "mid": 1, "senior": 2}
    df["age_group"] = df["age_group"].map(age_map).astype(int)
    return df


# ── Contact features ────────────────────────────────────────────
def compute_contact_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive two contact-related features.

    ``days_since_contact``
        If pdays == 999 (never contacted), set to -1 to distinguish
        from genuinely low values. Otherwise keep pdays as-is.

    ``contact_intensity``
        campaign × (previous + 1) — captures total contact effort.
    """
    df = df.copy()
    df["days_since_contact"] = df["pdays"].apply(
        lambda x: -1 if x == 999 else x
    )
    df["contact_intensity"] = df["campaign"] * (df["previous"] + 1)
    return df


# ── Numeric scaling ──────────────────────────────────────────────
NUMERIC_COLS: list[str] = [
    "age", "duration", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed",
    # Engineered features
    "age_group", "days_since_contact", "contact_intensity",
]


def scale_numerics(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Fit or apply a StandardScaler to all numeric feature columns.

    Parameters
    ----------
    df : DataFrame after encoding + feature engineering.
    scaler : If None, a new scaler is fitted (training).
             If provided, the scaler is used to transform (serving).

    Returns
    -------
    df : DataFrame with scaled numeric columns.
    scaler : The fitted StandardScaler (save this for serving).
    """
    df = df.copy()
    # Only scale columns that actually exist in the dataframe.
    cols_to_scale = [c for c in NUMERIC_COLS if c in df.columns]

    if scaler is None:
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    else:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return df, scaler


# ── Feature name list ────────────────────────────────────────────
def get_feature_names() -> list[str]:
    """Return the ordered list of final feature column names.

    This is the canonical feature order used during training and
    serving — any mismatch will cause prediction errors.
    """
    return CATEGORICAL_COLS + NUMERIC_COLS


# ── Full pipeline ────────────────────────────────────────────────
def build_features(
    df: pd.DataFrame,
    encoders: Optional[dict[str, LabelEncoder]] = None,
    scaler: Optional[StandardScaler] = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder], StandardScaler]:
    """Run the full feature pipeline: encode → bin → contact → scale.

    Convenience wrapper that calls each step in order.

    Returns
    -------
    df : Fully transformed DataFrame (only feature columns retained).
    encoders : Fitted label encoders.
    scaler : Fitted standard scaler.
    """
    df, encoders = encode_categoricals(df, encoders)
    df = bin_age(df)
    df = compute_contact_features(df)
    df, scaler = scale_numerics(df, scaler)

    # Keep only feature columns in canonical order.
    feature_cols = get_feature_names()
    available = [c for c in feature_cols if c in df.columns]
    df = df[available]

    return df, encoders, scaler
