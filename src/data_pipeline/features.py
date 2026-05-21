"""
Feature engineering utilities for the campaign conversion model.

Functions:
- encode_categoricals(df) -> df
- bin_age(df) -> df with age_group
- compute_contact_features(df) -> df with contact features
- scale_numerics(df, scaler=None) -> (df_scaled, scaler)
- get_feature_names() -> ordered list of feature columns

CRITICAL: scale_numerics accepts an optional pre-fitted scaler so train
and serve use identical transformations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Categorical columns in the UCI bank-full dataset we will encode.
CATEGORICAL_COLS: Sequence[str] = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

# Base numeric columns (excluding the target y)
NUMERIC_COLS: Sequence[str] = [
    "age",
    "balance",
    "duration",
    "campaign",
    "pdays",
    "previous",
]

# Engineered columns we add
ENGINEERED_COLS: Sequence[str] = [
    "age_group",
    "days_since_contact",
    "contact_intensity",
]


@dataclass
class FeatureConfig:
    categorical_cols: Sequence[str] = tuple(CATEGORICAL_COLS)
    numeric_cols: Sequence[str] = tuple(NUMERIC_COLS)
    engineered_cols: Sequence[str] = tuple(ENGINEERED_COLS)


FEATURE_CONFIG = FeatureConfig()


def encode_categoricals(df: pd.DataFrame, config: FeatureConfig = FEATURE_CONFIG) -> pd.DataFrame:
    """
    Label-encode categorical columns using pandas category codes.

    This is deterministic as long as the underlying string values remain
    consistent between train and serve. We keep the same column names,
    replacing string values by integer codes.
    """
    df = df.copy()
    for col in config.categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes.astype("int32")
    return df


def bin_age(df: pd.DataFrame, config: FeatureConfig = FEATURE_CONFIG) -> pd.DataFrame:
    """
    Create an age_group column with values: "young", "mid", "senior",
    then encode it as an ordered integer.

    - young : age < 30
    - mid   : 30 <= age <= 55
    - senior: age > 55
    """
    df = df.copy()
    if "age" not in df.columns:
        raise KeyError("age column is required to compute age_group.")

    bins = [-np.inf, 29, 55, np.inf]
    labels = ["young", "mid", "senior"]
    df["age_group_str"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)
    df["age_group"] = df["age_group_str"].astype("category").cat.codes.astype("int32")
    df.drop(columns=["age_group_str"], inplace=True)
    return df


def compute_contact_features(
    df: pd.DataFrame, config: FeatureConfig = FEATURE_CONFIG
) -> pd.DataFrame:
    """
    Compute simple contact-related features:

    - days_since_contact: pdays with -1 mapped to a high value (e.g. 999),
      indicating no previous contact.[web:68]
    - contact_intensity: campaign / max(duration_minutes, 1)
    """
    df = df.copy()

    if "pdays" not in df.columns or "duration" not in df.columns or "campaign" not in df.columns:
        missing = [c for c in ["pdays", "duration", "campaign"] if c not in df.columns]
        raise KeyError(f"Missing required columns for contact features: {missing}")

    # days_since_contact: -1 means no previous contact; send to a sentinel value.
    df["days_since_contact"] = df["pdays"].where(df["pdays"] >= 0, 999)

    # duration is in seconds; convert to minutes and guard against zero.
    duration_minutes = df["duration"].clip(lower=1) / 60.0
    df["contact_intensity"] = df["campaign"] / duration_minutes

    return df


def scale_numerics(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    config: FeatureConfig = FEATURE_CONFIG,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale numeric columns using StandardScaler.

    If `scaler` is None, a new scaler is fitted and returned along with the
    transformed DataFrame. If a pre-fitted scaler is provided, it is used
    to transform the DataFrame without refitting. This is critical so that
    train and serve use the exact same transformation.
    """
    df = df.copy()
    numeric_cols: List[str] = [
        c for c in config.numeric_cols + config.engineered_cols if c in df.columns
    ]
    if not numeric_cols:
        raise ValueError("No numeric columns found to scale.")

    values = df[numeric_cols].to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(values)
    else:
        scaled_values = scaler.transform(values)

    df[numeric_cols] = scaled_values
    return df, scaler


def get_feature_names(config: FeatureConfig = FEATURE_CONFIG) -> List[str]:
    """
    Return the ordered list of feature columns used by the model.

    This includes:
    - Encoded categorical columns
    - Base numeric columns
    - Engineered feature columns
    """
    return list(config.categorical_cols) + list(config.numeric_cols) + list(
        config.engineered_cols
    )
