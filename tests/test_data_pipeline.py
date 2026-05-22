"""
Data Pipeline Tests — validates Pandera schema rules, categorical encoding,
age binning, contact engineering, and standard scaling.
"""

import pytest
import pandas as pd
import numpy as np
import pandera as pa
from sklearn.preprocessing import StandardScaler

from src.data_pipeline.features import (
    encode_categoricals,
    bin_age,
    compute_contact_features,
    scale_numerics,
    build_features,
    get_feature_names,
)
from src.data_pipeline.validate import get_uci_schema


# ── Sample raw row matching the dataset structure ────────────────
def get_sample_raw_df() -> pd.DataFrame:
    return pd.DataFrame([{
        "age": 30,
        "job": "admin.",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "duration": 250,
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": -1.8,
        "cons.price.idx": 92.893,
        "cons.conf.idx": -46.2,
        "euribor3m": 1.299,
        "nr.employed": 5099.1,
        "y": "no",
    }])


# ── Pandera Validation Tests ─────────────────────────────────────
def test_valid_data_validation():
    """Verify that a compliant row passes the Pandera schema validation."""
    df = get_sample_raw_df()
    schema = get_uci_schema()
    # Should run successfully without throwing exception
    schema.validate(df)


def test_invalid_age_validation():
    """Verify that a row with age out-of-bounds throws a Pandera SchemaError."""
    df = get_sample_raw_df()
    df.loc[0, "age"] = 15  # too young (minimum is 17)
    
    schema = get_uci_schema()
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(df)


def test_invalid_campaign_validation():
    """Verify that a campaign value < 1 throws a Pandera SchemaError."""
    df = get_sample_raw_df()
    df.loc[0, "campaign"] = 0  # must be >= 1
    
    schema = get_uci_schema()
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(df)


# ── Feature Engineering Tests ────────────────────────────────────
def test_categorical_encoding():
    """Verify categorical encoders fit and transform unseen data to -1."""
    df = get_sample_raw_df()
    df_encoded, encoders = encode_categoricals(df)
    
    assert "job" in df_encoded.columns
    assert isinstance(df_encoded.loc[0, "job"], (int, np.integer))
    
    # Test serve-time transform of unseen labels maps to -1
    unseen_df = pd.DataFrame([{"job": "spaceman"}])
    df_unseen, _ = encode_categoricals(unseen_df, encoders)
    assert df_unseen.loc[0, "job"] == -1


def test_age_binning():
    """Verify that ages are correctly binned into young/mid/senior ordinals."""
    df = pd.DataFrame({"age": [25, 45, 75]})
    df_binned = bin_age(df)
    
    assert df_binned.loc[0, "age_group"] == 0  # young (<=30)
    assert df_binned.loc[1, "age_group"] == 1  # mid (31-55)
    assert df_binned.loc[2, "age_group"] == 2  # senior (>55)


def test_contact_features():
    """Verify that contact indicators and campaign intensity are computed."""
    df = pd.DataFrame({
        "pdays": [999, 5],
        "campaign": [2, 3],
        "previous": [0, 2]
    })
    df_feat = compute_contact_features(df)
    
    # Never contacted (999) -> -1
    assert df_feat.loc[0, "days_since_contact"] == -1
    # Genuine contacted -> keep as-is
    assert df_feat.loc[1, "days_since_contact"] == 5
    
    # Contact intensity = campaign * (previous + 1)
    assert df_feat.loc[0, "contact_intensity"] == 2 * (0 + 1)
    assert df_feat.loc[1, "contact_intensity"] == 3 * (2 + 1)


def test_features_pipeline():
    """Verify that the full build_features pipeline produces the correct shape and order."""
    df = get_sample_raw_df()
    feature_cols = get_feature_names()
    
    df_processed, encoders, scaler = build_features(df)
    
    # Check that processed df matches feature names exactly in column order
    assert list(df_processed.columns) == feature_cols
    assert df_processed.shape == (1, len(feature_cols))
    assert isinstance(scaler, StandardScaler)
    assert isinstance(encoders, dict)
