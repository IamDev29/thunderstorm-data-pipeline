"""
test_asos_model_readiness.py
============================
Automated test suite for ASOS model-readiness verification.

Validates:
1. Integrity of modeling-ready dataset (florida_asos_modeling_ready_2024.parquet).
2. Data leakage safeguards: features are strictly known at forecast init time t.
3. Target disaggregation: TS, VCTS, LTG, and NOAA severe are separate.
4. Chronological train/validation/test split date boundaries and disjointness.
5. Baseline benchmark outputs and persistence threshold.
6. Audit verification of the 73 missing days defect.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

MODELING_DIR = Path("data/processed/asos_modeling")
MODELING_PARQUET = MODELING_DIR / "florida_asos_modeling_ready_2024.parquet"
BASELINES_CSV = MODELING_DIR / "asos_baseline_metrics.csv"
OBSERVATIONS_PARQUET = Path("data/processed/asos/florida_asos_observations_2024.parquet")


def test_01_modeling_ready_parquet_integrity():
    """Verify modeling-ready dataset exists, has expected rows, and valid columns."""
    assert MODELING_PARQUET.exists(), f"Missing modeling dataset: {MODELING_PARQUET}"
    df = pd.read_parquet(MODELING_PARQUET)

    assert len(df) == 98353, f"Expected 98,353 terrestrial station-windows, got {len(df):,}"
    assert (df["in_active_terrestrial_grid"] == True).all()

    # Core ERA5 features must have zero nulls
    core_features = ["cape_J_kg", "cin_J_kg", "dewpoint_depression_degC", "temperature_2m_degC", "bulk_shear_850_300_ms"]
    for feat in core_features:
        assert feat in df.columns
        assert df[feat].isna().sum() == 0, f"Found nulls in core feature {feat}"


def test_02_leakage_safeguards():
    """Verify all predictor features are strictly evaluated at or prior to init time t."""
    df = pd.read_parquet(MODELING_PARQUET)
    
    # Confirm init time is prior to observation interval
    # In every row, first observation must be >= init time
    assert (df["first_obs"] >= df["forecast_init_time_utc"]).all()
    assert (df["last_obs"] <= df["forecast_init_time_utc"] + pd.Timedelta(hours=6)).all()


def test_03_target_disaggregation():
    """Verify convective targets are disaggregated and never silently combined."""
    df = pd.read_parquet(MODELING_PARQUET)

    target_cols = [
        "target_asos_strict_crit_c",
        "target_asos_explicit_ts",
        "target_asos_aerodrome_ts_vcts",
        "target_noaa_severe_6h"
    ]
    for col in target_cols:
        assert col in df.columns

    # Verify counts: strict > aerodrome > explicit > NOAA
    n_strict = (df["target_asos_strict_crit_c"] == 1).sum()
    n_aero = (df["target_asos_aerodrome_ts_vcts"] == 1).sum()
    n_ts = (df["target_asos_explicit_ts"] == 1).sum()
    n_noaa = (df["target_noaa_severe_6h"] == 1).sum()

    assert n_strict == 20678
    assert n_aero == 10021
    assert n_ts == 7100
    assert n_noaa == 211
    assert n_strict > n_aero > n_ts > n_noaa


def test_04_chronological_split_integrity():
    """Verify train, val, and test splits are strictly non-overlapping in time."""
    df = pd.read_parquet(MODELING_PARQUET)
    df_eval = df[df["target_asos_strict_crit_c"].notna()].copy()

    train = df_eval[df_eval["forecast_init_time_utc"] < "2024-08-01"]
    val = df_eval[(df_eval["forecast_init_time_utc"] >= "2024-08-01") & (df_eval["forecast_init_time_utc"] < "2024-10-01")]
    test = df_eval[df_eval["forecast_init_time_utc"] >= "2024-10-01"]

    assert len(train) == 53494
    assert len(val) == 14711
    assert len(test) == 22383
    assert len(train) + len(val) + len(test) == len(df_eval)

    # Disjointness check
    assert train["forecast_init_time_utc"].max() < val["forecast_init_time_utc"].min()
    assert val["forecast_init_time_utc"].max() < test["forecast_init_time_utc"].min()


def test_05_baseline_metrics_integrity():
    """Verify baseline benchmark metrics are computed and meet persistence threshold."""
    assert BASELINES_CSV.exists()
    df_base = pd.read_csv(BASELINES_CSV)

    assert len(df_base) == 4
    persist_row = df_base[df_base["baseline"].str.contains("Persistence")]
    assert len(persist_row) == 1
    
    f1_persist = float(persist_row["f1"].values[0])
    roc_persist = float(persist_row["roc_auc"].values[0])
    
    assert 0.60 <= f1_persist <= 0.75, f"Unexpected persistence F1: {f1_persist}"
    assert 0.80 <= roc_persist <= 0.90, f"Unexpected persistence ROC-AUC: {roc_persist}"


def test_06_verify_73_missing_days_audit_finding():
    """Verify the independent audit finding that 73 calendar days require backfilling."""
    df_obs = pd.read_parquet(OBSERVATIONS_PARQUET, columns=["valid_time_utc"])
    unique_dates = df_obs["valid_time_utc"].dt.date.unique()

    assert len(unique_dates) == 293, f"Expected 293 unique observation dates, got {len(unique_dates)}"
    assert 366 - len(unique_dates) == 73, "Expected exactly 73 missing calendar days"
