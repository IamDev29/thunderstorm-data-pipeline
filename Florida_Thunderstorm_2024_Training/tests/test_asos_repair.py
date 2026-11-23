"""
test_asos_repair.py
===================
Automated test suite verifying the 73-day ASOS backfill and repaired downstream datasets.

Validates:
1. Manifest completeness: all 73 missing dates retrieved with status SUCCESS.
2. Repaired observations dataset: 5,728,432 rows across exactly 366 calendar dates in 2024.
3. Repaired forecast windows under Criterion C: 143,687 rows.
4. Repaired modeling dataset: 119,663 terrestrial rows with complete ERA5 features.
5. Repaired baseline benchmark metrics: Persistence F1 >= 0.68.
6. Station-level outage consistency: Station AGR remains the sole offline facility.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

REPAIR_DIR = Path("data/processed/asos_repair")
MANIFEST_CSV = REPAIR_DIR / "asos_missing_dates_manifest.csv"
REPAIRED_OBS_PARQUET = REPAIR_DIR / "florida_asos_observations_repaired_2024.parquet"
REPAIRED_WINDOWS_PARQUET = REPAIR_DIR / "florida_asos_forecast_windows_repaired_2024.parquet"
REPAIRED_MODELING_PARQUET = REPAIR_DIR / "florida_asos_modeling_ready_repaired_2024.parquet"
REPAIRED_BASELINES_CSV = REPAIR_DIR / "asos_repaired_baseline_metrics.csv"


def test_01_manifest_completeness():
    """Verify all 73 missing dates are recorded with SUCCESS."""
    assert MANIFEST_CSV.exists(), f"Missing manifest: {MANIFEST_CSV}"
    df_man = pd.read_csv(MANIFEST_CSV)

    assert len(df_man) == 73, f"Expected 73 backfilled dates, got {len(df_man)}"
    assert (df_man["status"] == "SUCCESS").all(), "Not all backfill dates succeeded"
    assert df_man["date"].min() == "2024-01-05"
    assert df_man["date"].max() == "2024-12-30"


def test_02_repaired_observations_completeness():
    """Verify repaired observations dataset covers all 366 days without duplicates."""
    assert REPAIRED_OBS_PARQUET.exists(), f"Missing repaired observations: {REPAIRED_OBS_PARQUET}"
    df_obs = pd.read_parquet(REPAIRED_OBS_PARQUET, columns=["valid_time_utc", "station_id"])

    assert len(df_obs) == 5728432, f"Expected 5,728,432 observations, got {len(df_obs):,}"
    
    # Verify 366 unique calendar dates
    unique_dates = df_obs["valid_time_utc"].dt.date.unique()
    assert len(unique_dates) == 366, f"Expected 366 dates, got {len(unique_dates)}"

    # Check station count (104 active stations)
    assert df_obs["station_id"].nunique() == 104, f"Expected 104 stations, got {df_obs['station_id'].nunique()}"

    # Deduplication check
    dups = df_obs.duplicated(subset=["station_id", "valid_time_utc"]).sum()
    assert dups == 0, f"Found {dups} duplicate station timestamps!"


def test_03_repaired_forecast_windows_integrity():
    """Verify repaired forecast window dataset adheres to Criterion C."""
    assert REPAIRED_WINDOWS_PARQUET.exists(), f"Missing repaired windows: {REPAIRED_WINDOWS_PARQUET}"
    df_win = pd.read_parquet(REPAIRED_WINDOWS_PARQUET)

    assert len(df_win) == 143687, f"Expected 143,687 forecast windows, got {len(df_win):,}"
    
    # Check status values
    valid_statuses = {"COVERED_THUNDERSTORM", "COVERED_NO_THUNDERSTORM", "INCOMPLETE_MONITORING"}
    assert set(df_win["status_crit_c"].unique()).issubset(valid_statuses)

    # Verify Criterion C logic
    has_ts = df_win["strict_ts_count"] > 0
    assert (df_win.loc[has_ts, "target_crit_c"] == 1.0).all()
    assert (df_win.loc[has_ts, "status_crit_c"] == "COVERED_THUNDERSTORM").all()

    adequate_no_ts = (~has_ts) & (df_win["obs_count"] >= 6) & (df_win["max_gap_minutes"] <= 75.0) & (df_win["obs_span_hours"] >= 5.0)
    assert (df_win.loc[adequate_no_ts, "target_crit_c"] == 0.0).all()
    assert (df_win.loc[adequate_no_ts, "status_crit_c"] == "COVERED_NO_THUNDERSTORM").all()


def test_04_repaired_modeling_dataset_integrity():
    """Verify repaired modeling dataset rows increased and have complete ERA5 features."""
    assert REPAIRED_MODELING_PARQUET.exists(), f"Missing modeling dataset: {REPAIRED_MODELING_PARQUET}"
    df_mod = pd.read_parquet(REPAIRED_MODELING_PARQUET)

    assert len(df_mod) == 119663, f"Expected 119,663 terrestrial rows, got {len(df_mod):,}"
    assert (df_mod["in_active_terrestrial_grid"] == True).all()

    # Core ERA5 features must have zero nulls
    core_features = ["cape_J_kg", "cin_J_kg", "dewpoint_depression_degC", "temperature_2m_degC", "bulk_shear_850_300_ms"]
    for feat in core_features:
        assert feat in df_mod.columns
        assert df_mod[feat].isna().sum() == 0, f"Found nulls in core feature {feat}"


def test_05_repaired_baselines_integrity():
    """Verify updated baseline benchmark metrics reflect repaired unbroken persistence."""
    assert REPAIRED_BASELINES_CSV.exists(), f"Missing baselines CSV: {REPAIRED_BASELINES_CSV}"
    df_base = pd.read_csv(REPAIRED_BASELINES_CSV)

    assert len(df_base) == 4
    persist_row = df_base[df_base["baseline"].str.contains("Persistence")]
    assert len(persist_row) == 1

    f1_persist = float(persist_row["f1"].values[0])
    roc_persist = float(persist_row["roc_auc"].values[0])

    assert f1_persist >= 0.68, f"Expected persistence F1 >= 0.68 after repair, got {f1_persist}"
    assert roc_persist >= 0.82, f"Expected persistence ROC-AUC >= 0.82 after repair, got {roc_persist}"


def test_06_unresolved_station_date_gaps():
    """Verify station AGR is the only unobserved station in 2024."""
    df_obs = pd.read_parquet(REPAIRED_OBS_PARQUET, columns=["station_id"])
    active_stations = set(df_obs["station_id"].unique())

    df_st = pd.read_csv("data/processed/asos/florida_asos_stations_2024.csv")
    catalog_stations = set(df_st["station_id"].unique())

    missing_stations = catalog_stations - active_stations
    assert missing_stations == {"AGR"}, f"Expected only AGR missing, got {missing_stations}"
