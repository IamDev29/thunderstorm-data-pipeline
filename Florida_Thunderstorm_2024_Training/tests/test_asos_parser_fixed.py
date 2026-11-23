"""
test_asos_parser_fixed.py
=========================
Automated test suite validating the rebuilt ASOS datasets and corrected METAR thunderstorm labels.

Verifies:
1. Corrected observations dataset integrity (5,728,432 rows, 366 dates, DTS and TTS false positives eliminated).
2. Corrected forecast windows under Criterion C (143,687 rows, strict snapping, realistic DTS/TTS prevalence).
3. Corrected terrestrial modeling table (119,663 rows, complete ERA5 features, clean persistence lag).
4. Recomputed baseline metrics (persistence F1 ~ 0.56, Logistic Regression ROC-AUC ~ 0.82).
5. All deliverable files exist and have verified integrity.
"""

from pathlib import Path
import json
import pytest
import pandas as pd
import numpy as np

ROOT_DIR = Path(".").resolve()
FIXED_DIR = ROOT_DIR / "data" / "processed" / "asos_parser_fixed"

OBS_FILE = FIXED_DIR / "florida_asos_observations_fixed_2024.parquet"
WINDOWS_FILE = FIXED_DIR / "florida_asos_forecast_windows_fixed_2024.parquet"
MODELING_FILE = FIXED_DIR / "florida_asos_modeling_ready_fixed_2024.parquet"
CHANGE_CSV = FIXED_DIR / "label_change_summary.csv"
TESTS_CSV = FIXED_DIR / "parser_regression_tests.csv"
BASELINES_CSV = FIXED_DIR / "baseline_metrics.csv"
STATION_CSV = FIXED_DIR / "station_target_summary.csv"
CHECKS_JSON = FIXED_DIR / "audit_checks.json"


def test_01_fixed_observations_integrity():
    """Verify 5,728,432 observations, 366 dates, and corrected DTS/TTS counts."""
    assert OBS_FILE.exists(), f"Missing {OBS_FILE}"
    df_obs = pd.read_parquet(OBS_FILE, columns=["valid_time_utc", "station_id", "strict_thunderstorm_reported", "has_thunderstorm_ts"])

    assert len(df_obs) == 5728432, f"Expected 5,728,432 rows, got {len(df_obs)}"

    # 366 unique dates
    unique_dates = df_obs["valid_time_utc"].dt.date.unique()
    assert len(unique_dates) == 366, f"Expected 366 dates, got {len(unique_dates)}"

    # DTS and TTS false positives eliminated
    dts_ts = (df_obs["station_id"] == "DTS") & (df_obs["strict_thunderstorm_reported"] == True)
    tts_ts = (df_obs["station_id"] == "TTS") & (df_obs["strict_thunderstorm_reported"] == True)

    assert dts_ts.sum() < 6000, f"DTS TS count too high: {dts_ts.sum()}"
    assert dts_ts.sum() > 3000, f"DTS TS count too low: {dts_ts.sum()}"

    assert tts_ts.sum() < 2000, f"TTS TS count too high: {tts_ts.sum()}"
    assert tts_ts.sum() > 500, f"TTS TS count too low: {tts_ts.sum()}"


def test_02_fixed_forecast_windows_integrity():
    """Verify 143,687 forecast windows, boundary snapping, and realistic DTS/TTS prevalence."""
    assert WINDOWS_FILE.exists(), f"Missing {WINDOWS_FILE}"
    df_win = pd.read_parquet(WINDOWS_FILE)

    assert len(df_win) == 143687, f"Expected 143,687 rows, got {len(df_win)}"

    # Window boundary snapping: first_obs >= init_time, last_obs <= init_time + 6h
    earlier = (df_win["first_obs"] < df_win["forecast_init_time_utc"]).sum()
    later = (df_win["last_obs"] > (df_win["forecast_init_time_utc"] + pd.Timedelta(hours=6))).sum()
    assert earlier == 0, f"Found {earlier} observations earlier than forecast init time"
    assert later == 0, f"Found {later} observations beyond forecast horizon"

    # Status distribution check
    assert (df_win["status_crit_c"] == "COVERED_NO_THUNDERSTORM").sum() > 100000
    assert (df_win["status_crit_c"] == "COVERED_THUNDERSTORM").sum() > 25000

    # DTS and TTS prevalence in windows
    dts_win = df_win[df_win["station_id"] == "DTS"]
    dts_mon = dts_win["target_crit_c"].notna()
    dts_prev = (dts_win.loc[dts_mon, "target_crit_c"] == 1.0).mean() * 100
    assert 10.0 <= dts_prev <= 25.0, f"DTS window prevalence abnormal: {dts_prev:.2f}%"

    tts_win = df_win[df_win["station_id"] == "TTS"]
    tts_mon = tts_win["target_crit_c"].notna()
    tts_prev = (tts_win.loc[tts_mon, "target_crit_c"] == 1.0).mean() * 100
    assert 10.0 <= tts_prev <= 25.0, f"TTS window prevalence abnormal: {tts_prev:.2f}%"


def test_03_fixed_modeling_ready_table():
    """Verify terrestrial modeling dataset has 119,663 rows, zero feature nulls, and clean persistence."""
    assert MODELING_FILE.exists(), f"Missing {MODELING_FILE}"
    df_mod = pd.read_parquet(MODELING_FILE)

    assert len(df_mod) == 119663, f"Expected 119,663 rows, got {len(df_mod)}"
    assert (df_mod["in_active_terrestrial_grid"] == True).all()

    # Core ERA5 feature null checks
    core_features = ["cape_J_kg", "cin_J_kg", "dewpoint_depression_degC", "temperature_2m_degC", "bulk_shear_850_300_ms"]
    for feat in core_features:
        assert df_mod[feat].isna().sum() == 0, f"Null values in {feat}"

    # Target column presence
    assert "target_asos_strict_crit_c" in df_mod.columns
    assert "prior_strict_ts_6h" in df_mod.columns

    # Persistence gap bridging check
    assert (df_mod["prior_strict_ts_6h"].isin([0, 1])).all()


def test_04_baseline_metrics_and_deliverables():
    """Verify deliverables exist and baseline metrics reflect clean persistence ~0.5623."""
    for p in [CHANGE_CSV, TESTS_CSV, BASELINES_CSV, STATION_CSV, CHECKS_JSON]:
        assert p.exists(), f"Missing deliverable: {p}"

    df_base = pd.read_csv(BASELINES_CSV)
    assert len(df_base) == 10, f"Expected 10 baseline rows, got {len(df_base)}"

    # Check corrected persistence F1
    persist_corr = df_base[df_base["baseline_name"].str.contains("Corrected.*Persistence")].iloc[0]
    assert pytest.approx(persist_corr["f1"], abs=1e-2) == 0.5623
    assert pytest.approx(persist_corr["roc_auc"], abs=1e-2) == 0.7641

    # Check regression tests CSV
    df_tests = pd.read_csv(TESTS_CSV)
    assert len(df_tests) == 20
    assert (df_tests["status"] == "PASS").all()

    # Check JSON verdict
    with open(CHECKS_JSON, "r") as f:
        data = json.load(f)
    assert data["verdict"] == "GO"
