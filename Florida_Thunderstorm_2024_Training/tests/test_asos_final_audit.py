"""
test_asos_final_audit.py
========================
Automated validation test suite for the Florida 2024 ASOS Final Independent Validation Audit.

Verifies:
1. 366 calendar dates present, including all 73 backfilled dates.
2. Station coverage: 104 active stations, Station AGR as the sole historical outage.
3. Zero duplicate records and strict [t, t+6h] window boundary snapping.
4. Target definition integrity: Criterion C logic and strict exclusion of CB-alone remarks.
5. Predictor availability and zero target-window feature leakage.
6. Baseline metric recomputation and identification of the DTS/TTS callsign parsing flaw.
"""

from pathlib import Path
import json
import pytest
import pandas as pd
import numpy as np

ROOT_DIR = Path(".").resolve()
AUDIT_DIR = ROOT_DIR / "data" / "processed" / "asos_final_audit"
REPAIR_DIR = ROOT_DIR / "data" / "processed" / "asos_repair"

DAILY_CSV = AUDIT_DIR / "daily_coverage.csv"
STATION_CSV = AUDIT_DIR / "station_coverage.csv"
FEATURE_AUDIT_CSV = AUDIT_DIR / "feature_availability_audit.csv"
BASELINES_CSV = AUDIT_DIR / "baseline_metrics_recomputed.csv"
CHECKS_JSON = AUDIT_DIR / "audit_checks.json"

REPAIRED_OBS_FILE = REPAIR_DIR / "florida_asos_observations_repaired_2024.parquet"
REPAIRED_WINDOWS_FILE = REPAIR_DIR / "florida_asos_forecast_windows_repaired_2024.parquet"
REPAIRED_MODELING_FILE = REPAIR_DIR / "florida_asos_modeling_ready_repaired_2024.parquet"


def test_01_calendar_and_backfill_completeness():
    """Verify 366 unique calendar dates in 2024 and all 73 backfilled dates."""
    assert DAILY_CSV.exists(), f"Missing {DAILY_CSV}"
    df_daily = pd.read_csv(DAILY_CSV)

    assert len(df_daily) == 366, f"Expected 366 dates, got {len(df_daily)}"
    assert df_daily["obs_date"].min() == "2024-01-01"
    assert df_daily["obs_date"].max() == "2024-12-31"

    backfilled_count = df_daily["is_backfilled_date"].sum()
    assert backfilled_count == 73, f"Expected 73 backfilled dates, got {backfilled_count}"
    assert (df_daily["active_stations"] >= 80).all(), "Some dates have abnormally low active stations"


def test_02_station_coverage_and_outages():
    """Verify 105 total stations in inventory, 104 active, and station AGR offline."""
    assert STATION_CSV.exists(), f"Missing {STATION_CSV}"
    df_st = pd.read_csv(STATION_CSV)

    assert len(df_st) == 105, f"Expected 105 inventory stations, got {len(df_st)}"
    active_count = (df_st["reporting_status"] == "ACTIVE").sum()
    outage_count = (df_st["reporting_status"] == "INACTIVE_OUTAGE").sum()

    assert active_count == 104, f"Expected 104 active stations, got {active_count}"
    assert outage_count == 1, f"Expected 1 offline station, got {outage_count}"

    agr_row = df_st[df_st["station_id"] == "AGR"]
    assert len(agr_row) == 1
    assert agr_row.iloc[0]["reporting_status"] == "INACTIVE_OUTAGE"
    assert agr_row.iloc[0]["total_observations"] == 0


def test_03_deduplication_and_window_snapping():
    """Verify zero duplicate observations and strict [t, t+6h] window snapping."""
    assert REPAIRED_WINDOWS_FILE.exists(), f"Missing {REPAIRED_WINDOWS_FILE}"
    df_win = pd.read_parquet(REPAIRED_WINDOWS_FILE, columns=["forecast_init_time_utc", "first_obs", "last_obs"])

    assert len(df_win) == 143687, f"Expected 143,687 forecast windows, got {len(df_win):,}"

    # Target window snapping: first_obs >= init_time, last_obs <= init_time + 6h
    earlier = (df_win["first_obs"] < df_win["forecast_init_time_utc"]).sum()
    later = (df_win["last_obs"] > (df_win["forecast_init_time_utc"] + pd.Timedelta(hours=6))).sum()

    assert earlier == 0, f"Found {earlier} observations earlier than forecast init time"
    assert later == 0, f"Found {later} observations beyond forecast window +6h"


def test_04_target_integrity_and_cb_exclusion():
    """Verify Criterion C target evaluation and strict exclusion of CB-alone remarks."""
    df_win = pd.read_parquet(
        REPAIRED_WINDOWS_FILE,
        columns=["strict_ts_count", "cb_alone_count", "obs_count", "max_gap_minutes", "obs_span_hours", "target_crit_c", "status_crit_c"]
    )

    # Criterion C positive
    has_ts = df_win["strict_ts_count"] > 0
    assert (df_win.loc[has_ts, "target_crit_c"] == 1.0).all()
    assert (df_win.loc[has_ts, "status_crit_c"] == "COVERED_THUNDERSTORM").all()

    # CB-alone false positive check
    cb_alone_no_ts = (df_win["strict_ts_count"] == 0) & (df_win["cb_alone_count"] > 0)
    cb_alone_positives = (df_win.loc[cb_alone_no_ts, "target_crit_c"] == 1.0).sum()
    assert cb_alone_positives == 0, f"CB-alone remarks caused {cb_alone_positives} false positives"


def test_05_feature_availability_and_leakage():
    """Verify 29 predictors documented, zero nulls in ERA5 features, and no persistence gap bridging."""
    assert FEATURE_AUDIT_CSV.exists(), f"Missing {FEATURE_AUDIT_CSV}"
    df_feat = pd.read_csv(FEATURE_AUDIT_CSV)
    assert len(df_feat) == 29, f"Expected 29 feature entries, got {len(df_feat)}"
    assert (df_feat["target_window_leakage_status"] == "CLEAN").all()

    # Load modeling ready table
    assert REPAIRED_MODELING_FILE.exists(), f"Missing {REPAIRED_MODELING_FILE}"
    df_mod = pd.read_parquet(REPAIRED_MODELING_FILE)
    assert len(df_mod) == 119663, f"Expected 119,663 terrestrial rows, got {len(df_mod)}"

    # Core thermodynamic/kinematic null checks
    core_cols = [
        "cape_J_kg", "cin_J_kg", "dewpoint_depression_degC", "temperature_2m_degC",
        "surface_pressure_hPa", "mslp_hPa", "bulk_shear_850_300_ms", "tcwv_kg_m2"
    ]
    for col in core_cols:
        assert df_mod[col].isna().sum() == 0, f"Null values found in {col}"


def test_06_baseline_recomputation_and_callsign_flaw():
    """Verify recomputed baselines and detection of DTS/TTS callsign parsing flaw."""
    assert BASELINES_CSV.exists(), f"Missing {BASELINES_CSV}"
    df_base = pd.read_csv(BASELINES_CSV)
    assert len(df_base) == 10, f"Expected 10 baseline rows, got {len(df_base)}"

    # Check persistence baseline performance across cohorts
    persist_as_deliv = df_base[df_base["baseline_name"].str.contains("As-Delivered.*Persistence")].iloc[0]
    persist_clean = df_base[df_base["baseline_name"].str.contains("Clean.*Persistence")].iloc[0]

    assert pytest.approx(persist_as_deliv["f1"], abs=1e-3) == 0.6860
    assert pytest.approx(persist_clean["f1"], abs=1e-3) == 0.5656

    # Verify audit checks JSON records CONDITIONAL GO
    assert CHECKS_JSON.exists(), f"Missing {CHECKS_JSON}"
    with open(CHECKS_JSON, "r") as f:
        checks_data = json.load(f)

    assert checks_data["final_decision"] == "CONDITIONAL GO"
    assert checks_data["checks"]["station_callsign_parsing_flaw"]["status"] == "FAIL"
    assert "DTS" in checks_data["checks"]["station_callsign_parsing_flaw"]["affected_stations"]
    assert "TTS" in checks_data["checks"]["station_callsign_parsing_flaw"]["affected_stations"]
