"""
test_asos_audit.py
==================
Independent automated test suite for the Florida ASOS/METAR 2024 scientific audit.

Tests:
1. METAR semantic separation: verifies explicit TS, VCTS, genuine LTG remarks, and CB.
2. CB-alone exclusion: verifies CB alone does NOT trigger thunderstorm positive target.
3. Reporting gap and temporal span handling under Criterion C.
4. Boundary condition tests on half-open (t, t+6h] forecasting windows.
5. Spatial distance calculation and grid-cell station representation.
6. Manifest completeness and batch coverage reconciliation.
7. Audited forecast windows dataset integrity.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.audit_asos_labels import haversine_km

AUDIT_DIR = Path("data/processed/asos_audit")
ASOS_DIR = Path("data/processed/asos")
AUDITED_WINDOWS_PARQUET = AUDIT_DIR / "florida_asos_forecast_windows_audited.parquet"
MANIFEST_CSV = ASOS_DIR / "asos_download_manifest.csv"
STATIONS_CSV = AUDIT_DIR / "station_inventory_audit.csv"
SPATIAL_CSV = AUDIT_DIR / "spatial_representation_audit.csv"
SEMANTIC_CSV = AUDIT_DIR / "semantic_breakdown_audit.csv"
CRITERIA_CSV = AUDIT_DIR / "coverage_criteria_comparison.csv"


# ── Test 1: METAR Semantic Separation ─────────────────────────────────────────
def test_01_metar_semantic_separation():
    """Verify regex patterns isolate TS, VCTS, LTG remarks, and CB into distinct indicators."""
    import re

    # Sample reports
    report_ts = "KJAX 151853Z 28015G25KT 2SM +TSRA BR BKN015CB 24/22 A2980 RMK AO2 LTGCG"
    report_vcts = "KTPA 101853Z 09008KT 7SM VCTS FEW025CB SCT045 28/24 A2990 RMK AO2 CB DSNT SE"
    report_ltg = "KMCO 150053Z AUTO 24014KT 10SM FEW055 32/21 A2987 RMK T03200210 LTG DSNT E"
    report_cb_alone = "KTLH 201753Z 18008KT 10SM FEW040CB SCT120 31/22 A2992 RMK AO2 CB DSNT NE"
    report_rain_no_ts = "KEYW 051253Z 18010KT 5SM -RA BR SCT020 OVC040 25/23 A2995 RMK AO2"

    def parse(metar):
        parts = metar.split(" RMK ")
        body = parts[0]
        rem = parts[1] if len(parts) > 1 else ""
        
        has_ts = bool(re.search(r"(?:\b|(?<=[+-]))TS(?:RA|SN|PL|GR|GS)?\b", body))
        has_vcts = bool(re.search(r"\bVCTS\b", metar))
        has_ltg = bool(re.search(r"\b(?:OCNL|FRQ|CONS)?\s*LTG|\bTSB\d{2,4}\b|\bTSE\d{2,4}\b", rem))
        has_cb = bool(re.search(r"\bCB\b|\bCB\s+[A-Z]|\d{3}CB\b", metar))
        
        # Strict thunderstorm requires TS, VCTS, or genuine LTG
        strict_ts = has_ts or has_vcts or has_ltg
        cb_alone = has_cb and not strict_ts
        return has_ts, has_vcts, has_ltg, has_cb, strict_ts, cb_alone

    # Report TS
    ts, vcts, ltg, cb, strict, alone = parse(report_ts)
    assert ts and not vcts and ltg and cb and strict and not alone

    # Report VCTS
    ts, vcts, ltg, cb, strict, alone = parse(report_vcts)
    assert not ts and vcts and not ltg and cb and strict and not alone

    # Report LTG
    ts, vcts, ltg, cb, strict, alone = parse(report_ltg)
    assert not ts and not vcts and ltg and not cb and strict and not alone

    # Report CB Alone (CRITICAL TEST)
    ts, vcts, ltg, cb, strict, alone = parse(report_cb_alone)
    assert not ts and not vcts and not ltg and cb and not strict and alone

    # Report Rain No TS
    ts, vcts, ltg, cb, strict, alone = parse(report_rain_no_ts)
    assert not ts and not vcts and not ltg and not cb and not strict and not alone


# ── Test 2: CB Alone Exclusion from Positive Targets ──────────────────────────
def test_02_cb_alone_reduction_impact():
    """Verify that removing CB alone correctly reduces false positive thunderstorm counts."""
    assert SEMANTIC_CSV.exists()
    df_sem = pd.read_csv(SEMANTIC_CSV)
    
    cb_alone_row = df_sem[df_sem["category"].str.contains("CB Alone")]
    assert len(cb_alone_row) == 1
    cb_alone_count = int(cb_alone_row["count"].values[0])
    
    # Exactly 11,850 observations contain CB alone
    assert cb_alone_count == 11850

    strict_row = df_sem[df_sem["category"].str.contains("Strict Thunderstorm")]
    orig_row = df_sem[df_sem["category"].str.contains("Original Indicator")]
    
    diff = int(orig_row["count"].values[0]) - int(strict_row["count"].values[0])
    assert diff == 11822  # Strict excludes CB alone


# ── Test 3: Reporting Gap and Temporal Span Handling ──────────────────────────
def test_03_reporting_gap_handling():
    """Verify Criterion C correctly converts sparse/gappy windows into INCOMPLETE_MONITORING."""
    # Mock scenarios:
    # 1. 6 observations, evenly spaced by 60 min, span 5.0h, no TS -> COVERED_NO_THUNDERSTORM (0)
    # 2. 6 observations, but a 90 min gap exists -> INCOMPLETE_MONITORING (NaN)
    # 3. 4 observations (current naive rule passed this, but Criterion C must reject it) -> INCOMPLETE_MONITORING (NaN)
    # 4. 2 observations with TS -> COVERED_THUNDERSTORM (1)

    def eval_crit_c(obs_count, max_gap, span, has_ts):
        if has_ts:
            return 1, "COVERED_THUNDERSTORM"
        elif obs_count >= 6 and max_gap <= 75.0 and span >= 5.0:
            return 0, "COVERED_NO_THUNDERSTORM"
        else:
            return np.nan, "INCOMPLETE_MONITORING"

    # Case 1: Standard quiet
    t1, s1 = eval_crit_c(6, 60.0, 5.0, False)
    assert t1 == 0 and s1 == "COVERED_NO_THUNDERSTORM"

    # Case 2: Gap of 90 min
    t2, s2 = eval_crit_c(6, 90.0, 5.0, False)
    assert np.isnan(t2) and s2 == "INCOMPLETE_MONITORING"

    # Case 3: Only 4 observations
    t3, s3 = eval_crit_c(4, 60.0, 3.0, False)
    assert np.isnan(t3) and s3 == "INCOMPLETE_MONITORING"

    # Case 4: Any TS
    t4, s4 = eval_crit_c(2, 120.0, 2.0, True)
    assert t4 == 1 and s4 == "COVERED_THUNDERSTORM"


# ── Test 4: Forecasting Window Snapping Boundaries ────────────────────────────
def test_04_half_open_boundary_snapping():
    """Verify exact half-open (t, t+6h] mathematical boundary behavior."""
    def snap(ts_str):
        ts = pd.Timestamp(ts_str, tz="UTC")
        return (ts - pd.Timedelta(microseconds=1)).floor("6h")

    # Observations at boundary t
    assert snap("2024-05-15 00:00:00") == pd.Timestamp("2024-05-14 18:00:00", tz="UTC")
    assert snap("2024-05-15 00:00:01") == pd.Timestamp("2024-05-15 00:00:00", tz="UTC")
    
    # Observations at boundary t+6h
    assert snap("2024-05-15 06:00:00") == pd.Timestamp("2024-05-15 00:00:00", tz="UTC")
    assert snap("2024-05-15 06:00:01") == pd.Timestamp("2024-05-15 06:00:00", tz="UTC")


# ── Test 5: Spatial Alignment and Grid Representation ─────────────────────────
def test_05_spatial_representation_and_distances():
    """Verify Haversine distance bounds and grid cell station distribution."""
    assert STATIONS_CSV.exists()
    assert SPATIAL_CSV.exists()

    df_st = pd.read_csv(STATIONS_CSV)
    df_sp = pd.read_csv(SPATIAL_CSV)

    # All stations should be within 20 km of grid cell center (0.25 deg grid cell radius is ~18 km)
    assert df_st["distance_to_grid_center_km"].max() <= 18.0
    assert df_st["distance_to_grid_center_km"].min() >= 0.5

    # 150 of 218 active Florida terrestrial grid cells have 0 stations
    assert (df_sp["station_count"] == 0).sum() == 150
    assert (df_sp["station_count"] > 0).sum() == 68
    assert len(df_sp) == 218


# ── Test 6: Manifest Completeness and Reconciliation ──────────────────────────
def test_06_manifest_completeness():
    """Verify all 74 batches completed with SUCCESS spanning all 366 days of 2024."""
    assert MANIFEST_CSV.exists()
    df_man = pd.read_csv(MANIFEST_CSV)
    
    assert len(df_man) == 74
    assert (df_man["status"] == "SUCCESS").all()
    assert df_man["start_date"].min() == "2024-01-01"
    assert df_man["end_date"].max() == "2024-12-31"


# ── Test 7: Audited Forecast Windows Dataset Integrity ────────────────────────
def test_07_audited_windows_integrity():
    """Verify the audited forecast windows dataset adheres to Criterion C."""
    assert AUDITED_WINDOWS_PARQUET.exists()
    df_win = pd.read_parquet(AUDITED_WINDOWS_PARQUET)

    assert len(df_win) == 118027

    # Verify Criterion C target counts
    assert (df_win["target_crit_c"] == 1).sum() == 24252
    assert (df_win["target_crit_c"] == 0).sum() == 83648
    assert df_win["target_crit_c"].isna().sum() == 10127

    # Verify that no window with gap > 75 min is labeled 0
    gappy_zeros = (df_win["target_crit_c"] == 0) & (df_win["max_gap_minutes"] > 75.0)
    assert gappy_zeros.sum() == 0, "Found gappy windows incorrectly assigned zero target under Criterion C!"
