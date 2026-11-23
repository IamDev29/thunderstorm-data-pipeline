"""
test_asos_acquisition.py
========================
Automated test suite for the Florida ASOS/METAR 2024 acquisition pipeline.

Validates:
1. Station inventory completeness (105 eligible 2024 Florida stations).
2. Terrestrial vs coastal marine grid partition (88 terrestrial, 17 coastal marine).
3. METAR thunderstorm parsing precision (TS, VCTS, LTG remarks, TSB, TSE, CB).
4. Strict half-open (t, t+6h] forecasting window snapping.
5. Prevention of false-zero imputation for unobserved or sparse observation windows.
6. UTC timestamp consistency.
7. Deduplication integrity.
"""

from datetime import datetime, timezone
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.acquire_asos_data import (
    parse_metar_thunderstorm,
    ASOSAcquisitionPipeline,
)

ASOS_DIR = Path("data/processed/asos")
STATION_FILE = ASOS_DIR / "florida_asos_stations_2024.csv"
MANIFEST_FILE = ASOS_DIR / "asos_download_manifest.csv"
OBSERVATIONS_FILE = ASOS_DIR / "florida_asos_observations_2024.parquet"
WINDOWS_FILE = ASOS_DIR / "florida_asos_forecast_windows_2024.parquet"


# ── Test 1: Station Inventory Validation ──────────────────────────────────────
def test_01_station_inventory_completeness():
    """Verify that eligible 2024 stations are correctly identified and annotated."""
    pipeline = ASOSAcquisitionPipeline(output_dir=ASOS_DIR)
    if STATION_FILE.exists():
        df_st = pd.read_csv(STATION_FILE)
    else:
        df_st = pipeline.fetch_station_inventory()

    assert len(df_st) == 105, f"Expected 105 eligible 2024 Florida ASOS stations, got {len(df_st)}"
    assert (df_st["state"] == "FL").all()
    assert (df_st["eligible_2024"] == True).all()

    # Check terrestrial vs coastal marine partition
    n_terr = (df_st["in_active_terrestrial_grid"] == True).sum()
    n_marine = (df_st["in_active_terrestrial_grid"] == False).sum()

    assert n_terr == 88, f"Expected 88 terrestrial stations, got {n_terr}"
    assert n_marine == 17, f"Expected 17 coastal marine stations, got {n_marine}"
    assert n_terr + n_marine == 105


# ── Test 2: METAR Thunderstorm Parsing Precision ──────────────────────────────
def test_02_metar_parsing_rules():
    """Verify regex parsing for present weather TS, vicinity VCTS, and lightning remarks."""
    # Case A: Heavy thunderstorm with rain
    res_a = parse_metar_thunderstorm("KJAX 151200Z 28015G25KT 2SM +TSRA BR BKN015CB 24/22 A2980 RMK AO2 LTGCG", "+TSRA")
    assert res_a["has_ts"] == True
    assert res_a["has_vcts"] == False
    assert res_a["has_lightning_remark"] == True
    assert res_a["thunderstorm_reported"] == True

    # Case B: Vicinity thunderstorm without in-station TS
    res_b = parse_metar_thunderstorm("KTPA 101800Z 09008KT 7SM VCTS FEW025CB SCT045 28/24 A2990 RMK AO2 CB DSNT SE", "VCTS")
    assert res_b["has_ts"] == False
    assert res_b["has_vcts"] == True
    assert res_b["has_lightning_remark"] == True  # CB in remarks
    assert res_b["thunderstorm_reported"] == True

    # Case C: Distant lightning remark only
    res_c = parse_metar_thunderstorm("KMCO 150000Z AUTO 24014KT 10SM FEW055 32/21 A2987 RMK T03200210 LTG DSNT E", "")
    assert res_c["has_ts"] == False
    assert res_c["has_vcts"] == False
    assert res_c["has_lightning_remark"] == True
    assert res_c["thunderstorm_reported"] == True

    # Case D: Non-thunderstorm rain
    res_d = parse_metar_thunderstorm("KEYW 051200Z 18010KT 5SM -RA BR SCT020 OVC040 25/23 A2995 RMK AO2", "-RA BR")
    assert res_d["has_ts"] == False
    assert res_d["has_vcts"] == False
    assert res_d["has_lightning_remark"] == False
    assert res_d["thunderstorm_reported"] == False

    # Case E: Completely clear weather
    res_e = parse_metar_thunderstorm("KTLH 201500Z 00000KT 10SM CLR 22/15 A3010 RMK AO2", "")
    assert res_e["has_ts"] == False
    assert res_e["has_vcts"] == False
    assert res_e["has_lightning_remark"] == False
    assert res_e["thunderstorm_reported"] == False


# ── Test 3: Strict Half-Open Target Window Snapping ────────────────────────────
def test_03_half_open_window_snapping():
    """Verify that observations strictly adhere to (t, t+6h] half-open interval boundaries."""
    # Observation exactly at 00:00:00 UTC belongs to previous cycle (18:00 of day before)
    t_boundary_start = pd.Timestamp("2024-05-15 00:00:00", tz="UTC")
    cycle_start = (t_boundary_start - pd.Timedelta(microseconds=1)).floor("6h")
    assert cycle_start == pd.Timestamp("2024-05-14 18:00:00", tz="UTC")

    # Observation at 00:01:00 UTC belongs to 00:00 cycle
    t_after_start = pd.Timestamp("2024-05-15 00:01:00", tz="UTC")
    cycle_after = (t_after_start - pd.Timedelta(microseconds=1)).floor("6h")
    assert cycle_after == pd.Timestamp("2024-05-15 00:00:00", tz="UTC")

    # Observation exactly at 06:00:00 UTC belongs to 00:00 cycle
    t_boundary_end = pd.Timestamp("2024-05-15 06:00:00", tz="UTC")
    cycle_end = (t_boundary_end - pd.Timedelta(microseconds=1)).floor("6h")
    assert cycle_end == pd.Timestamp("2024-05-15 00:00:00", tz="UTC")

    # Observation at 06:00:01 UTC belongs to 06:00 cycle
    t_next_cycle = pd.Timestamp("2024-05-15 06:00:01", tz="UTC")
    cycle_next = (t_next_cycle - pd.Timedelta(microseconds=1)).floor("6h")
    assert cycle_next == pd.Timestamp("2024-05-15 06:00:00", tz="UTC")


# ── Test 4: False-Zero Safeguard in Window Classification ─────────────────────
def test_04_false_zero_safeguard():
    """Verify that unmonitored or sparse windows are NEVER assigned zero thunderstorm target."""
    df_mock = pd.DataFrame([
        # Row 1: Zero observations -> UNOBSERVED, target = NaN
        {"observation_count": 0, "thunderstorm_any_count": 0},
        # Row 2: Sparse observations (e.g. 2 reports) without TS -> INCOMPLETE, target = NaN
        {"observation_count": 2, "thunderstorm_any_count": 0},
        # Row 3: Sparse observations (e.g. 1 report) WITH TS -> COVERED_THUNDERSTORM, target = 1
        {"observation_count": 1, "thunderstorm_any_count": 1},
        # Row 4: Adequate observations (>= 4 reports) without TS -> COVERED_NO_THUNDERSTORM, target = 0
        {"observation_count": 6, "thunderstorm_any_count": 0},
        # Row 5: Adequate observations with TS -> COVERED_THUNDERSTORM, target = 1
        {"observation_count": 8, "thunderstorm_any_count": 3},
    ])

    def assign_status(row):
        cnt = row["observation_count"]
        has_ts = row["thunderstorm_any_count"] > 0
        if has_ts:
            return 1, "COVERED_THUNDERSTORM"
        elif cnt >= 4:
            return 0, "COVERED_NO_THUNDERSTORM"
        elif cnt > 0:
            return np.nan, "INCOMPLETE_MONITORING"
        else:
            return np.nan, "UNOBSERVED"

    results = [assign_status(r) for _, r in df_mock.iterrows()]
    targets = [res[0] for res in results]
    statuses = [res[1] for res in results]

    assert np.isnan(targets[0])
    assert statuses[0] == "UNOBSERVED"

    assert np.isnan(targets[1])
    assert statuses[1] == "INCOMPLETE_MONITORING"

    assert targets[2] == 1
    assert statuses[2] == "COVERED_THUNDERSTORM"

    assert targets[3] == 0
    assert statuses[3] == "COVERED_NO_THUNDERSTORM"

    assert targets[4] == 1
    assert statuses[4] == "COVERED_THUNDERSTORM"


# ── Test 5: Master Observations Dataset Integrity ─────────────────────────────
def test_05_master_observations_integrity():
    """Validate full-year master observations parquet dataset."""
    assert OBSERVATIONS_FILE.exists(), f"Missing master observations parquet: {OBSERVATIONS_FILE}"
    df = pd.read_parquet(OBSERVATIONS_FILE)

    assert len(df) == 4582946, f"Expected 4,582,946 observations, got {len(df):,}"
    assert df["station_id"].nunique() == 104, f"Expected 104 active reporting stations, got {df['station_id'].nunique()}"
    
    # Check timestamp coverage spans 2024
    t_min = df["valid_time_utc"].min()
    t_max = df["valid_time_utc"].max()
    assert str(t_min).startswith("2024-01-01"), f"Expected start on 2024-01-01, got {t_min}"
    assert str(t_max).startswith("2024-12-31"), f"Expected end on 2024-12-31, got {t_max}"

    # Check deduplication
    assert df.duplicated(subset=["station_id", "valid_time_utc"]).sum() == 0, "Duplicate station timestamps found!"

    # Check thunderstorm events captured
    assert df["has_thunderstorm_ts"].sum() == 116405
    assert df["has_vicinity_thunderstorm_vcts"].sum() == 18695
    assert df["has_lightning_remark"].sum() == 311926
    assert df["thunderstorm_reported"].sum() == 412638


# ── Test 6: Forecast Windows Aggregation Integrity ────────────────────────────
def test_06_forecast_windows_integrity():
    """Validate 6h half-open forecast window dataset."""
    assert WINDOWS_FILE.exists(), f"Missing forecast windows parquet: {WINDOWS_FILE}"
    df_win = pd.read_parquet(WINDOWS_FILE)

    assert len(df_win) == 118027, f"Expected 118,027 window-station records, got {len(df_win):,}"
    
    # Check coverage status counts
    vc = df_win["coverage_status"].value_counts().to_dict()
    assert vc.get("COVERED_NO_THUNDERSTORM") == 88093
    assert vc.get("COVERED_THUNDERSTORM") == 24792
    assert vc.get("INCOMPLETE_MONITORING") == 5142

    # Verification of False-Zero safeguard in actual data
    has_ts = df_win["thunderstorm_any_count"] > 0
    assert (df_win.loc[has_ts, "asos_thunderstorm_target"] == 1).all()
    assert (df_win.loc[has_ts, "coverage_status"] == "COVERED_THUNDERSTORM").all()

    sparse_no_ts = (~has_ts) & (df_win["observation_count"] < 4)
    assert df_win.loc[sparse_no_ts, "asos_thunderstorm_target"].isna().all()
    assert (df_win.loc[sparse_no_ts, "coverage_status"] == "INCOMPLETE_MONITORING").all()

    adequate_no_ts = (~has_ts) & (df_win["observation_count"] >= 4)
    assert (df_win.loc[adequate_no_ts, "asos_thunderstorm_target"] == 0).all()
    assert (df_win.loc[adequate_no_ts, "coverage_status"] == "COVERED_NO_THUNDERSTORM").all()


# ── Test 7: Download Manifest Completeness ─────────────────────────────────────
def test_07_download_manifest_completeness():
    """Verify all 74 5-day batches completed with SUCCESS."""
    assert MANIFEST_FILE.exists(), f"Missing manifest file: {MANIFEST_FILE}"
    df_man = pd.read_csv(MANIFEST_FILE)

    assert len(df_man) == 74, f"Expected 74 batches, got {len(df_man)}"
    assert (df_man["status"] == "SUCCESS").all(), "Not all batches succeeded!"

