"""
Unit and regression tests for Florida 2024 Lightning Ground Truth Specification.
Covers all 12 required specification scenarios:
1. Complete observed interval containing valid flashes.
2. Complete observed interval containing no valid flashes.
3. Missing granule (0 granules).
4. Partial coverage (< 95% granules or gap > 15 min).
5. Corrupted or invalid granule.
6. Unknown coverage metadata.
7. Flash exactly at interval start t (must be excluded from (t, t+6h]).
8. Flash exactly at interval end t+6h (must be included in (t, t+6h]).
9. Missing flash data versus an actual zero count.
10. Separation of GLM lightning occurrence and NOAA severe convective labels.
11. Prevention of synthetic group/event count calculations (prohibits flash*12 / flash*28).
12. Prevention of unknown/NULL target labels from entering supervised training as negatives.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pytest
import pandas as pd
import numpy as np

from src.glm_ground_truth_spec import (
    GranuleMetadata,
    FlashRecord,
    CoverageStatus,
    evaluate_window_coverage,
    filter_flashes_in_window,
    compute_target_a_glm,
    compute_target_b_noaa,
    validate_group_and_event_extraction,
    filter_training_samples,
    EXPECTED_GRANULES_6H,
    MIN_GRANULES_COMPLETE
)


def generate_granule_series(
    start_utc: datetime,
    count: int,
    granule_duration_sec: int = 20,
    has_corruption: bool = False,
    gap_after_idx: Optional[int] = None,
    gap_duration_minutes: float = 30.0
) -> list:
    """Helper to generate synthetic sequences of 20-second GLM granules."""
    granules = []
    curr = start_utc
    for i in range(count):
        if gap_after_idx is not None and i == gap_after_idx:
            curr += timedelta(minutes=gap_duration_minutes)
        g_end = curr + timedelta(seconds=granule_duration_sec)
        is_corrupt = has_corruption and (i == 10)
        granules.append(GranuleMetadata(
            granule_id=f"G16_{curr.strftime('%Y%j%H%M%S')}",
            start_time_utc=curr,
            end_time_utc=g_end,
            is_corrupted=is_corrupt,
            percent_uncorrectable_l0=10.0 if is_corrupt else 0.0
        ))
        curr = g_end
    return granules


# ── Test 1: Complete observed interval containing valid flashes ───────────────
def test_01_complete_observed_with_flashes():
    t0 = datetime(2024, 5, 15, 0, 0, 0)
    # 1080 granules = full 6 hours (00:00 to 06:00 UTC)
    granules = generate_granule_series(t0, count=EXPECTED_GRANULES_6H)
    flashes = [
        FlashRecord(flash_id=1, timestamp_utc=t0 + timedelta(hours=2), latitude=28.0, longitude=-81.5),
        FlashRecord(flash_id=2, timestamp_utc=t0 + timedelta(hours=4, minutes=30), latitude=28.25, longitude=-81.25)
    ]
    target_a, coverage, count = compute_target_a_glm(t0, granules, flashes)
    assert coverage == CoverageStatus.COMPLETE
    assert target_a == 1
    assert count == 2


# ── Test 2: Complete observed interval containing no valid flashes ────────────
def test_02_complete_observed_with_no_flashes():
    t0 = datetime(2024, 5, 15, 0, 0, 0)
    granules = generate_granule_series(t0, count=EXPECTED_GRANULES_6H)
    flashes = []  # Exactly zero flashes
    target_a, coverage, count = compute_target_a_glm(t0, granules, flashes)
    assert coverage == CoverageStatus.COMPLETE
    assert target_a == 0  # Defensibly verified negative!
    assert count == 0


# ── Test 3: Missing granule (0 granules) ──────────────────────────────────────
def test_03_missing_granules():
    t0 = datetime(2024, 5, 15, 0, 0, 0)
    granules = []  # 0 granules present
    flashes = []
    target_a, coverage, count = compute_target_a_glm(t0, granules, flashes)
    assert coverage == CoverageStatus.MISSING
    assert target_a is None  # MUST NOT be assigned 0!
    assert count == 0


# ── Test 4: Partial coverage (< 95% or gap > 15 min) ──────────────────────────
def test_04_partial_coverage_behavior():
    t0 = datetime(2024, 5, 15, 0, 0, 0)
    # Scenario A: Only 100 granules present (like the pilot archive)
    sparse_granules = generate_granule_series(t0, count=100)
    target_a, coverage, count = compute_target_a_glm(t0, sparse_granules, [])
    assert coverage == CoverageStatus.PARTIAL
    assert target_a is None  # Without flashes, PARTIAL cannot prove negative!

    # Scenario B: Gap > 15 minutes inside otherwise complete series
    gapped_granules = generate_granule_series(t0, count=1050, gap_after_idx=500, gap_duration_minutes=25.0)
    target_a_gap, cov_gap, _ = compute_target_a_glm(t0, gapped_granules, [])
    assert cov_gap == CoverageStatus.PARTIAL
    assert target_a_gap is None

    # Scenario C: If flash occurs during PARTIAL, target_a is verified positive (1)
    flash_in_partial = [FlashRecord(flash_id=1, timestamp_utc=t0 + timedelta(minutes=10), latitude=28.0, longitude=-81.5)]
    target_a_pos, cov_pos, c_pos = compute_target_a_glm(t0, sparse_granules, flash_in_partial)
    assert cov_pos == CoverageStatus.PARTIAL
    assert target_a_pos == 1  # 1 confirmed flash proves occurrence regardless of missing remainder
    assert c_pos == 1


# ── Test 5: Corrupted granule ─────────────────────────────────────────────────
def test_05_corrupted_granule():
    t0 = datetime(2024, 5, 15, 0, 0, 0)
    corrupt_granules = generate_granule_series(t0, count=EXPECTED_GRANULES_6H, has_corruption=True)
    target_a, coverage, count = compute_target_a_glm(t0, corrupt_granules, [])
    assert coverage == CoverageStatus.INVALID
    assert target_a is None


# ── Test 6: Unknown coverage metadata ─────────────────────────────────────────
def test_06_unknown_coverage():
    t0 = datetime(2024, 5, 15, 0, 0, 0)
    target_a, coverage, count = compute_target_a_glm(t0, None, [])
    assert coverage == CoverageStatus.UNKNOWN
    assert target_a is None


# ── Test 7: Flash exactly at interval start t ─────────────────────────────────
def test_07_flash_exactly_at_interval_start():
    t0 = datetime(2024, 5, 15, 0, 0, 0)
    # Flash occurs exactly on the boundary: t0
    # In half-open interval (t, t+6h], t0 is strictly EXCLUDED
    flashes = [
        FlashRecord(flash_id=101, timestamp_utc=t0, latitude=28.0, longitude=-81.5),
        FlashRecord(flash_id=102, timestamp_utc=t0 + timedelta(seconds=1), latitude=28.0, longitude=-81.5)
    ]
    filtered = filter_flashes_in_window(t0, flashes, window_hours=6.0)
    assert len(filtered) == 1
    assert filtered[0].flash_id == 102
    assert filtered[0].timestamp_utc > t0


# ── Test 8: Flash exactly at interval end t+6h ────────────────────────────────
def test_08_flash_exactly_at_interval_end():
    t0 = datetime(2024, 5, 15, 0, 0, 0)
    t_end = t0 + timedelta(hours=6)
    # In half-open interval (t, t+6h], t_end is strictly INCLUDED
    flashes = [
        FlashRecord(flash_id=201, timestamp_utc=t_end, latitude=28.0, longitude=-81.5),
        FlashRecord(flash_id=202, timestamp_utc=t_end + timedelta(seconds=1), latitude=28.0, longitude=-81.5)
    ]
    filtered = filter_flashes_in_window(t0, flashes, window_hours=6.0)
    assert len(filtered) == 1
    assert filtered[0].flash_id == 201
    assert filtered[0].timestamp_utc == t_end


# ── Test 9: Missing flash data vs. actual zero count ──────────────────────────
def test_09_missing_data_vs_actual_zero():
    t0 = datetime(2024, 5, 15, 0, 0, 0)
    # Case A: Complete monitoring, verified zero
    g_complete = generate_granule_series(t0, count=EXPECTED_GRANULES_6H)
    t_a_zero, cov_zero, n_zero = compute_target_a_glm(t0, g_complete, [])
    assert t_a_zero == 0
    assert cov_zero == CoverageStatus.COMPLETE
    assert n_zero == 0

    # Case B: Zero granules, missing data
    t_a_miss, cov_miss, n_miss = compute_target_a_glm(t0, [], [])
    assert t_a_miss is None
    assert cov_miss == CoverageStatus.MISSING
    assert n_miss == 0

    # Ensure they are mathematically distinct
    assert t_a_zero != t_a_miss
    assert cov_zero != cov_miss


# ── Test 10: Separation of GLM and NOAA labels ────────────────────────────────
def test_10_separation_of_glm_and_noaa_labels():
    t0 = datetime(2024, 5, 15, 0, 0, 0)
    g_complete = generate_granule_series(t0, count=EXPECTED_GRANULES_6H)
    flashes = [FlashRecord(flash_id=1, timestamp_utc=t0 + timedelta(hours=1), latitude=28.0, longitude=-81.5)]
    
    # Subcase A: Convective storm with GLM lightning, but NO severe NOAA report (pulse storm)
    noaa_empty = []
    t_a, _, _ = compute_target_a_glm(t0, g_complete, flashes)
    t_b, n_b = compute_target_b_noaa(t0, noaa_empty)
    assert t_a == 1
    assert t_b == 0  # Independent!
    assert n_b == 0

    # Subcase B: Severe NOAA report (wind damage), but GLM unmonitored
    noaa_severe = [{"timestamp_utc": t0 + timedelta(hours=2), "event_type": "Thunderstorm Wind"}]
    t_a_unmon, cov_unmon, _ = compute_target_a_glm(t0, [], [])
    t_b_sev, n_sev = compute_target_b_noaa(t0, noaa_severe)
    assert t_a_unmon is None  # NOAA presence does NOT invent GLM flashes!
    assert cov_unmon == CoverageStatus.MISSING
    assert t_b_sev == 1
    assert n_sev == 1


# ── Test 11: Prevention of synthetic group/event count calculations ───────────
def test_11_prevention_of_synthetic_group_event_calculations():
    # Attempting synthetic estimation must raise ValueError unless explicitly flagged as invalid
    with pytest.raises(ValueError, match="Synthetic group/event estimation .* is strictly prohibited"):
        validate_group_and_event_extraction(
            measured_groups=None,
            measured_events=None,
            flash_count=10,
            allow_synthetic=False
        )

    # Valid measured counts pass successfully
    res = validate_group_and_event_extraction(
        measured_groups=124,
        measured_events=312,
        flash_count=10,
        allow_synthetic=False
    )
    assert res["is_valid"] is True
    assert res["is_synthetic"] is False
    assert res["group_count"] == 124
    assert res["event_count"] == 312


# ── Test 12: Prevention of unknown target labels entering supervised training ─
def test_12_prevention_of_unknown_labels_entering_training():
    # Create sample DataFrame containing mix of 1, 0, and NULL (NaN)
    df_raw = pd.DataFrame({
        "cell_id": [f"FL_{i}" for i in range(6)],
        "cape": [1200, 1500, 800, 2200, 400, 950],
        "target_a_glm": [1.0, 0.0, np.nan, 1.0, np.nan, 0.0]
    })
    
    # Filtering function strictly removes NULL rows
    df_train = filter_training_samples(df_raw, target_col="target_a_glm")
    
    assert len(df_train) == 4
    assert not df_train["target_a_glm"].isna().any()
    assert set(df_train["target_a_glm"].unique()) == {0.0, 1.0}
    # Ensure unobserved rows (NaN) were NOT coerced to 0.0
    assert "FL_2" not in df_train["cell_id"].values
    assert "FL_4" not in df_train["cell_id"].values
