"""
GLM Ground Truth Specification Logic and Validation Functions.

Implements the formal rules for:
1. Target A (GLM Lightning Occurrence: 1, 0, NULL) over (t, t+6h]
2. Target B (NOAA Severe Convective Events: 1, 0)
3. Defensible Coverage Criteria (COMPLETE, PARTIAL, MISSING, INVALID, UNKNOWN)
4. Temporal boundary handling (strictly (t, t+6h], start excluded, end included)
5. Prevention of synthetic multipliers (prohibits flash_count * 12 or * 28)
6. Model-training eligibility filtering (prevents NULL from becoming 0)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd


# Expected granules in a full 6-hour synoptic window (6 hours * 60 min * 3 granules/min)
EXPECTED_GRANULES_6H = 1080
# Minimum valid granules required for COMPLETE coverage (95% of expected)
MIN_GRANULES_COMPLETE = 1026
# Maximum allowable consecutive missing duration (15 minutes = 45 granules)
MAX_CONSECUTIVE_GAP_MINUTES = 15.0


class CoverageStatus:
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


@dataclass
class GranuleMetadata:
    granule_id: str
    start_time_utc: datetime
    end_time_utc: datetime
    is_corrupted: bool = False
    percent_uncorrectable_l0: float = 0.0
    percent_navigated_l1b: float = 100.0


@dataclass
class FlashRecord:
    flash_id: int
    timestamp_utc: datetime
    latitude: float
    longitude: float
    quality_flag: int = 0


def evaluate_window_coverage(
    init_time_utc: datetime,
    granules: List[GranuleMetadata],
    window_hours: float = 6.0
) -> str:
    """
    Evaluates coverage status for the forward window (init_time, init_time + window_hours].
    
    Status definitions:
    - COMPLETE: >= 95% valid granules present AND max consecutive gap <= 15 min AND all valid.
    - PARTIAL: Some valid granules present (< 95% or gap > 15 min).
    - MISSING: 0 granules present.
    - INVALID: Any present granule is corrupted or exceeds uncorrectable L0 error threshold (> 5%).
    - UNKNOWN: Granules list is None or metadata indeterminate.
    """
    if granules is None:
        return CoverageStatus.UNKNOWN

    window_end = init_time_utc + timedelta(hours=window_hours)
    
    # Filter granules touching (init_time_utc, window_end]
    window_granules = [
        g for g in granules 
        if g.end_time_utc > init_time_utc and g.start_time_utc <= window_end
    ]
    
    if len(window_granules) == 0:
        return CoverageStatus.MISSING
        
    # Check for corruption or telemetry invalidity
    for g in window_granules:
        if g.is_corrupted or g.percent_uncorrectable_l0 > 5.0 or g.percent_navigated_l1b < 90.0:
            return CoverageStatus.INVALID
            
    # Check granule count completeness
    valid_count = len(window_granules)
    if valid_count < MIN_GRANULES_COMPLETE:
        return CoverageStatus.PARTIAL
        
    # Check max consecutive gap
    sorted_granules = sorted(window_granules, key=lambda x: x.start_time_utc)
    # Check gap at start
    gap_start = (sorted_granules[0].start_time_utc - init_time_utc).total_seconds() / 60.0
    if gap_start > MAX_CONSECUTIVE_GAP_MINUTES:
        return CoverageStatus.PARTIAL
        
    # Check internal gaps
    for i in range(len(sorted_granules) - 1):
        gap = (sorted_granules[i+1].start_time_utc - sorted_granules[i].end_time_utc).total_seconds() / 60.0
        if gap > MAX_CONSECUTIVE_GAP_MINUTES:
            return CoverageStatus.PARTIAL
            
    # Check gap at end
    gap_end = (window_end - sorted_granules[-1].end_time_utc).total_seconds() / 60.0
    if gap_end > MAX_CONSECUTIVE_GAP_MINUTES:
        return CoverageStatus.PARTIAL
        
    return CoverageStatus.COMPLETE


def filter_flashes_in_window(
    init_time_utc: datetime,
    flashes: List[FlashRecord],
    window_hours: float = 6.0
) -> List[FlashRecord]:
    """
    Filters flashes to strictly the half-open interval (init_time_utc, init_time_utc + window_hours].
    - Flashes exactly at init_time_utc are EXCLUDED (belong to preceding window).
    - Flashes exactly at init_time_utc + window_hours are INCLUDED.
    """
    window_end = init_time_utc + timedelta(hours=window_hours)
    return [
        f for f in flashes
        if init_time_utc < f.timestamp_utc <= window_end and f.quality_flag == 0
    ]


def compute_target_a_glm(
    init_time_utc: datetime,
    granules: List[GranuleMetadata],
    flashes: List[FlashRecord],
    window_hours: float = 6.0
) -> Tuple[Optional[int], str, int]:
    """
    Computes Target A (GLM-observed lightning occurrence):
    - State 1: At least one valid flash observed in (t, t+6h] (even under PARTIAL coverage).
    - State 0: Exactly 0 flashes observed AND coverage is COMPLETE.
    - State NULL (None): Insufficient coverage to verify 0 lightning (PARTIAL, MISSING, INVALID, UNKNOWN).
    
    Returns: (target_a, coverage_status, valid_flash_count)
    """
    coverage = evaluate_window_coverage(init_time_utc, granules, window_hours)
    valid_flashes = filter_flashes_in_window(init_time_utc, flashes, window_hours)
    n_flashes = len(valid_flashes)
    
    if n_flashes > 0:
        # Physical detection of lightning confirms occurrence regardless of whether remaining window was observed
        return 1, coverage, n_flashes
    else:
        # Zero flashes detected: Can ONLY assign 0 if coverage was COMPLETE
        if coverage == CoverageStatus.COMPLETE:
            return 0, coverage, 0
        else:
            return None, coverage, 0


def compute_target_b_noaa(
    init_time_utc: datetime,
    noaa_events: List[Dict[str, Any]],
    window_hours: float = 6.0
) -> Tuple[int, int]:
    """
    Computes Target B (NOAA-reported severe convective events):
    - Strictly separate from GLM.
    - State 1: At least one severe convective event (Wind >= 50kt, Hail >= 0.75 in, Tornado) in (t, t+6h].
    - State 0: No NOAA severe events reported.
    
    Returns: (target_b, noaa_event_count)
    """
    window_end = init_time_utc + timedelta(hours=window_hours)
    valid_events = [
        e for e in noaa_events
        if init_time_utc < e["timestamp_utc"] <= window_end
    ]
    return (1 if len(valid_events) > 0 else 0), len(valid_events)


def validate_group_and_event_extraction(
    measured_groups: Optional[int],
    measured_events: Optional[int],
    flash_count: int,
    allow_synthetic: bool = False
) -> Dict[str, Any]:
    """
    Enforces strict extraction requirements:
    Rejects hardcoded synthetic multipliers (group = flash * 12, event = flash * 28).
    Requires measured group and event counts from L2 arrays.
    """
    if measured_groups is None or measured_events is None:
        if allow_synthetic:
            return {
                "group_count": flash_count * 12,
                "event_count": flash_count * 28,
                "is_synthetic": True,
                "is_valid": False
            }
        raise ValueError(
            "CRITICAL: Synthetic group/event estimation (flash*12, flash*28) is strictly prohibited. "
            "Must parse measured group_lat and event_lat from source NetCDF product."
        )
    return {
        "group_count": measured_groups,
        "event_count": measured_events,
        "is_synthetic": False,
        "is_valid": True
    }


def filter_training_samples(
    df: pd.DataFrame,
    target_col: str = "target_a_glm"
) -> pd.DataFrame:
    """
    Enforces the rule that NULL / unknown target labels MUST NEVER enter supervised training as negatives.
    Filters exclusively to non-null target values (1 or 0).
    """
    # Guard against accidental imputation of NULL as 0
    clean_df = df[df[target_col].notna()].copy()
    assert not clean_df[target_col].isna().any(), "Target column contains NaNs after filtering!"
    # Ensure targets are strictly binary 0 or 1
    assert clean_df[target_col].isin([0, 1]).all(), "Target column contains invalid non-binary states!"
    return clean_df
