"""
test_glm_extraction_reproducibility.py
======================================
Comprehensive reproducibility test suite for the improved GOES-16 GLM flash
extraction pipeline.

Validates:
1. Complete preservation of Florida polygon flashes (exact count = 181).
2. Terrestrial active grid vs coastal marine partition (173 terrestrial, 8 coastal marine).
3. Rejection accounting: 2,477 rejected out of 2,658 raw flashes (1,970 out of bbox, 507 out of polygon).
4. Sub-second UTC timestamp reconstruction (verifying microsecond precision and zero synoptic collapse).
5. Elimination of synthetic multipliers (measured group and event counts from NetCDF hierarchy: 4,534 groups, 9,243 events).
6. Quality flag accuracy (172 good quality, 9 degraded due to high event count).
7. Zero duplicate flash records across granule boundaries.
8. Parquet and CSV schema, column names, and row count parity.
9. Granule audit log completeness across all 12 granules (0% uncorrectable L0 errors, 100% navigated L1b events).
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

DATA_DIR = Path("data/processed/glm_pilot_extracted")
SUBPROJECT_DATA_DIR = Path("Florida_Thunderstorm_2024_Training/data/processed/glm_pilot_extracted")

PARQUET_FILE = DATA_DIR / "florida_glm_flash_observations_2024.parquet"
CSV_FILE = DATA_DIR / "florida_glm_flash_observations_2024.csv"
AUDIT_FILE = DATA_DIR / "glm_extraction_audit.csv"
REJECTION_FILE = DATA_DIR / "glm_rejection_audit.csv"


@pytest.fixture(scope="module")
def df_flashes() -> pd.DataFrame:
    """Load extracted Parquet flash observations."""
    path = PARQUET_FILE if PARQUET_FILE.exists() else (SUBPROJECT_DATA_DIR / "florida_glm_flash_observations_2024.parquet")
    assert path.exists(), f"Extracted flashes Parquet file not found at {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def df_audit() -> pd.DataFrame:
    """Load granule extraction audit CSV."""
    path = AUDIT_FILE if AUDIT_FILE.exists() else (SUBPROJECT_DATA_DIR / "glm_extraction_audit.csv")
    assert path.exists(), f"Granule audit CSV not found at {path}"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def df_rejections() -> pd.DataFrame:
    """Load observation rejections CSV."""
    path = REJECTION_FILE if REJECTION_FILE.exists() else (SUBPROJECT_DATA_DIR / "glm_rejection_audit.csv")
    assert path.exists(), f"Rejections CSV not found at {path}"
    return pd.read_csv(path)


def test_01_exact_florida_flash_counts(df_flashes):
    """Test exact count of accepted Florida flashes and terrestrial vs coastal partition."""
    assert len(df_flashes) == 181, f"Expected 181 Florida polygon flashes, got {len(df_flashes)}"
    assert (df_flashes["in_florida_polygon"] == True).all(), "All extracted records must be in Florida polygon"

    terrestrial_count = (df_flashes["in_active_terrestrial_grid"] == True).sum()
    coastal_marine_count = (df_flashes["in_active_terrestrial_grid"] == False).sum()

    assert terrestrial_count == 173, f"Expected 173 active terrestrial flashes, got {terrestrial_count}"
    assert coastal_marine_count == 8, f"Expected 8 coastal marine flashes, got {coastal_marine_count}"
    assert terrestrial_count + coastal_marine_count == 181


def test_02_rejection_accounting_integrity(df_flashes, df_rejections, df_audit):
    """Test that all raw flashes are strictly accounted for (accepted + rejected = raw)."""
    total_raw = df_audit["raw_flash_count"].sum()
    assert total_raw == 2658, f"Expected 2,658 raw flashes across 12 granules, got {total_raw}"

    n_accepted = len(df_flashes)
    n_rejected = len(df_rejections)

    assert n_accepted == 181
    assert n_rejected == 2477
    assert n_accepted + n_rejected == total_raw

    rej_counts = df_rejections["rejection_reason"].value_counts()
    assert rej_counts["OUT_OF_DOMAIN_BBOX"] == 1970
    assert rej_counts["OUTSIDE_FLORIDA_POLYGON"] == 507


def test_03_timestamp_subsecond_precision(df_flashes):
    """Test that flash observation timestamps preserve microsecond precision and do NOT collapse."""
    ts = pd.to_datetime(df_flashes["flash_timestamp_utc"])

    # Check that timestamps are not all rounded to exact second / minute / hour
    microsecond_values = ts.dt.microsecond
    assert (microsecond_values > 0).any(), "Timestamps must preserve sub-second microsecond precision"
    assert len(microsecond_values.unique()) > 100, "Observation times should be uniquely dispersed"

    # Verify no timestamps are collapsed to synoptic hours (00:00:00, 06:00:00, 12:00:00, 18:00:00)
    synoptic_exact_matches = ts.isin([
        pd.Timestamp("2024-05-15 00:00:00", tz="UTC"),
        pd.Timestamp("2024-05-15 06:00:00", tz="UTC"),
        pd.Timestamp("2024-05-15 12:00:00", tz="UTC"),
        pd.Timestamp("2024-05-15 18:00:00", tz="UTC"),
    ]).sum()
    assert synoptic_exact_matches == 0, "Observation timestamps must not be truncated to synoptic hours"


def test_04_measured_group_and_event_hierarchy(df_flashes, df_audit):
    """Test that physical constituent group and event counts are measured without synthetic multipliers."""
    # Ensure every single flash has at least 1 measured constituent group and event
    assert (df_flashes["group_count_measured"] >= 1).all(), "Every physical flash must contain >= 1 group"
    assert (df_flashes["event_count_measured"] >= 1).all(), "Every physical flash must contain >= 1 event"

    total_hierarchy_groups = df_flashes["group_count_measured"].sum()
    total_hierarchy_events = df_flashes["event_count_measured"].sum()

    assert total_hierarchy_groups == 4534, f"Expected 4,534 hierarchy groups, got {total_hierarchy_groups}"
    assert total_hierarchy_events == 9243, f"Expected 9,243 hierarchy events, got {total_hierarchy_events}"

    # Mean ratios must reflect physical measurements, NOT synthetic constants (12 and 28)
    mean_groups_per_flash = df_flashes["group_count_measured"].mean()
    mean_events_per_flash = df_flashes["event_count_measured"].mean()

    assert not np.isclose(mean_groups_per_flash, 12.0), "Group count must not use synthetic 12x multiplier"
    assert not np.isclose(mean_events_per_flash, 28.0), "Event count must not use synthetic 28x multiplier"
    assert 24.0 < mean_groups_per_flash < 26.0  # Measured: 25.05
    assert 50.0 < mean_events_per_flash < 52.0  # Measured: 51.07

    # Spatial centroid containment from audit log
    total_spatial_groups = df_audit["polygon_group_count_spatial"].sum()
    total_spatial_events = df_audit["polygon_event_count_spatial"].sum()
    assert total_spatial_groups == 4480, f"Expected 4,480 spatial polygon groups, got {total_spatial_groups}"
    assert total_spatial_events == 9073, f"Expected 9,073 spatial polygon events, got {total_spatial_events}"


def test_05_quality_flags_audit(df_flashes):
    """Test that flash quality flags and meanings are correctly preserved and interpreted."""
    qf_counts = df_flashes["flash_quality_flag"].value_counts()
    assert qf_counts.get(0, 0) == 172, f"Expected 172 good quality flashes, got {qf_counts.get(0, 0)}"
    assert qf_counts.get(3, 0) == 9, f"Expected 9 degraded-event-count flashes, got {qf_counts.get(3, 0)}"
    assert len(qf_counts) == 2, f"Unexpected quality flags detected: {qf_counts.to_dict()}"

    # Verify quality meanings
    assert (df_flashes[df_flashes["flash_quality_flag"] == 0]["flash_quality_meaning"] == "good_quality_qf").all()
    assert (df_flashes[df_flashes["flash_quality_flag"] == 3]["flash_quality_meaning"] == "degraded_due_to_flash_constituent_event_count_exceeds_threshold_qf").all()


def test_06_zero_cross_granule_duplicates(df_flashes):
    """Test that no duplicate flash records exist within or across granules."""
    within_granule_dups = df_flashes.duplicated(subset=["source_granule_id", "flash_id"])
    assert within_granule_dups.sum() == 0, "No duplicate flash_id within any granule"

    cross_granule_dups = df_flashes.duplicated(subset=["latitude", "longitude", "flash_timestamp_utc"])
    assert cross_granule_dups.sum() == 0, "No duplicate flashes across granules"


def test_07_parquet_and_csv_parity(df_flashes):
    """Test that CSV mirror matches Parquet data exactly in row count and core columns."""
    path_csv = CSV_FILE if CSV_FILE.exists() else (SUBPROJECT_DATA_DIR / "florida_glm_flash_observations_2024.csv")
    assert path_csv.exists(), f"CSV mirror file not found at {path_csv}"
    df_csv = pd.read_csv(path_csv)

    assert len(df_csv) == len(df_flashes) == 181
    assert set(df_csv.columns) == set(df_flashes.columns)
    np.testing.assert_allclose(df_csv["latitude"].values, df_flashes["latitude"].values)
    np.testing.assert_allclose(df_csv["longitude"].values, df_flashes["longitude"].values)
    np.testing.assert_allclose(df_csv["flash_energy_joules"].values, df_flashes["flash_energy_joules"].values)


def test_08_granule_audit_integrity(df_audit):
    """Test that granule audit records contain all 12 granules with zero uncorrectable errors."""
    assert len(df_audit) == 12, f"Expected 12 granules in audit log, got {len(df_audit)}"
    assert (df_audit["percent_uncorrectable_l0_errors"] == 0.0).all(), "All granules must have 0.0% uncorrectable L0 errors"
    assert (df_audit["percent_navigated_l1b_events"] == 1.0).all(), "All granules must have 100% navigated L1b events"
    assert (df_audit["extraction_status"] == "PROCESSED_SUCCESS").all()
    assert (df_audit["duplicate_flashes_detected"] == 0).all()
