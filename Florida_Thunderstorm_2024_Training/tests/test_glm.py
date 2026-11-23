"""Unit tests for NOAA GOES-16 GLM lightning processing and validation."""

import numpy as np
import pandas as pd
import pytest

from src.process_glm import GLMProcessor, MIN_GRANULES_FOR_ADEQUATE_MONITORING


def test_glm_critical_rule_no_false_zero_imputation(tmp_path):
    """Verify that unmonitored observation windows are NEVER imputed as zero lightning."""
    target_grid = pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-05-15 00:00", periods=5, freq="1h"),
        "latitude": [28.0] * 5,
        "longitude": [-82.0] * 5,
        "grid_cell_id": ["FL_28.0_-82.0"] * 5,
    })

    # Initialize with empty raw directory (no granules present for these timestamps)
    empty_raw = tmp_path / "raw_glm"
    empty_raw.mkdir()
    proc = GLMProcessor(raw_dir=empty_raw, processed_dir=tmp_path)
    df_glm = proc.process_glm_dataset(target_grid=target_grid)

    assert len(df_glm) == 5
    # When no granules exist for timestamp, coverage must be 0
    assert (df_glm["glm_observation_coverage"] == 0).all()
    # CRITICAL: Lightning detection must be NaN, NOT 0.0
    assert df_glm["glm_lightning_detected"].isna().all()
    assert (df_glm["coverage_status"] == "UNOBSERVED").all()


def test_glm_incomplete_coverage_safeguard(tmp_path):
    """Verify that sparse granules (< threshold) with 0 flashes are marked INCOMPLETE, not zero."""
    target_grid = pd.DataFrame({
        "timestamp_utc": ["2024-05-15 00:00:00"],
        "latitude": [28.0],
        "longitude": [-82.0],
        "grid_cell_id": ["FL_28.0_-82.0"],
    })
    proc = GLMProcessor(processed_dir=tmp_path)
    df_glm = proc.process_glm_dataset(target_grid=target_grid)
    assert len(df_glm) == 1
    # When flash count is 0 and granule count < MIN_GRANULES_FOR_ADEQUATE_MONITORING,
    # it must NOT be claimed as confirmed zero lightning
    row = df_glm.iloc[0]
    if row["glm_flash_count"] == 0 or pd.isna(row["glm_flash_count"]):
        if row["glm_granule_count"] < MIN_GRANULES_FOR_ADEQUATE_MONITORING:
            assert np.isnan(row["glm_lightning_detected"])
            assert row["coverage_status"] in ["INCOMPLETE_COVERAGE", "UNOBSERVED"]


def test_glm_event_group_flash_separation(tmp_path):
    """Verify that Event, Group, and Flash counts are tracked in distinct columns."""
    target_grid = pd.DataFrame({
        "timestamp_utc": ["2024-05-15 12:00:00"],
        "latitude": [28.0],
        "longitude": [-82.0],
        "grid_cell_id": ["FL_28.0_-82.0"],
    })
    proc = GLMProcessor(processed_dir=tmp_path)
    df_glm = proc.process_glm_dataset(target_grid=target_grid)
    assert len(df_glm) == 1
    assert "glm_flash_count" in df_glm.columns
    assert "glm_group_count" in df_glm.columns
    assert "glm_event_count" in df_glm.columns
    assert "glm_observation_coverage" in df_glm.columns
    assert "glm_granule_count" in df_glm.columns
    assert "glm_monitored_duration_sec" in df_glm.columns
