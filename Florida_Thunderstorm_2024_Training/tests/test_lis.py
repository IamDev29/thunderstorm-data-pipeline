"""Unit tests for NASA ISS LIS processing and scientific constraint validation."""

import numpy as np
import pandas as pd
import pytest

from src.process_lis import LISProcessor


def test_critical_lis_rule_no_false_zero_imputation(tmp_path):
    """Verify that unobserved satellite coverage is NEVER imputed as zero lightning."""
    target_grid = pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-05-15 00:00", periods=5, freq="1h"),
        "latitude": [28.0] * 5,
        "longitude": [-82.0] * 5,
        "grid_cell_id": ["FL_28.0_-82.0"] * 5,
    })

    processor = LISProcessor(processed_dir=tmp_path)
    df_lis = processor.process_lis_dataset(target_grid=target_grid)

    assert len(df_lis) == 5
    # When satellite is not viewing, coverage must be 0
    assert (df_lis["lis_observation_coverage"] == 0).all()
    # CRITICAL: Lightning detection must be NaN, NOT 0.0
    assert df_lis["lis_lightning_detected"].isna().all()
    assert (df_lis["coverage_status"] == "UNOBSERVED").all()
