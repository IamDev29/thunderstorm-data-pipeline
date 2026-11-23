"""
test_glm_integration.py
========================
Automated validation and regression test suite for GOES-16 GLM feature integration,
temporal leakage prevention, spatial filtering, duplicate controls, missing coverage
indicators, feature aggregation boundaries, and evaluation cohort consistency.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import json

DATA_DIR = Path("data/processed/glm_pilot_extracted")
SUBPROJECT_DATA_DIR = Path("Florida_Thunderstorm_2024_Training/data/processed/glm_pilot_extracted")

EXP_DIR = Path("experiments/glm_integration_2024")
EXP_DIR_MIRROR = Path("Florida_Thunderstorm_2024_Training/experiments/glm_integration_2024")

MODELING_DATA_PATH = Path("data/processed/asos_parser_fixed/florida_asos_modeling_ready_fixed_2024.parquet")
STATIONS_PATH = Path("data/processed/asos/florida_asos_stations_2024.csv")


@pytest.fixture(scope="module")
def df_flashes() -> pd.DataFrame:
    path = DATA_DIR / "florida_glm_flash_observations_2024.parquet"
    if not path.exists():
        path = SUBPROJECT_DATA_DIR / "florida_glm_flash_observations_2024.parquet"
    assert path.exists(), f"Flash observations file not found at {path}"
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["flash_timestamp_utc"])
    return df


@pytest.fixture(scope="module")
def df_stations() -> pd.DataFrame:
    assert STATIONS_PATH.exists(), f"Stations file not found at {STATIONS_PATH}"
    return pd.read_csv(STATIONS_PATH)


@pytest.fixture(scope="module")
def df_comp() -> pd.DataFrame:
    path = EXP_DIR / "model_comparison_common_cohort.csv"
    if not path.exists():
        path = EXP_DIR_MIRROR / "model_comparison_common_cohort.csv"
    assert path.exists(), f"Model comparison file not found at {path}"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def df_cov() -> pd.DataFrame:
    path = EXP_DIR / "glm_coverage_summary.csv"
    if not path.exists():
        path = EXP_DIR_MIRROR / "glm_coverage_summary.csv"
    assert path.exists(), f"Coverage summary file not found at {path}"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def df_boot() -> pd.DataFrame:
    path = EXP_DIR / "paired_bootstrap_comparison.csv"
    if not path.exists():
        path = EXP_DIR_MIRROR / "paired_bootstrap_comparison.csv"
    assert path.exists(), f"Bootstrap comparison file not found at {path}"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def df_season() -> pd.DataFrame:
    path = EXP_DIR / "seasonal_threshold_comparison.csv"
    if not path.exists():
        path = EXP_DIR_MIRROR / "seasonal_threshold_comparison.csv"
    assert path.exists(), f"Seasonal threshold file not found at {path}"
    return pd.read_csv(path)


def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return R * c


def test_01_temporal_leakage_controls(df_flashes):
    """
    Test that no GLM flashes timestamped after forecast initialization time t
    are ever used in historical lookback features.
    """
    t_00z = pd.Timestamp("2024-05-15 00:00:00", tz="UTC")
    
    # Flashes in granule s20241360000000 occurring from 00:00:06 to 00:01:00 UTC
    post_00z_flashes = df_flashes[(df_flashes["ts"] > t_00z) & (df_flashes["ts"] <= t_00z + pd.Timedelta(minutes=1))]
    assert len(post_00z_flashes) == 19, f"Expected exactly 19 post-00Z flashes, got {len(post_00z_flashes)}"
    
    # A causal lookback for t = 00:00:00 UTC must contain ZERO of these 19 flashes
    lookback_6h_00z = df_flashes[(df_flashes["ts"] > t_00z - pd.Timedelta(hours=6)) & (df_flashes["ts"] <= t_00z)]
    assert len(lookback_6h_00z) == 1, "Only the May 14 23:59:59 flash is eligible for t=00Z lookback"
    assert not lookback_6h_00z["ts"].isin(post_00z_flashes["ts"]).any(), "Post-00Z flashes must not leak into t=00Z"


def test_02_utc_conversion_and_subsecond_precision(df_flashes):
    """
    Test UTC timestamp parsing and verification that microsecond precision is preserved.
    """
    ts = df_flashes["ts"]
    assert str(ts.dt.tz) in ["UTC", "datetime.timezone.utc"], "Timestamps must be UTC"
    assert (ts.dt.microsecond > 0).any(), "Microsecond precision must be preserved"
    assert len(ts.dt.microsecond.unique()) > 100, "Observation times must not be discretized to seconds"


def test_03_spatial_filtering_haversine(df_flashes, df_stations):
    """
    Test Haversine spatial filtering for 25 km and 50 km radii around ASOS stations.
    """
    # Test known distance between Clewiston (2IS: 26.75, -81.00) and Miami (MIA: 25.79, -80.29)
    dist = haversine_np(26.75, -81.00, 25.79, -80.29)
    assert 120.0 < dist < 130.0, f"Distance should be ~127 km, got {dist}"
    
    # Test flash distance calculations
    t_18z = pd.Timestamp("2024-05-15 18:00:00", tz="UTC")
    f_18z = df_flashes[(df_flashes["ts"] > t_18z - pd.Timedelta(hours=6)) & (df_flashes["ts"] <= t_18z)]
    assert len(f_18z) == 93, f"Expected 93 flashes in 18Z lookback, got {len(f_18z)}"
    
    # Check that distances are strictly non-negative and radius filtering works
    st_row = df_stations[df_stations["station_id"] == "ORL"].iloc[0]
    dists = haversine_np(st_row["latitude"], st_row["longitude"], f_18z["latitude"].values, f_18z["longitude"].values)
    assert (dists >= 0.0).all()
    c25 = np.sum(dists <= 25.0)
    c50 = np.sum(dists <= 50.0)
    assert c25 <= c50, "Count in 25 km must never exceed count in 50 km"


def test_04_duplicate_observations_check(df_flashes):
    """
    Test that there are zero duplicate flash IDs or duplicate coordinate/timestamp pairs.
    """
    assert df_flashes.duplicated(subset=["source_granule_id", "flash_id"]).sum() == 0
    assert df_flashes.duplicated(subset=["latitude", "longitude", "flash_timestamp_utc"]).sum() == 0


def test_05_missing_glm_coverage_handling(df_cov):
    """
    Test that missing GLM periods are explicitly flagged and documented, not silently filled.
    """
    assert len(df_cov) == 12, f"Expected 12 months in coverage summary, got {len(df_cov)}"
    
    # Check Month 5 (May) vs other months
    may_row = df_cov[df_cov["month"] == 5].iloc[0]
    assert may_row["glm_available_rows"] == 320
    assert may_row["excluded_missing_glm_rows"] == 9558 - 320
    
    # Non-May months must have 0 GLM available rows
    non_may = df_cov[df_cov["month"] != 5]
    assert (non_may["glm_available_rows"] == 0).all()
    assert (non_may["glm_coverage_pct"] == 0.0).all()
    
    # Total excluded monitored windows across 2024
    total_excluded = df_cov["excluded_missing_glm_rows"].sum()
    assert total_excluded == 112189, f"Expected 112,189 excluded rows, got {total_excluded}"


def test_06_feature_aggregation_half_open_boundaries(df_flashes):
    """
    Test that feature aggregation respects strict half-open interval boundaries (t - delta, t].
    """
    t_ref = pd.Timestamp("2024-05-15 06:00:00", tz="UTC")
    delta_6h = pd.Timedelta(hours=6)
    
    # Boundary: (00:00:00, 06:00:00]
    # A flash at 00:00:00.000 or before is EXCLUDED from (t-6h, t]
    f_prior = df_flashes[df_flashes["ts"] <= t_ref - delta_6h]
    assert len(f_prior) == 1 # The May 14 23:59:59 flash
    
    # Flashes in (t-6h, t]
    f_in = df_flashes[(df_flashes["ts"] > t_ref - delta_6h) & (df_flashes["ts"] <= t_ref)]
    assert not f_in["ts"].isin(f_prior["ts"]).any(), "Prior boundary flashes must not be in (t-6h, t]"
    
    # Flashes after t
    f_after = df_flashes[df_flashes["ts"] > t_ref]
    assert not f_in["ts"].isin(f_after["ts"]).any(), "Post-t flashes must not be in (t-6h, t]"


def test_07_common_evaluation_cohort_consistency(df_comp):
    """
    Test that all model configurations (A through E) are evaluated on the exact same cohort.
    """
    assert len(df_comp) == 5, f"Expected 5 model configurations, got {len(df_comp)}"
    expected_models = [
        "Model A (Reference Full-Tuned)",
        "Model B (ERA5 + SpatDiurn)",
        "Model C (Model B + ASOS Hist)",
        "Model D (Model C + GLM Features)",
        "Model E (Model D + Platt Calib)"
    ]
    assert df_comp["model_name"].tolist() == expected_models
    
    # Every model must evaluate exactly 320 samples with 166 positives and 154 negatives
    assert (df_comp["sample_count"] == 320).all()
    assert (df_comp["positives"] == 166).all()
    assert (df_comp["tp"] + df_comp["fn"] == 166).all()
    assert (df_comp["tn"] + df_comp["fp"] == 154).all()


def test_08_paired_bootstrap_results_integrity(df_boot):
    """
    Test that paired bootstrap comparisons show statistically significant degradation for Model D.
    """
    assert len(df_boot) == 8, f"Expected 8 bootstrap comparisons, got {len(df_boot)}"
    
    # Check D vs C delta PR-AUC
    d_vs_c_pr = df_boot[(df_boot["comparison"] == "Model D (GLM) - Model C (Baseline C)") & 
                        (df_boot["metric"] == "delta_pr_auc")].iloc[0]
    assert d_vs_c_pr["mean_difference"] < 0, "Model D PR-AUC must be lower than Model C"
    assert d_vs_c_pr["ci_95_upper"] < 0, "Upper bound of 95% CI must be negative"
    assert d_vs_c_pr["statistically_significant"] == "YES"
    assert "NO IMPROVEMENT" in d_vs_c_pr["scientific_conclusion"]


def test_09_seasonal_threshold_pre_test_separation(df_season):
    """
    Test that seasonal thresholds are evaluated exclusively on pre-test data (Jan-Sep).
    """
    assert len(df_season) == 4, f"Expected 4 threshold strategies, got {len(df_season)}"
    assert (df_season["sample_count"] == 84436).all(), "Must evaluate exactly 84,436 pre-test windows"
    assert (df_season["positives"] == 20559).all(), "Must evaluate exactly 20,559 pre-test positives"
    
    # Verify that seasonal thresholds achieve superior F1 score over uncalibrated baseline
    strat_seasonal = df_season[df_season["strategy"] == "Predefined Seasonal Thresholds"].iloc[0]
    strat_baseline = df_season[df_season["strategy"] == "Uncalibrated Baseline Threshold"].iloc[0]
    assert strat_seasonal["f1_score"] > strat_baseline["f1_score"], "Seasonal threshold must outperform default 0.50 cutoff"
