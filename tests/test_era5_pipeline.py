"""Unit tests for ERA5 Climate Similarity Pipeline."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.era5_analysis import ERA5Analyzer, SEASONS
from src.era5_preprocessing import ERA5Preprocessor


@pytest.fixture
def synthetic_era5_dataset():
    """Create a synthetic multidimensional ERA5 dataset for testing."""
    times = pd.date_range("2024-01-01", "2024-12-31", freq="6h")
    lats = np.array([23.0, 22.0, 21.0, 20.0])
    lons = np.array([85.0, 86.0, 87.0, 88.0])

    nt, ny, nx = len(times), len(lats), len(lons)

    # Base temperature with seasonal cycle (warmer in pre-monsoon/monsoon)
    day_of_year = times.dayofyear.values[:, None, None]
    t2m_k = 295.0 + 10.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.normal(0, 1.5, (nt, ny, nx))
    d2m_k = t2m_k - 4.0 + np.random.normal(0, 1.0, (nt, ny, nx))
    cape = np.maximum(0, 1500.0 * np.sin(2 * np.pi * (day_of_year - 60) / 365) + np.random.normal(500, 300, (nt, ny, nx)))
    cin = np.maximum(0, 100.0 - 50.0 * np.sin(2 * np.pi * (day_of_year - 60) / 365) + np.random.normal(20, 10, (nt, ny, nx)))
    tcwv = 30.0 + 25.0 * np.sin(2 * np.pi * (day_of_year - 120) / 365) + np.random.normal(0, 3, (nt, ny, nx))
    tp_m = np.maximum(0, 0.005 * np.sin(2 * np.pi * (day_of_year - 130) / 365) + np.random.normal(0.001, 0.002, (nt, ny, nx)))
    u10 = np.random.normal(2.0, 3.0, (nt, ny, nx))
    v10 = np.random.normal(1.0, 3.0, (nt, ny, nx))

    ds = xr.Dataset(
        data_vars={
            "t2m": (["valid_time", "latitude", "longitude"], t2m_k),
            "d2m": (["valid_time", "latitude", "longitude"], d2m_k),
            "cape": (["valid_time", "latitude", "longitude"], cape),
            "cin": (["valid_time", "latitude", "longitude"], cin),
            "tcwv": (["valid_time", "latitude", "longitude"], tcwv),
            "tp": (["valid_time", "latitude", "longitude"], tp_m),
            "u10": (["valid_time", "latitude", "longitude"], u10),
            "v10": (["valid_time", "latitude", "longitude"], v10),
        },
        coords={
            "valid_time": times,
            "latitude": lats,
            "longitude": lons,
        },
    )
    return ds


def test_preprocessing_coordinate_standardization_and_unit_conversions(synthetic_era5_dataset):
    preprocessor = ERA5Preprocessor()
    processed_ds = preprocessor.process_dataset(synthetic_era5_dataset, "test_region")

    # Time coord should be renamed to 'time'
    assert "time" in processed_ds.coords

    # Derived temperature in Celsius
    assert "t2m_degC" in processed_ds.data_vars
    assert processed_ds["t2m_degC"].attrs["units"] == "°C"
    sample_k = synthetic_era5_dataset["t2m"].values[0, 0, 0]
    sample_c = processed_ds["t2m_degC"].values[0, 0, 0]
    assert np.isclose(sample_c, sample_k - 273.15)

    # Derived dewpoint in Celsius
    assert "d2m_degC" in processed_ds.data_vars
    assert processed_ds["d2m_degC"].attrs["units"] == "°C"

    # Derived precipitation in mm
    assert "tp_mm" in processed_ds.data_vars
    assert processed_ds["tp_mm"].attrs["units"] == "mm"
    sample_tp_m = synthetic_era5_dataset["tp"].values[0, 0, 0]
    sample_tp_mm = processed_ds["tp_mm"].values[0, 0, 0]
    assert np.isclose(sample_tp_mm, sample_tp_m * 1000.0)

    # Derived wind speed
    assert "wind_speed_10m" in processed_ds.data_vars
    sample_u = synthetic_era5_dataset["u10"].values[0, 0, 0]
    sample_v = synthetic_era5_dataset["v10"].values[0, 0, 0]
    expected_ws = np.sqrt(sample_u**2 + sample_v**2)
    assert np.isclose(processed_ds["wind_speed_10m"].values[0, 0, 0], expected_ws)


def test_longitude_wrapping_for_us_regions():
    # Test longitude normalization from [0, 360] to [-180, 180]
    lons_360 = np.array([262.0, 265.0, 270.0]) # Equivalent to -98, -95, -90 in Texas/Gulf
    ds = xr.Dataset(
        data_vars={"t2m": (["latitude", "longitude"], np.ones((2, 3)))},
        coords={"latitude": [28.0, 27.0], "longitude": lons_360},
    )
    standardized = ERA5Preprocessor.standardize_coords(ds)
    assert standardized["longitude"].values[0] == -98.0
    assert standardized["longitude"].values[1] == -95.0
    assert standardized["longitude"].values[2] == -90.0


def test_seasonal_masking_and_stats(synthetic_era5_dataset):
    preprocessor = ERA5Preprocessor()
    processed_ds = preprocessor.process_dataset(synthetic_era5_dataset, "chandipur")

    analyzer = ERA5Analyzer({"chandipur": processed_ds})
    stats_df = analyzer.compute_all_regional_seasonal_stats()

    assert not stats_df.empty
    assert "season" in stats_df.columns
    seasons_present = set(stats_df["season"].unique())
    assert seasons_present == {"pre_monsoon", "monsoon", "post_monsoon", "annual"}

    # Verify pre-monsoon row
    mam_t2m = stats_df[(stats_df["season"] == "pre_monsoon") & (stats_df["variable"] == "t2m_degC")]
    assert len(mam_t2m) == 1
    row = mam_t2m.iloc[0]
    assert row["valid_count"] > 0
    assert row["missing_pct"] == 0.0
    assert not np.isnan(row["mean"])


def test_climate_similarity_scoring_synthetic():
    preprocessor = ERA5Preprocessor()

    # Create Chandipur reference dataset
    times = pd.date_range("2024-01-01", "2024-12-31", freq="6h")
    lats = [22.0, 21.0]
    lons = [86.0, 87.0]
    nt, ny, nx = len(times), 2, 2

    # Baseline Chandipur
    chandi_ds = xr.Dataset(
        data_vars={
            "t2m": (["time", "latitude", "longitude"], np.full((nt, ny, nx), 302.0)),
            "d2m": (["time", "latitude", "longitude"], np.full((nt, ny, nx), 297.0)),
            "cape": (["time", "latitude", "longitude"], np.full((nt, ny, nx), 2000.0)),
            "cin": (["time", "latitude", "longitude"], np.full((nt, ny, nx), 30.0)),
            "tcwv": (["time", "latitude", "longitude"], np.full((nt, ny, nx), 50.0)),
            "tp": (["time", "latitude", "longitude"], np.full((nt, ny, nx), 0.005)),
            "u10": (["time", "latitude", "longitude"], np.full((nt, ny, nx), 3.0)),
            "v10": (["time", "latitude", "longitude"], np.full((nt, ny, nx), 2.0)),
        },
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    p_chandi = preprocessor.process_dataset(chandi_ds, "chandipur")

    # Region A (Very close to Chandipur: slight shift)
    reg_a = chandi_ds.copy(deep=True)
    reg_a["cape"] = reg_a["cape"] + 100.0
    p_reg_a = preprocessor.process_dataset(reg_a, "region_a")

    # Region B (Far from Chandipur: large shift)
    reg_b = chandi_ds.copy(deep=True)
    reg_b["cape"] = reg_b["cape"] + 1500.0
    reg_b["tcwv"] = reg_b["tcwv"] - 30.0
    p_reg_b = preprocessor.process_dataset(reg_b, "region_b")

    datasets = {"chandipur": p_chandi, "region_a": p_reg_a, "region_b": p_reg_b}
    analyzer = ERA5Analyzer(datasets)

    sim_df, rankings = analyzer.compute_climate_similarity(ref_region="chandipur")

    assert not sim_df.empty
    # Region A must have a smaller distance score than Region B (meaning closer similarity!)
    a_score = sim_df[(sim_df["region"] == "region_a") & (sim_df["season"] == "annual")]["composite_distance_score"].values[0]
    b_score = sim_df[(sim_df["region"] == "region_b") & (sim_df["season"] == "annual")]["composite_distance_score"].values[0]

    assert a_score < b_score
    # In ranking, region_a should rank 1st
    assert rankings["equal_weights"][0][0] == "region_a"
