"""Unit tests for ERA5 atmospheric processing, coordinates, and leap year handling."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.process_era5 import ERA5Processor


def test_era5_derived_features_and_unit_conversions():
    """Verify conversion of K to C, wind vector magnitude, and Magnus RH formula."""
    times = pd.date_range("2024-05-15 00:00", periods=2, freq="6h")
    lats = [30.0, 29.0]
    lons = [-82.0, -81.0]

    ds = xr.Dataset(
        data_vars={
            "t2m": (("time", "latitude", "longitude"), np.array([[[300.15, 295.15], [302.15, 298.15]], [[301.15, 296.15], [303.15, 299.15]]])),
            "d2m": (("time", "latitude", "longitude"), np.array([[[295.15, 290.15], [297.15, 293.15]], [[296.15, 291.15], [298.15, 294.15]]])),
            "u10": (("time", "latitude", "longitude"), np.array([[[3.0, 4.0], [0.0, -3.0]], [[2.0, 3.0], [1.0, -2.0]]])),
            "v10": (("time", "latitude", "longitude"), np.array([[[4.0, 3.0], [5.0, 4.0]], [[3.0, 2.0], [4.0, 3.0]]])),
            "tp": (("time", "latitude", "longitude"), np.array([[[0.005, 0.0], [0.012, 0.002]], [[0.001, 0.0], [0.008, 0.003]]])),
        },
        coords={"time": times, "latitude": lats, "longitude": lons},
    )

    ds_proc = ERA5Processor.derive_meteorological_fields(ds)

    # 1. Check Temperature (300.15 K = 27.0 °C)
    assert np.isclose(ds_proc["temperature_2m_degC"].values[0, 0, 0], 27.0)

    # 2. Check Wind speed (sqrt(3^2 + 4^2) = 5.0 m/s)
    assert np.isclose(ds_proc["wind_speed_10m_ms"].values[0, 0, 0], 5.0)

    # 3. Check Precip (0.005 m = 5.0 mm)
    assert np.isclose(ds_proc["total_precipitation_mm"].values[0, 0, 0], 5.0)

    # 4. Check Relative Humidity (bounded 0 - 100%)
    rh_vals = ds_proc["relative_humidity_2m_pct"].values
    assert (rh_vals >= 0.0).all() and (rh_vals <= 100.0).all()


def test_era5_coordinates_and_leap_year_handling():
    """Verify coordinate standardization and leap day 2024-02-29 compatibility."""
    # Test leap year date range contains Feb 29
    leap_dates = pd.date_range("2024-02-28 00:00", "2024-03-01 00:00", freq="1h")
    assert "2024-02-29 12:00:00" in [str(t) for t in leap_dates]
    assert len(leap_dates) == 49

    # Test coordinate sorting and renaming
    ds = xr.Dataset(
        data_vars={"t2m": (("valid_time", "lat", "lon"), np.ones((2, 2, 2)))},
        coords={
            "valid_time": pd.date_range("2024-02-29 00:00", periods=2, freq="12h"),
            "lat": [25.0, 30.0],  # Ascending, needs sorting descending
            "lon": [278.0, 280.0],  # 0-360 deg, needs wrapping to -180 to 180
        },
    )
    ds_std = ERA5Processor.standardize_coordinates(ds)
    assert "time" in ds_std.coords
    assert "latitude" in ds_std.coords
    assert "longitude" in ds_std.coords
    assert ds_std["latitude"].values[0] == 30.0  # Descending
    assert ds_std["longitude"].values[0] < 0.0   # Wrapped to [-180, 180]


def test_actual_downloaded_february_leap_year_contents():
    """Verify actual downloaded February 2024 file contains leap day 2024-02-29."""
    from pathlib import Path
    import zipfile
    feb_file = Path("data/raw/era5/single_levels/era5_florida_2024_02.nc")
    if not feb_file.exists():
        pytest.skip("February file not yet downloaded")

    datasets = []
    if zipfile.is_zipfile(feb_file):
        with zipfile.ZipFile(feb_file) as z:
            import tempfile
            t_dir = Path(tempfile.mkdtemp(prefix="test_feb_"))
            z.extractall(t_dir)
            for nc_p in t_dir.glob("*.nc"):
                datasets.append(xr.open_dataset(nc_p))
    else:
        datasets.append(xr.open_dataset(feb_file))

    for ds in datasets:
        t_var = "valid_time" if "valid_time" in ds.coords else "time"
        t_strings = [str(t) for t in pd.to_datetime(ds[t_var].values)]
        assert any("2024-02-29" in s for s in t_strings), "February 29 leap day must be present"


def test_actual_downloaded_full_year_single_levels():
    """Verify downloaded full-year 2024 single-level NetCDF dataset integrity."""
    from pathlib import Path
    candidate_paths = [
        Path("Florida_Thunderstorm_2024_Training/data/raw/era5/full_year/era5_florida_2024_single_levels.nc"),
        Path("data/raw/era5/full_year/era5_florida_2024_single_levels.nc"),
    ]
    target_file = next((p for p in candidate_paths if p.exists()), None)
    if not target_file:
        pytest.skip("Full-year single-levels file not found")

    ds = xr.open_dataset(target_file, engine="netcdf4")
    t_var = "time" if "time" in ds.coords else "valid_time"
    assert t_var in ds.coords

    times = pd.to_datetime(ds[t_var].values)
    assert len(times) == 1464, f"Expected 1464 timestamps, found {len(times)}"
    assert times.nunique() == 1464, "Duplicate timestamps detected"
    assert str(times.min()) == "2024-01-01 00:00:00"
    assert str(times.max()) == "2024-12-31 18:00:00"

    # Check Feb 29
    feb29 = times[(times.month == 2) & (times.day == 29)]
    assert len(feb29) == 4, "February 29 must contain all 4 6-hourly timestamps"

    # Check 12 variables
    expected_vars = ["t2m", "d2m", "sp", "msl", "tp", "cp", "cape", "cin", "tcc", "u10", "v10", "tcwv"]
    for v in expected_vars:
        assert v in ds.data_vars, f"Variable {v} missing in full-year NetCDF"


