"""ERA5 Preprocessing Module.

Cleans, standardizes, and processes multidimensional ERA5 NetCDF datasets using xarray.
Performs unit conversions, derives meteorological diagnostic fields, and handles coordinates.
"""

from __future__ import annotations

import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


class ERA5Preprocessor:
    """Preprocessor for ERA5 single levels atmospheric reanalysis NetCDF files."""

    def __init__(self, processed_dir: Path = Path("data/processed/era5_2024")):
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.inventory_records: List[Dict[str, Any]] = []

    @staticmethod
    def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
        """Ensure standard coordinate names: time, latitude, longitude."""
        rename_map = {}
        if "valid_time" in ds.coords and "time" not in ds.coords:
            rename_map["valid_time"] = "time"
        if "lat" in ds.coords and "latitude" not in ds.coords:
            rename_map["lat"] = "latitude"
        if "lon" in ds.coords and "longitude" not in ds.coords:
            rename_map["lon"] = "longitude"

        if rename_map:
            ds = ds.rename(rename_map)

        # Standardize longitude to [-180, 180] if necessary
        if "longitude" in ds.coords:
            lons = ds["longitude"].values
            if (lons > 180).any():
                logger.info("Normalizing longitude coordinate from [0, 360] to [-180, 180].")
                ds = ds.assign_coords(longitude=(((ds["longitude"] + 180) % 360) - 180))
                ds = ds.sortby("longitude")

        # Sort latitude descending
        if "latitude" in ds.coords:
            ds = ds.sortby("latitude", ascending=False)

        return ds

    def process_dataset(self, ds: xr.Dataset, region_name: str) -> xr.Dataset:
        """Perform unit conversions and derive diagnostic meteorological fields."""
        ds = self.standardize_coords(ds)

        # 1. 2m Temperature: K -> °C
        if "t2m" in ds.data_vars:
            ds["t2m_degC"] = ds["t2m"] - 273.15
            ds["t2m_degC"].attrs = {
                "long_name": "2 metre air temperature",
                "units": "°C",
            }

        # 2. 2m Dewpoint Temperature: K -> °C
        if "d2m" in ds.data_vars:
            ds["d2m_degC"] = ds["d2m"] - 273.15
            ds["d2m_degC"].attrs = {
                "long_name": "2 metre dewpoint temperature",
                "units": "°C",
            }

        # 3. Total Precipitation: m -> mm
        if "tp" in ds.data_vars:
            # ERA5 tp is accumulated over step in meters. Convert to mm.
            ds["tp_mm"] = ds["tp"] * 1000.0
            ds["tp_mm"].attrs = {
                "long_name": "Total precipitation",
                "units": "mm",
            }

        # 4. 10m Wind Speed from u10 and v10
        if "u10" in ds.data_vars and "v10" in ds.data_vars:
            ds["wind_speed_10m"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
            ds["wind_speed_10m"].attrs = {
                "long_name": "10 metre horizontal wind speed",
                "units": "m/s",
            }

        # Set attributes for CAPE, CIN, TCWV if present
        if "cape" in ds.data_vars:
            ds["cape"].attrs["units"] = "J/kg"
            ds["cape"].attrs["long_name"] = "Convective available potential energy"
        if "cin" in ds.data_vars:
            ds["cin"].attrs["units"] = "J/kg"
            ds["cin"].attrs["long_name"] = "Convective inhibition"
        if "tcwv" in ds.data_vars:
            ds["tcwv"].attrs["units"] = "kg/m²"
            ds["tcwv"].attrs["long_name"] = "Total column water vapour"

        ds.attrs["region"] = region_name
        return ds

    def load_and_process_region_files(
        self, region_key: str, file_paths: List[Path]
    ) -> xr.Dataset:
        """Load multiple monthly/chunked NetCDF files for a region and combine."""
        if not file_paths:
            raise FileNotFoundError(f"No NetCDF files found for region: {region_key}")

        logger.info(f"Opening {len(file_paths)} NetCDF files for {region_key}...")
        datasets = []
        temp_dirs = []
        for p in sorted(file_paths):
            try:
                if zipfile.is_zipfile(p):
                    td = tempfile.TemporaryDirectory()
                    temp_dirs.append(td)
                    with zipfile.ZipFile(p) as z:
                        z.extractall(td.name)
                    nc_files = [Path(td.name) / f for f in os.listdir(td.name) if f.endswith(".nc")]
                    if nc_files:
                        sub_ds = [xr.open_dataset(f) for f in nc_files]
                        datasets.append(xr.merge(sub_ds))
                else:
                    ds_chunk = xr.open_dataset(p)
                    datasets.append(ds_chunk)
            except Exception as exc:
                logger.error(f"Error opening NetCDF file {p}: {exc}")

        if not datasets:
            raise ValueError(f"Failed to open any valid NetCDF files for {region_key}")

        # Combine along time coordinate
        if len(datasets) == 1:
            combined_ds = datasets[0]
        else:
            combined_ds = xr.concat(datasets, dim="valid_time" if "valid_time" in datasets[0].dims else "time")
            # Drop any duplicate timestamps
            time_dim = "valid_time" if "valid_time" in combined_ds.dims else "time"
            _, index = np.unique(combined_ds[time_dim], return_index=True)
            combined_ds = combined_ds.isel({time_dim: index})

        processed_ds = self.process_dataset(combined_ds, region_key)

        # Save processed NetCDF
        out_nc = self.processed_dir / f"era5_processed_{region_key}_2024.nc"
        processed_ds.to_netcdf(out_nc)
        logger.info(f"Processed dataset saved to {out_nc}")

        return processed_ds
