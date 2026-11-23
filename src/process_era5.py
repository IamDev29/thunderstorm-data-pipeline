"""Process, standardize, and derive meteorological features from ERA5 NetCDF datasets.

Supports both single-level (surface) and pressure-level (850, 500, 300 hPa) datasets.
Converts units (K -> °C, Pa -> hPa, m -> mm, kg/kg -> g/kg), computes derived
wind speed, wind direction, and 2m relative humidity via Magnus formula, applies the
authoritative Florida polygon boundary mask, and generates:
data/processed/era5_florida_2024.csv
"""

from __future__ import annotations

import json
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shapely.geometry
import xarray as xr

logger = logging.getLogger(__name__)

SYNOPTIC_HOURS = [0, 6, 12, 18]


class ERA5Processor:
    """Processes multidimensional ERA5 NetCDF files for Florida."""

    def __init__(
        self,
        raw_single_dir: Path = Path("data/raw/era5/single_levels"),
        raw_pressure_dir: Path = Path("data/raw/era5/pressure_levels"),
        boundary_file: Path = Path("data/raw/geographic/florida_boundary.geojson"),
        interim_dir: Path = Path("data/interim/era5"),
        processed_dir: Path = Path("data/processed"),
        synoptic_only: bool = True,
    ):
        self.raw_single_dir = raw_single_dir
        self.raw_pressure_dir = raw_pressure_dir
        self.boundary_file = boundary_file
        self.interim_dir = interim_dir
        self.processed_dir = processed_dir
        self.synoptic_only = synoptic_only

        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.florida_polygon = self._load_florida_boundary()

    def _load_florida_boundary(self) -> Optional[shapely.geometry.base.BaseGeometry]:
        """Load Shapely geometry for Florida state boundary."""
        if not self.boundary_file.exists():
            return None
        try:
            with open(self.boundary_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            geom = data["features"][0]["geometry"] if "features" in data else data
            return shapely.geometry.shape(geom)
        except Exception as exc:
            logger.error(f"Error loading boundary: {exc}")
            return None

    @staticmethod
    def standardize_coordinates(ds: xr.Dataset) -> xr.Dataset:
        """Standardize coordinates: valid_time -> time, normalize longitudes, descending latitude."""
        rename_map = {}
        if "valid_time" in ds.coords and "time" not in ds.coords:
            rename_map["valid_time"] = "time"
        if "lat" in ds.coords and "latitude" not in ds.coords:
            rename_map["lat"] = "latitude"
        if "lon" in ds.coords and "longitude" not in ds.coords:
            rename_map["lon"] = "longitude"
        if "pressure_level" in ds.coords and "level" not in ds.coords:
            rename_map["pressure_level"] = "level"
        if "isobaricInhPa" in ds.coords and "level" not in ds.coords:
            rename_map["isobaricInhPa"] = "level"

        if rename_map:
            ds = ds.rename(rename_map)

        # Normalize longitude to [-180, 180]
        if "longitude" in ds.coords and (ds["longitude"].values > 180).any():
            ds = ds.assign_coords(longitude=(((ds["longitude"] + 180) % 360) - 180))
            ds = ds.sortby("longitude")

        # Sort latitude descending
        if "latitude" in ds.coords:
            ds = ds.sortby("latitude", ascending=False)

        return ds

    @staticmethod
    def derive_meteorological_fields(ds: xr.Dataset) -> xr.Dataset:
        """Derive standard units and diagnostic meteorological features for single levels."""
        # 1. 2m Temperature (K -> °C)
        if "t2m" in ds.data_vars:
            ds["temperature_2m_degC"] = ds["t2m"] - 273.15
            ds["temperature_2m_degC"].attrs = {"units": "°C", "long_name": "2 metre air temperature"}

        # 2. 2m Dewpoint Temperature (K -> °C)
        if "d2m" in ds.data_vars:
            ds["dewpoint_2m_degC"] = ds["d2m"] - 273.15
            ds["dewpoint_2m_degC"].attrs = {"units": "°C", "long_name": "2 metre dewpoint temperature"}

        # 3. Surface pressure & MSLP (Pa -> hPa)
        if "sp" in ds.data_vars:
            ds["surface_pressure_hPa"] = ds["sp"] / 100.0
            ds["surface_pressure_hPa"].attrs = {"units": "hPa"}
        if "msl" in ds.data_vars:
            ds["mslp_hPa"] = ds["msl"] / 100.0
            ds["mslp_hPa"].attrs = {"units": "hPa"}

        # 4. Total and Convective Precipitation (m -> mm)
        if "tp" in ds.data_vars:
            ds["total_precipitation_mm"] = ds["tp"] * 1000.0
            ds["total_precipitation_mm"].attrs = {"units": "mm"}
        if "cp" in ds.data_vars:
            ds["convective_precipitation_mm"] = ds["cp"] * 1000.0
            ds["convective_precipitation_mm"].attrs = {"units": "mm"}

        # 5. Wind components and derived 10m wind speed & direction
        if "u10" in ds.data_vars and "v10" in ds.data_vars:
            ds["u_wind_10m_ms"] = ds["u10"]
            ds["v_wind_10m_ms"] = ds["v10"]
            ds["wind_speed_10m_ms"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
            ds["wind_speed_10m_ms"].attrs = {"units": "m/s", "long_name": "10 metre horizontal wind speed"}
            # Meteorological wind direction: direction FROM which wind blows
            wind_dir = (180.0 / np.pi * np.arctan2(-ds["u10"], -ds["v10"])) % 360.0
            ds["wind_direction_10m_deg"] = wind_dir
            ds["wind_direction_10m_deg"].attrs = {"units": "degrees", "long_name": "10 metre wind direction"}

        # 6. 2m Relative Humidity using Magnus formula
        if "t2m" in ds.data_vars and "d2m" in ds.data_vars:
            t_c = ds["temperature_2m_degC"]
            td_c = ds["dewpoint_2m_degC"]
            e_s = 6.112 * np.exp((17.625 * t_c) / (243.04 + t_c))
            e = 6.112 * np.exp((17.625 * td_c) / (243.04 + td_c))
            rh = (e / e_s) * 100.0
            rh = xr.where(rh > 100.0, 100.0, rh)
            rh = xr.where(rh < 0.0, 0.0, rh)
            ds["relative_humidity_2m_pct"] = rh
            ds["relative_humidity_2m_pct"].attrs = {"units": "%", "long_name": "2 metre relative humidity"}

        # 7. Standardize CAPE, CIN, TCWV, TCC
        if "cape" in ds.data_vars:
            ds["cape_J_kg"] = ds["cape"]
            ds["cape_J_kg"].attrs = {"units": "J/kg"}
        if "cin" in ds.data_vars:
            ds["cin_J_kg"] = ds["cin"]
            ds["cin_J_kg"].attrs = {"units": "J/kg"}
        if "tcwv" in ds.data_vars:
            ds["tcwv_kg_m2"] = ds["tcwv"]
            ds["tcwv_kg_m2"].attrs = {"units": "kg/m²"}
        if "tcc" in ds.data_vars:
            ds["cloud_cover_fraction"] = ds["tcc"]
            ds["cloud_cover_fraction"].attrs = {"units": "0-1"}

        return ds

    def _open_nc_or_zip(self, f: Path) -> List[xr.Dataset]:
        """Open a NetCDF file or unpack a zipped CDS bundle and return datasets."""
        if zipfile.is_zipfile(f):
            unpacked_dir = self.interim_dir / "unpacked" / f.stem
            unpacked_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(f, "r") as z:
                z.extractall(unpacked_dir)
            instant_p = unpacked_dir / "data_stream-oper_stepType-instant.nc"
            accum_p = unpacked_dir / "data_stream-oper_stepType-accum.nc"
            if instant_p.exists() and accum_p.exists():
                ds_inst = xr.open_dataset(instant_p, engine="netcdf4")
                ds_acc = xr.open_dataset(accum_p, engine="netcdf4")
                return [xr.merge([ds_inst, ds_acc])]
            else:
                return [xr.open_dataset(nc_f, engine="netcdf4") for nc_f in unpacked_dir.glob("*.nc")]
        else:
            return [xr.open_dataset(f, engine="netcdf4")]

    def _process_single_levels(self) -> pd.DataFrame:
        """Process and extract single-level ERA5 datasets."""
        full_year_candidates = [
            Path("data/raw/era5/full_year/era5_florida_2024_single_levels.nc"),
            self.raw_single_dir.parent / "full_year" / "era5_florida_2024_single_levels.nc",
            self.raw_single_dir / "era5_florida_2024_single_levels.nc",
        ]
        nc_files = []
        for cand in full_year_candidates:
            if cand.exists() and cand.stat().st_size > 1024:
                logger.info(f"Using full-year single-level NetCDF dataset: {cand}")
                nc_files = [cand]
                break

        if not nc_files:
            nc_files = sorted(list(self.raw_single_dir.glob("*.nc")))
            # Filter out partial files
            nc_files = [f for f in nc_files if not f.name.endswith(".part")]

        if not nc_files:
            fallback = Path("../data/raw/era5_2024/florida/era5_florida_2024_05.nc")
            if fallback.exists():
                nc_files = [fallback]
            else:
                raise FileNotFoundError(f"No valid ERA5 single-level files found in {self.raw_single_dir}")

        logger.info(f"Loading {len(nc_files)} single-level NetCDF / zip bundle(s)...")
        datasets = []
        for f in nc_files:
            datasets.extend(self._open_nc_or_zip(f))

        processed_dfs = []
        for ds_raw in datasets:
            ds = self.standardize_coordinates(ds_raw)

            # Filter to synoptic hours if requested
            time_dim = "time" if "time" in ds.coords else "valid_time"
            if self.synoptic_only and time_dim in ds.coords:
                t_hours = pd.to_datetime(ds[time_dim].values).hour
                synoptic_mask = np.isin(t_hours, SYNOPTIC_HOURS)
                if synoptic_mask.any():
                    ds = ds.isel({time_dim: synoptic_mask})

            ds = self.derive_meteorological_fields(ds)
            df = ds.to_dataframe().reset_index()
            t_col = "time" if "time" in df.columns else "valid_time"
            df["timestamp_utc"] = pd.to_datetime(df[t_col])
            processed_dfs.append(df)

        if not processed_dfs:
            return pd.DataFrame()

        df_single = pd.concat(processed_dfs, ignore_index=True)
        # Drop duplicates on timestamp, latitude, longitude
        df_single = df_single.drop_duplicates(subset=["timestamp_utc", "latitude", "longitude"])
        return df_single

    def _process_pressure_levels(self) -> pd.DataFrame:
        """Process and extract pressure-level ERA5 datasets (850, 500, 300 hPa)."""
        full_year_candidates = [
            Path("data/raw/era5/full_year/era5_florida_2024_pressure_levels.nc"),
            self.raw_pressure_dir.parent / "full_year" / "era5_florida_2024_pressure_levels.nc",
            self.raw_pressure_dir / "era5_florida_2024_pressure_levels.nc",
        ]
        nc_files = []
        for cand in full_year_candidates:
            if cand.exists() and cand.stat().st_size > 1024:
                logger.info(f"Using full-year pressure-level NetCDF dataset: {cand}")
                nc_files = [cand]
                break

        if not nc_files:
            nc_files = sorted(list(self.raw_pressure_dir.glob("*.nc")))
            nc_files = [f for f in nc_files if not f.name.endswith(".part")]

        if not nc_files:
            logger.info("No pressure-level files found in raw directory. Proceeding without pressure levels.")
            return pd.DataFrame()

        logger.info(f"Loading {len(nc_files)} pressure-level NetCDF / zip bundle(s)...")
        datasets = []
        for f in nc_files:
            datasets.extend(self._open_nc_or_zip(f))

        processed_dfs = []
        for ds_raw in datasets:
            ds = self.standardize_coordinates(ds_raw)
            time_dim = "time" if "time" in ds.coords else "valid_time"
            if self.synoptic_only and time_dim in ds.coords:
                t_hours = pd.to_datetime(ds[time_dim].values).hour
                synoptic_mask = np.isin(t_hours, SYNOPTIC_HOURS)
                if synoptic_mask.any():
                    ds = ds.isel({time_dim: synoptic_mask})

            # Check level coordinate
            lvl_coord = "level" if "level" in ds.coords else None
            if not lvl_coord:
                continue

            df_p = ds.to_dataframe().reset_index()
            t_col = "time" if "time" in df_p.columns else "valid_time"
            df_p["timestamp_utc"] = pd.to_datetime(df_p[t_col])

            # Pivot levels to wide columns
            # Levels of interest: 850, 500, 300
            for lvl in [850, 500, 300]:
                sub = df_p[df_p[lvl_coord] == lvl].copy()
                if sub.empty:
                    continue

                lvl_prefix = f"_{lvl}hPa"
                rename_map = {}
                if "t" in sub.columns:
                    sub[f"temperature{lvl_prefix}_degC"] = sub["t"] - 273.15
                if "r" in sub.columns:
                    sub[f"relative_humidity{lvl_prefix}_pct"] = sub["r"]
                if "q" in sub.columns:
                    sub[f"specific_humidity{lvl_prefix}_g_kg"] = sub["q"] * 1000.0
                if "u" in sub.columns:
                    sub[f"u_wind{lvl_prefix}_ms"] = sub["u"]
                if "v" in sub.columns:
                    sub[f"v_wind{lvl_prefix}_ms"] = sub["v"]
                if "z" in sub.columns:
                    sub[f"geopotential{lvl_prefix}_m2_s2"] = sub["z"]
                if "w" in sub.columns:
                    sub[f"vertical_velocity{lvl_prefix}_Pa_s"] = sub["w"]

                cols_to_keep = ["timestamp_utc", "latitude", "longitude"] + [
                    c for c in sub.columns if f"{lvl}hPa" in c
                ]
                processed_dfs.append(sub[cols_to_keep])

        if not processed_dfs:
            return pd.DataFrame()

        # Merge wide level dataframes
        df_p_all = processed_dfs[0]
        for sub in processed_dfs[1:]:
            df_p_all = pd.merge(df_p_all, sub, on=["timestamp_utc", "latitude", "longitude"], how="outer")

        df_p_all = df_p_all.drop_duplicates(subset=["timestamp_utc", "latitude", "longitude"])
        return df_p_all

    def process_to_dataframe(self) -> pd.DataFrame:
        """Process NetCDF files, apply Florida boundary mask, and export CSV."""
        df_single = self._process_single_levels()
        df_pressure = self._process_pressure_levels()

        if not df_pressure.empty:
            df_merged = pd.merge(
                df_single,
                df_pressure,
                on=["timestamp_utc", "latitude", "longitude"],
                how="left",
            )
        else:
            df_merged = df_single

        # Apply Florida polygon mask
        if self.florida_polygon is not None:
            lats = df_merged["latitude"].values
            lons = df_merged["longitude"].values
            unique_coords = {(lat, lon) for lat, lon in zip(lats, lons)}
            in_fl_map = {}
            for lat, lon in unique_coords:
                pt = shapely.geometry.Point(lon, lat)
                in_fl_map[(lat, lon)] = self.florida_polygon.contains(pt)

            mask = [in_fl_map[(lat, lon)] for lat, lon in zip(lats, lons)]
            df_fl = df_merged[mask].copy()
            logger.info(
                f"Florida polygon mask reduced grid from {len(df_merged):,} to {len(df_fl):,} grid points "
                f"({len(unique_coords)} unique cells -> {sum(in_fl_map.values())} inside Florida)."
            )
        else:
            df_fl = df_merged.copy()

        # Grid cell ID
        df_fl["grid_cell_id"] = (
            "FL_" + df_fl["latitude"].round(2).astype(str) + "_" + df_fl["longitude"].round(2).astype(str)
        )

        desired_cols = [
            "timestamp_utc",
            "latitude",
            "longitude",
            "grid_cell_id",
            "temperature_2m_degC",
            "dewpoint_2m_degC",
            "relative_humidity_2m_pct",
            "surface_pressure_hPa",
            "mslp_hPa",
            "total_precipitation_mm",
            "convective_precipitation_mm",
            "cape_J_kg",
            "cin_J_kg",
            "cloud_cover_fraction",
            "u_wind_10m_ms",
            "v_wind_10m_ms",
            "wind_speed_10m_ms",
            "wind_direction_10m_deg",
            "tcwv_kg_m2",
            # Pressure levels
            "temperature_850hPa_degC",
            "temperature_500hPa_degC",
            "temperature_300hPa_degC",
            "relative_humidity_850hPa_pct",
            "relative_humidity_500hPa_pct",
            "relative_humidity_300hPa_pct",
            "specific_humidity_850hPa_g_kg",
            "specific_humidity_500hPa_g_kg",
            "specific_humidity_300hPa_g_kg",
            "u_wind_850hPa_ms",
            "u_wind_500hPa_ms",
            "u_wind_300hPa_ms",
            "v_wind_850hPa_ms",
            "v_wind_500hPa_ms",
            "v_wind_300hPa_ms",
            "geopotential_850hPa_m2_s2",
            "geopotential_500hPa_m2_s2",
            "geopotential_300hPa_m2_s2",
            "vertical_velocity_850hPa_Pa_s",
            "vertical_velocity_500hPa_Pa_s",
            "vertical_velocity_300hPa_Pa_s",
        ]
        available_cols = [c for c in desired_cols if c in df_fl.columns]
        df_out = df_fl[available_cols].copy()

        # Sort chronologically and spatially
        df_out = df_out.sort_values(by=["timestamp_utc", "latitude", "longitude"]).reset_index(drop=True)

        out_csv = self.processed_dir / "era5_florida_2024.csv"
        df_out.to_csv(out_csv, index=False)
        logger.info(f"Processed ERA5 Florida CSV saved to {out_csv} ({len(df_out):,} rows, {len(df_out.columns)} cols)")

        return df_out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    processor = ERA5Processor()
    df_res = processor.process_to_dataframe()
    print("ERA5 processed rows:", len(df_res))
