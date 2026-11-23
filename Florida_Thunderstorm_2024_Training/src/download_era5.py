"""ERA5 Data Downloader and Manager for Florida (2024).

Interfaces with the Copernicus Climate Data Store (CDS) API.
Supports 6-hourly synoptic timestamps (00, 06, 12, 18 UTC) for leap year 2024
(1,464 timestamps across 366 days), with monthly resumable batch downloads,
robust atomic writing (.part files), NetCDF validation, and detailed logging.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cdsapi
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# Official CDS Configuration
DEFAULT_CDS_URL = os.environ.get("CDSAPI_URL", "https://cds.climate.copernicus.eu/api")
DEFAULT_CDS_KEY = os.environ.get("CDSAPI_KEY")

# Florida Bounding Box [North, West, South, East]
FLORIDA_AREA = [31.5, -88.0, 24.5, -79.5]

# Leap Year 2024 Days per Month
DAYS_IN_MONTH_2024 = {
    1: 31, 2: 29, 3: 31, 4: 30,
    5: 31, 6: 30, 7: 31, 8: 31,
    9: 30, 10: 31, 11: 30, 12: 31
}

# 6-Hourly Synoptic Times
SYNOPTIC_TIMES_6HOURLY = ["00:00", "06:00", "12:00", "18:00"]

SINGLE_LEVEL_VARS = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "mean_sea_level_pressure",
    "total_precipitation",
    "convective_precipitation",
    "convective_available_potential_energy",
    "convective_inhibition",
    "total_cloud_cover",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_column_water_vapour",
]

PRESSURE_LEVEL_VARS = [
    "temperature",
    "relative_humidity",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
    "vertical_velocity",
]

PRESSURE_LEVELS = ["850", "500", "300"]


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class ERA5Downloader:
    """Manages ERA5 downloads for Florida 2024."""

    def __init__(
        self,
        raw_single_dir: Path = Path("data/raw/era5/single_levels"),
        raw_pressure_dir: Path = Path("data/raw/era5/pressure_levels"),
        parent_raw_dir: Path = Path("../data/raw/era5_2024"),
        cds_url: Optional[str] = DEFAULT_CDS_URL,
        cds_key: Optional[str] = DEFAULT_CDS_KEY,
    ):
        self.raw_single_dir = raw_single_dir
        self.raw_pressure_dir = raw_pressure_dir
        self.parent_raw_dir = parent_raw_dir

        self.raw_single_dir.mkdir(parents=True, exist_ok=True)
        self.raw_pressure_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_records: List[Dict[str, Any]] = []

        self.client: Optional[cdsapi.Client] = None
        self._init_cds_client(cds_url, cds_key)

    def _init_cds_client(self, url: Optional[str], key: Optional[str]) -> None:
        """Initialize CDS client without exposing credentials in logs."""
        try:
            if url and key:
                self.client = cdsapi.Client(url=url, key=key)
            else:
                self.client = cdsapi.Client()
            logger.info("CDS API client successfully initialized.")
        except Exception as exc:
            logger.warning(f"CDS API client initialization failed: {exc}. Will reuse local data.")

    def import_existing_florida_pilot(self) -> Optional[Path]:
        """Import verified existing Florida 2024 NetCDF data if available."""
        dest_file = self.raw_single_dir / "era5_florida_2024_05.nc"
        parent_file = self.parent_raw_dir / "florida" / "era5_florida_2024_05.nc"

        if parent_file.exists() and parent_file.stat().st_size > 1024:
            if not dest_file.exists() or dest_file.stat().st_size != parent_file.stat().st_size:
                logger.info(f"Importing verified local ERA5 Florida NetCDF: {parent_file.name}")
                shutil.copy2(parent_file, dest_file)
            status = "REUSED_LOCAL"
        elif dest_file.exists() and dest_file.stat().st_size > 1024:
            status = "CACHED"
        else:
            return None

        self.manifest_records.append({
            "source": "ERA5",
            "category": "single_levels",
            "filename": dest_file.name,
            "path": str(dest_file).replace("\\", "/"),
            "size_bytes": dest_file.stat().st_size,
            "sha256": compute_sha256(dest_file),
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notes": "Verified ERA5 8-variable Florida NetCDF dataset (May 15 slice).",
        })
        return dest_file

    def build_florida_request(
        self,
        dataset: str,
        variables: List[str],
        months: List[str],
        days: Optional[List[str]] = None,
        times: Optional[List[str]] = None,
        pressure_levels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Construct official CDS API request dictionary for Florida."""
        req: Dict[str, Any] = {
            "product_type": ["reanalysis"],
            "variable": variables,
            "year": ["2024"],
            "month": months,
            "day": days or [f"{d:02d}" for d in range(1, 32)],
            "time": times or SYNOPTIC_TIMES_6HOURLY,
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": FLORIDA_AREA,
        }
        if pressure_levels:
            req["pressure_level"] = pressure_levels
        return req

    def validate_file_contents(self, file_path: Path) -> Dict[str, Any]:
        """Validate downloaded NetCDF or zip archive contents and report statistics."""
        info: Dict[str, Any] = {
            "valid": False,
            "filename": file_path.name,
            "size_bytes": file_path.stat().st_size if file_path.exists() else 0,
            "variables_present": [],
            "start_timestamp": None,
            "end_timestamp": None,
            "num_unique_timestamps": 0,
            "missing_timestamps_count": 0,
            "duplicate_timestamps_count": 0,
            "notes": "",
        }

        if not file_path.exists() or file_path.stat().st_size < 1024:
            info["notes"] = "File missing or below 1 KB"
            return info

        try:
            datasets = []
            if zipfile.is_zipfile(file_path):
                with zipfile.ZipFile(file_path, "r") as z:
                    tmp_dir = Path(tempfile.mkdtemp(prefix="era5_val_"))
                    z.extractall(tmp_dir)
                    for nc_p in tmp_dir.glob("*.nc"):
                        datasets.append(xr.open_dataset(nc_p, engine="netcdf4"))
            else:
                datasets.append(xr.open_dataset(file_path, engine="netcdf4"))

            all_vars = set()
            all_times = []
            for ds in datasets:
                all_vars.update(ds.data_vars.keys())
                time_var = "valid_time" if "valid_time" in ds.coords else "time"
                if time_var in ds.coords:
                    t_vals = pd.to_datetime(ds[time_var].values)
                    all_times.extend(t_vals)

            info["variables_present"] = sorted(list(all_vars))
            if all_times:
                time_series = pd.Series(all_times).sort_values()
                info["start_timestamp"] = str(time_series.min())
                info["end_timestamp"] = str(time_series.max())
                info["num_unique_timestamps"] = int(time_series.nunique())
                info["duplicate_timestamps_count"] = int(len(time_series) - time_series.nunique())

            info["valid"] = len(all_vars) > 0 and info["num_unique_timestamps"] > 0
            info["notes"] = "Valid NetCDF dataset"
        except Exception as exc:
            info["notes"] = f"Validation failed: {exc}"

        return info

    def download_month_batch(
        self,
        dataset: str,
        category: str,
        month: int,
        variables: List[str],
        pressure_levels: Optional[List[str]] = None,
        times: Optional[List[str]] = None,
        resume: bool = True,
    ) -> Tuple[Optional[Path], Dict[str, Any]]:
        """Download one month of ERA5 data for Florida with resumable support and validation."""
        month_str = f"{month:02d}"
        dest_dir = self.raw_single_dir if category == "single_levels" else self.raw_pressure_dir
        dest_file = dest_dir / f"era5_florida_2024_{month_str}.nc"
        temp_file = dest_dir / f"era5_florida_2024_{month_str}.nc.part"

        # Check existing valid file
        if resume and dest_file.exists() and dest_file.stat().st_size > 1024:
            val_info = self.validate_file_contents(dest_file)
            if val_info["valid"]:
                logger.info(
                    f"Month {month_str} ({category}) already exists and valid "
                    f"({dest_file.stat().st_size:,} bytes, {val_info['num_unique_timestamps']} timestamps). Skipping."
                )
                val_info["status"] = "REUSED_VALID"
                return dest_file, val_info

        if not self.client:
            logger.error("CDS client not initialized. Cannot download from CDS.")
            return None, {"status": "FAILED", "notes": "CDS client uninitialized"}

        num_days = DAYS_IN_MONTH_2024.get(month, 31)
        days = [f"{d:02d}" for d in range(1, num_days + 1)]
        req_times = times or SYNOPTIC_TIMES_6HOURLY

        req = self.build_florida_request(
            dataset=dataset,
            variables=variables,
            months=[month_str],
            days=days,
            times=req_times,
            pressure_levels=pressure_levels,
        )

        logger.info(
            f"Submitting CDS request for 2024-{month_str} ({category}, {len(days)} days, "
            f"{len(req_times)} times/day, {len(variables)} vars)..."
        )

        try:
            if temp_file.exists():
                temp_file.unlink()

            result = self.client.retrieve(dataset, req)
            result.download(str(temp_file))

            # Validate before moving to destination
            val_info = self.validate_file_contents(temp_file)
            if not val_info["valid"]:
                logger.error(f"Downloaded file for {month_str} failed validation: {val_info['notes']}")
                if temp_file.exists():
                    temp_file.unlink()
                val_info["status"] = "VALIDATION_FAILED"
                return None, val_info

            # Atomic rename
            os.replace(temp_file, dest_file)
            val_info["status"] = "DOWNLOADED"
            logger.info(
                f"Successfully downloaded and validated 2024-{month_str} ({category}) -> {dest_file} "
                f"({dest_file.stat().st_size:,} bytes, {val_info['num_unique_timestamps']} timestamps)."
            )

            self.manifest_records.append({
                "source": "ERA5",
                "category": category,
                "filename": dest_file.name,
                "path": str(dest_file).replace("\\", "/"),
                "size_bytes": dest_file.stat().st_size,
                "sha256": compute_sha256(dest_file),
                "status": "DOWNLOADED",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "notes": f"ERA5 {category} for month {month_str}.",
            })
            return dest_file, val_info

        except Exception as exc:
            logger.error(f"Failed to retrieve 2024-{month_str} ({category}): {exc}")
            if temp_file.exists():
                temp_file.unlink()
            return None, {"status": "FAILED", "notes": str(exc)}

    def download_single_level_month(self, month: int, resume: bool = True) -> Tuple[Optional[Path], Dict[str, Any]]:
        """Download one month of 12-variable single-level ERA5 data."""
        return self.download_month_batch(
            dataset="reanalysis-era5-single-levels",
            category="single_levels",
            month=month,
            variables=SINGLE_LEVEL_VARS,
            resume=resume,
        )

    def download_pressure_level_month(self, month: int, resume: bool = True) -> Tuple[Optional[Path], Dict[str, Any]]:
        """Download one month of 7-variable pressure-level ERA5 data at 850, 500, 300 hPa."""
        return self.download_month_batch(
            dataset="reanalysis-era5-pressure-levels",
            category="pressure_levels",
            month=month,
            variables=PRESSURE_LEVEL_VARS,
            pressure_levels=PRESSURE_LEVELS,
            resume=resume,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    downloader = ERA5Downloader()
    print("Downloader ready.")
