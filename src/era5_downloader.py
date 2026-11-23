"""ERA5 Atmospheric Data Downloader for Climate Similarity Pilot.

Retrieves ECMWF ERA5 atmospheric reanalysis data via Copernicus Climate Data Store (CDS) API.
Implements spatial subsetting, variable selection, temporal batching, and manifest generation.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cdsapi
import xarray as xr

logger = logging.getLogger(__name__)

# Standard ERA5 Single Levels shortname to CDS parameter mapping
ERA5_VARIABLE_MAP = {
    "cape": "convective_available_potential_energy",
    "cin": "convective_inhibition",
    "tcwv": "total_column_water_vapour",
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "tp": "total_precipitation",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
}


@dataclass
class ERA5DownloadRecord:
    """Record of an ERA5 download request and local file status."""

    region_key: str
    year: int
    month: str
    variables: List[str]
    local_path: str
    file_size_bytes: int
    status: str  # 'DOWNLOADED', 'CACHED', 'FAILED'
    timestamp: str
    request_id: Optional[str] = None
    error_message: Optional[str] = None


class ERA5Downloader:
    """Automated downloader for ERA5 single levels data."""

    def __init__(
        self,
        config: Dict[str, Any],
        raw_base_dir: Path = Path("data/raw/era5_2024"),
        reports_dir: Path = Path("data/reports/climate_similarity_2024"),
    ):
        self.config = config
        self.era5_cfg = config.get("era5_pilot", {})
        self.raw_base_dir = raw_base_dir
        self.reports_dir = reports_dir

        self.raw_base_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_name = self.era5_cfg.get("dataset", "reanalysis-era5-single-levels")
        self.sampling_times = self.era5_cfg.get(
            "sampling_times", ["00:00", "06:00", "12:00", "18:00"]
        )
        self.year = int(self.era5_cfg.get("year", 2024))
        self.manifest_records: List[ERA5DownloadRecord] = []

        try:
            self.client = cdsapi.Client()
        except Exception as exc:
            logger.error(f"Failed to initialize CDS API client: {exc}")
            raise

    def get_cds_area(self, region_key: str) -> List[float]:
        """Convert region config into CDS area [North, West, South, East]."""
        domain_cfg = self.era5_cfg.get("domains", {}).get(region_key, {})
        if not domain_cfg:
            raise ValueError(f"Domain configuration not found for region: {region_key}")

        lat_min = float(domain_cfg["lat_min"])
        lat_max = float(domain_cfg["lat_max"])
        lon_min = float(domain_cfg["lon_min"])
        lon_max = float(domain_cfg["lon_max"])

        # CDS Area format is [North (lat_max), West (lon_min), South (lat_min), East (lon_max)]
        return [lat_max, lon_min, lat_min, lon_max]

    def build_request(
        self,
        region_key: str,
        months: List[str],
        variables: Optional[List[str]] = None,
        days: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Construct the CDS API payload."""
        vars_to_fetch = variables or self.era5_cfg.get("variables", list(ERA5_VARIABLE_MAP.keys()))
        cds_vars = [ERA5_VARIABLE_MAP.get(v, v) for v in vars_to_fetch]

        if days is None:
            # All days of month
            days = [f"{d:02d}" for d in range(1, 32)]

        area = self.get_cds_area(region_key)

        return {
            "product_type": ["reanalysis"],
            "variable": cds_vars,
            "year": [str(self.year)],
            "month": months,
            "day": days,
            "time": self.sampling_times,
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": area,
        }

    def download_region_month(
        self,
        region_key: str,
        month: str,
        variables: Optional[List[str]] = None,
        days: Optional[List[str]] = None,
        force_redownload: bool = False,
    ) -> ERA5DownloadRecord:
        """Download ERA5 data for a single region and month in NetCDF format."""
        region_dir = self.raw_base_dir / region_key
        region_dir.mkdir(parents=True, exist_ok=True)
        dest_file = region_dir / f"era5_{region_key}_{self.year}_{month}.nc"

        now_iso = datetime.now(timezone.utc).isoformat()
        vars_to_fetch = variables or self.era5_cfg.get("variables", list(ERA5_VARIABLE_MAP.keys()))

        if dest_file.exists() and dest_file.stat().st_size > 1024 and not force_redownload:
            logger.info(f"Skipping download (cached): {dest_file.name} ({dest_file.stat().st_size:,} bytes)")
            record = ERA5DownloadRecord(
                region_key=region_key,
                year=self.year,
                month=month,
                variables=vars_to_fetch,
                local_path=str(dest_file),
                file_size_bytes=dest_file.stat().st_size,
                status="CACHED",
                timestamp=now_iso,
            )
            self.manifest_records.append(record)
            return record

        req = self.build_request(region_key, [month], variables=vars_to_fetch, days=days)
        logger.info(f"Submitting CDS request for {region_key} - {self.year}/{month} ({len(vars_to_fetch)} vars)...")

        temp_file = dest_file.with_suffix(dest_file.suffix + ".tmp")
        try:
            result = self.client.retrieve(self.dataset_name, req, str(temp_file))
            if temp_file.exists():
                if zipfile.is_zipfile(temp_file):
                    logger.info(f"CDS returned zip archive; extracting and merging NetCDF components...")
                    with tempfile.TemporaryDirectory() as tmpdir:
                        with zipfile.ZipFile(temp_file) as z:
                            z.extractall(tmpdir)
                        nc_files = [Path(tmpdir) / f for f in os.listdir(tmpdir) if f.endswith(".nc")]
                        if nc_files:
                            datasets = [xr.open_dataset(f) for f in nc_files]
                            merged = xr.merge(datasets)
                            if dest_file.exists():
                                dest_file.unlink()
                            merged.to_netcdf(dest_file)
                            for d in datasets:
                                d.close()
                            merged.close()
                        else:
                            temp_file.rename(dest_file)
                    temp_file.unlink(missing_ok=True)
                else:
                    if dest_file.exists():
                        dest_file.unlink()
                    temp_file.rename(dest_file)

            size = dest_file.stat().st_size
            logger.info(f"Successfully retrieved {dest_file.name} ({size:,} bytes).")
            record = ERA5DownloadRecord(
                region_key=region_key,
                year=self.year,
                month=month,
                variables=vars_to_fetch,
                local_path=str(dest_file),
                file_size_bytes=size,
                status="DOWNLOADED",
                timestamp=now_iso,
            )
            self.manifest_records.append(record)
            return record

        except Exception as exc:
            logger.error(f"Failed to download {region_key} - {month}: {exc}")
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)
            record = ERA5DownloadRecord(
                region_key=region_key,
                year=self.year,
                month=month,
                variables=vars_to_fetch,
                local_path=str(dest_file),
                file_size_bytes=0,
                status="FAILED",
                timestamp=now_iso,
                error_message=str(exc),
            )
            self.manifest_records.append(record)
            return record

    def save_manifest(self) -> Tuple[Path, Path]:
        """Save the ERA5 download manifest to CSV and JSON."""
        csv_path = self.reports_dir / "era5_download_manifest.csv"
        json_path = self.reports_dir / "era5_download_manifest.json"

        with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
            fieldnames = [
                "region_key",
                "year",
                "month",
                "variables",
                "local_path",
                "file_size_bytes",
                "status",
                "timestamp",
                "error_message",
            ]
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.manifest_records:
                d = asdict(r)
                d["variables"] = ";".join(d["variables"])
                writer.writerow(d)

        with open(json_path, "w", encoding="utf-8") as f_json:
            json.dump([asdict(r) for r in self.manifest_records], f_json, indent=2)

        logger.info(f"ERA5 download manifest saved to {csv_path} and {json_path}")
        return csv_path, json_path
