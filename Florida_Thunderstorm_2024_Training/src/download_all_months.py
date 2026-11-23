"""Sequential, Resumable Monthly Downloader for ERA5 Florida 2024.

Retrieves 6-hourly single-level (12 variables) and pressure-level (850, 500, 300 hPa)
data for all 12 months (January through December 2024).
Validates each downloaded file's NetCDF structure, coordinates, variables, and timestamps.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.download_era5 import (
    DAYS_IN_MONTH_2024,
    DEFAULT_CDS_KEY,
    DEFAULT_CDS_URL,
    ERA5Downloader,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "era5_monthly_downloads.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("ERA5MonthlyManager")


def run_monthly_pipeline(start_month: int = 1, end_month: int = 12) -> List[Dict[str, Any]]:
    """Download and validate all months sequentially."""
    downloader = ERA5Downloader(
        raw_single_dir=PROJECT_ROOT / "data" / "raw" / "era5" / "single_levels",
        raw_pressure_dir=PROJECT_ROOT / "data" / "raw" / "era5" / "pressure_levels",
        parent_raw_dir=PROJECT_ROOT.parent / "data" / "raw" / "era5_2024",
    )

    monthly_logs: List[Dict[str, Any]] = []
    log_csv = PROJECT_ROOT / "reports" / "era5_monthly_download_log.csv"

    logger.info("==================================================================")
    logger.info(f"STARTING ERA5 2024 6-HOURLY RETRIEVAL (MONTHS {start_month:02d} TO {end_month:02d})")
    logger.info("==================================================================")

    for month in range(start_month, end_month + 1):
        month_str = f"{month:02d}"
        logger.info(f"\n>>> PROCESSING MONTH 2024-{month_str} <<<")

        # 1. Single Levels
        logger.info(f"Checking 2024-{month_str} Single Levels (12 variables, 6-hourly)...")
        dest_single, info_single = downloader.download_single_level_month(month, resume=True)
        rec_single = {
            "month": month_str,
            "category": "single_levels",
            "status": info_single.get("status", "UNKNOWN"),
            "filename": info_single.get("filename", ""),
            "size_bytes": info_single.get("size_bytes", 0),
            "num_unique_timestamps": info_single.get("num_unique_timestamps", 0),
            "start_timestamp": info_single.get("start_timestamp", ""),
            "end_timestamp": info_single.get("end_timestamp", ""),
            "variables_count": len(info_single.get("variables_present", [])),
            "variables": ";".join(info_single.get("variables_present", [])),
            "notes": info_single.get("notes", ""),
        }
        monthly_logs.append(rec_single)
        logger.info(f"Month {month_str} Single Levels Status: {rec_single['status']} ({rec_single['size_bytes']:,} bytes)")

        # 2. Pressure Levels
        logger.info(f"Checking 2024-{month_str} Pressure Levels (850, 500, 300 hPa, 6-hourly)...")
        dest_pressure, info_pressure = downloader.download_pressure_level_month(month, resume=True)
        rec_pressure = {
            "month": month_str,
            "category": "pressure_levels",
            "status": info_pressure.get("status", "UNKNOWN"),
            "filename": info_pressure.get("filename", ""),
            "size_bytes": info_pressure.get("size_bytes", 0),
            "num_unique_timestamps": info_pressure.get("num_unique_timestamps", 0),
            "start_timestamp": info_pressure.get("start_timestamp", ""),
            "end_timestamp": info_pressure.get("end_timestamp", ""),
            "variables_count": len(info_pressure.get("variables_present", [])),
            "variables": ";".join(info_pressure.get("variables_present", [])),
            "notes": info_pressure.get("notes", ""),
        }
        monthly_logs.append(rec_pressure)
        logger.info(f"Month {month_str} Pressure Levels Status: {rec_pressure['status']} ({rec_pressure['size_bytes']:,} bytes)")

        # Save incremental log
        fieldnames = [
            "month", "category", "status", "filename", "size_bytes",
            "num_unique_timestamps", "start_timestamp", "end_timestamp",
            "variables_count", "variables", "notes"
        ]
        with open(log_csv, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(monthly_logs)

    logger.info("==================================================================")
    logger.info(f"MONTHLY RETRIEVAL CYCLE COMPLETED. Log saved to {log_csv}")
    logger.info("==================================================================")
    return monthly_logs


if __name__ == "__main__":
    run_monthly_pipeline(1, 12)
