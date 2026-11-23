import csv
import os
from typing import Any, Dict, List, Optional
import netCDF4
import numpy as np

from scripts.geo_filter import BoundingBox
from scripts.time_utils import format_observation_timestamps, tai93_to_utc_datetime


def extract_granule_coverage(
    ds: netCDF4.Dataset,
    granule_name: str,
    target_box: BoundingBox,
) -> Optional[Dict[str, Any]]:
    """
    Examine viewtime variables in granule to determine if target_box was observed.
    Returns coverage details dictionary if target was in FOV, otherwise None.
    """
    if "viewtime_lat" not in ds.variables or "viewtime_lon" not in ds.variables:
        return None

    vt_lats = ds.variables["viewtime_lat"][:]
    vt_lons = ds.variables["viewtime_lon"][:]

    # viewtime grid cells are 0.5 deg x 0.5 deg.
    # Allow a small buffer (0.25 deg) around bounding box to catch any cell center overlapping the region
    buffer = 0.25
    mask = (
        (vt_lats >= (target_box.lat_min - buffer))
        & (vt_lats <= (target_box.lat_max + buffer))
        & (vt_lons >= (target_box.lon_min - buffer))
        & (vt_lons <= (target_box.lon_max + buffer))
    )

    if not np.any(mask):
        return None

    # Target region was viewed during this orbit!
    matching_indices = np.where(mask)[0]
    vt_starts = ds.variables["viewtime_TAI93_start"][:][mask]
    vt_ends = ds.variables["viewtime_TAI93_end"][:][mask]
    vt_eff = ds.variables["viewtime_effective_obs"][:][mask]
    vt_alerts = (
        ds.variables["viewtime_alert_flag"][:][mask]
        if "viewtime_alert_flag" in ds.variables
        else np.zeros(len(matching_indices), dtype=np.uint8)
    )

    min_tai_start = float(np.min(vt_starts))
    max_tai_end = float(np.max(vt_ends))
    max_eff_obs = float(np.max(vt_eff))
    avg_eff_obs = float(np.mean(vt_eff))
    max_alert = int(np.max(vt_alerts))

    t_start_info = format_observation_timestamps(min_tai_start)
    t_end_info = format_observation_timestamps(max_tai_end)

    orbit_num = (
        int(ds.variables["orbit_summary_id_number"][:])
        if "orbit_summary_id_number" in ds.variables
        else -1
    )

    return {
        "granule_id": granule_name,
        "orbit_number": orbit_num,
        "overpass_start_utc": t_start_info["timestamp_utc"],
        "overpass_end_utc": t_end_info["timestamp_utc"],
        "overpass_start_ist": t_start_info["timestamp_ist"],
        "overpass_end_ist": t_end_info["timestamp_ist"],
        "date_utc": t_start_info["date_utc"],
        "date_ist": t_start_info["date_ist"],
        "tai93_start": min_tai_start,
        "tai93_end": max_tai_end,
        "overpass_duration_seconds": round(max_tai_end - min_tai_start, 2),
        "effective_obs_seconds_max": round(max_eff_obs, 2),
        "effective_obs_seconds_avg": round(avg_eff_obs, 2),
        "viewtime_grid_cells_count": len(matching_indices),
        "alert_flag_max": max_alert,
    }


def save_coverage_records(
    coverage_records: List[Dict[str, Any]],
    output_csv_path: str,
    logger=None,
):
    """
    Write overpasses coverage manifest to CSV.
    """
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    if not coverage_records:
        if logger:
            logger.warning("No coverage records found to write.")
        return

    # Sort chronologically by overpass_start_utc
    coverage_records.sort(key=lambda x: x["overpass_start_utc"])

    fieldnames = [
        "granule_id",
        "orbit_number",
        "date_utc",
        "date_ist",
        "overpass_start_utc",
        "overpass_end_utc",
        "overpass_start_ist",
        "overpass_end_ist",
        "overpass_duration_seconds",
        "effective_obs_seconds_max",
        "effective_obs_seconds_avg",
        "viewtime_grid_cells_count",
        "lightning_flashes_detected",
        "is_thunderstorm_active",
        "alert_flag_max",
        "tai93_start",
        "tai93_end",
    ]

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(coverage_records)

    if logger:
        logger.info(f"Wrote {len(coverage_records)} satellite coverage records to {output_csv_path}")
