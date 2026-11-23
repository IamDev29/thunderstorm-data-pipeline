import csv
import glob
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import netCDF4
import numpy as np
from tqdm import tqdm

from scripts.coverage import extract_granule_coverage, save_coverage_records
from scripts.geo_filter import BoundingBox, get_bounding_boxes_from_config
from scripts.time_utils import format_observation_timestamps, tai93_to_utc_datetime


def process_single_granule(
    nc_path: str,
    chandipur_box: BoundingBox,
    regional_box: BoundingBox,
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    observation_level: str = "flash",
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Safely process one NetCDF science granule.
    Returns:
        (observations_list, coverage_info_or_none, manifest_entry)
    """
    fname = os.path.basename(nc_path)
    granule_id = os.path.splitext(fname)[0]

    manifest = {
        "filename": fname,
        "filepath": os.path.abspath(nc_path),
        "granule_id": granule_id,
        "orbit_number": -1,
        "granule_utc_start": "",
        "granule_utc_end": "",
        "total_observations_in_file": 0,
        "extracted_observations": 0,
        "chandipur_observations": 0,
        "regional_observations": 0,
        "invalid_records": 0,
        "chandipur_coverage_observed": False,
        "status": "PENDING",
        "error_message": "",
    }

    observations = []
    coverage_info = None

    try:
        with netCDF4.Dataset(nc_path, "r") as ds:
            # Orbit metadata
            if "orbit_summary_UTC_start" in ds.variables:
                manifest["granule_utc_start"] = str(ds.variables["orbit_summary_UTC_start"][:])
            if "orbit_summary_TAI93_end" in ds.variables:
                manifest["granule_utc_end"] = tai93_to_utc_datetime(float(ds.variables["orbit_summary_TAI93_end"][:])).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            if "orbit_summary_id_number" in ds.variables:
                manifest["orbit_number"] = int(ds.variables["orbit_summary_id_number"][:])

            # Check coverage over Chandipur
            cov = extract_granule_coverage(ds, granule_id, chandipur_box)
            if cov:
                coverage_info = cov
                manifest["chandipur_coverage_observed"] = True

            # Observation extraction (Flash level)
            if observation_level == "flash":
                flash_count = (
                    int(ds.variables["point_summary_flash_count"][:])
                    if "point_summary_flash_count" in ds.variables
                    else 0
                )
                manifest["total_observations_in_file"] = flash_count

                if flash_count > 0 and "lightning_flash_lat" in ds.variables:
                    lats = ds.variables["lightning_flash_lat"][:]
                    lons = ds.variables["lightning_flash_lon"][:]
                    tais = ds.variables["lightning_flash_TAI93_time"][:]
                    rads = ds.variables["lightning_flash_radiance"][:]
                    footprints = (
                        ds.variables["lightning_flash_footprint"][:]
                        if "lightning_flash_footprint" in ds.variables
                        else np.zeros(flash_count, dtype=np.float32)
                    )
                    deltas = (
                        ds.variables["lightning_flash_delta_time"][:]
                        if "lightning_flash_delta_time" in ds.variables
                        else np.zeros(flash_count, dtype=np.float32)
                    )
                    obs_times = (
                        ds.variables["lightning_flash_observe_time"][:]
                        if "lightning_flash_observe_time" in ds.variables
                        else np.zeros(flash_count, dtype=np.int16)
                    )
                    addrs = (
                        ds.variables["lightning_flash_address"][:]
                        if "lightning_flash_address" in ds.variables
                        else np.arange(flash_count, dtype=np.int32)
                    )
                    parent_addrs = (
                        ds.variables["lightning_flash_parent_address"][:]
                        if "lightning_flash_parent_address" in ds.variables
                        else np.zeros(flash_count, dtype=np.int32)
                    )
                    child_counts = (
                        ds.variables["lightning_flash_child_count"][:]
                        if "lightning_flash_child_count" in ds.variables
                        else np.zeros(flash_count, dtype=np.int32)
                    )
                    grandchild_counts = (
                        ds.variables["lightning_flash_grandchild_count"][:]
                        if "lightning_flash_grandchild_count" in ds.variables
                        else np.zeros(flash_count, dtype=np.int32)
                    )
                    alerts = (
                        ds.variables["lightning_flash_alert_flag"][:]
                        if "lightning_flash_alert_flag" in ds.variables
                        else np.zeros(flash_count, dtype=np.uint8)
                    )
                    noise_idx = (
                        ds.variables["lightning_flash_noise_index"][:]
                        if "lightning_flash_noise_index" in ds.variables
                        else np.zeros(flash_count, dtype=np.int8)
                    )
                    oblong_idx = (
                        ds.variables["lightning_flash_oblong_index"][:]
                        if "lightning_flash_oblong_index" in ds.variables
                        else np.zeros(flash_count, dtype=np.float32)
                    )

                    for i in range(flash_count):
                        lat_val = float(lats[i])
                        lon_val = float(lons[i])
                        tai_val = float(tais[i])

                        # Coordinate sanity check
                        if not (-90.0 <= lat_val <= 90.0 and -180.0 <= lon_val <= 180.0):
                            manifest["invalid_records"] += 1
                            continue

                        # Timestamp normalization
                        ts_info = format_observation_timestamps(tai_val)

                        # Filter by actual observation date (e.g. 2023)
                        obs_date = ts_info["date_utc"]
                        if not (start_date <= obs_date <= end_date):
                            continue

                        # Check bounding boxes
                        in_chandipur = chandipur_box.contains_point(lat_val, lon_val)
                        in_regional = regional_box.contains_point(lat_val, lon_val)

                        if in_chandipur:
                            manifest["chandipur_observations"] += 1
                        if in_regional:
                            manifest["regional_observations"] += 1

                        record = {
                            "granule_id": granule_id,
                            "orbit_number": manifest["orbit_number"],
                            "observation_type": "flash",
                            "record_id": int(addrs[i]),
                            "parent_area_id": int(parent_addrs[i]),
                            "latitude": round(lat_val, 6),
                            "longitude": round(lon_val, 6),
                            "timestamp_utc": ts_info["timestamp_utc"],
                            "timestamp_ist": ts_info["timestamp_ist"],
                            "date_utc": ts_info["date_utc"],
                            "date_ist": ts_info["date_ist"],
                            "hour_utc": ts_info["hour_utc"],
                            "hour_ist": ts_info["hour_ist"],
                            "tai93_time": round(tai_val, 6),
                            "delta_time_seconds": round(float(deltas[i]), 6),
                            "observe_time_seconds": int(obs_times[i]),
                            "radiance": round(float(rads[i]), 4),
                            "footprint_km2": round(float(footprints[i]), 2),
                            "child_group_count": int(child_counts[i]),
                            "child_event_count": int(grandchild_counts[i]),
                            "alert_flag": int(alerts[i]),
                            "noise_index": int(noise_idx[i]),
                            "oblong_index": round(float(oblong_idx[i]), 4),
                            "in_chandipur": in_chandipur,
                            "in_regional": in_regional,
                            "source_file": fname,
                        }
                        observations.append(record)

            manifest["extracted_observations"] = len(observations)
            manifest["status"] = "SUCCESS"

    except Exception as e:
        manifest["status"] = "FAILED"
        manifest["error_message"] = str(e)

    # Attach flash detection count to coverage record
    if coverage_info:
        coverage_info["lightning_flashes_detected"] = manifest["chandipur_observations"]
        coverage_info["is_thunderstorm_active"] = int(manifest["chandipur_observations"] > 0)

    return observations, coverage_info, manifest


def run_extraction_pipeline(
    config: Dict[str, Any],
    limit_files: Optional[int] = None,
    logger=None,
) -> Dict[str, Any]:
    """
    Execute batch extraction across all granules in config['input_dir'].
    Saves:
    - output/reports/processed_granules.csv
    - output/chandipur/chandipur_satellite_overpasses.csv
    - Raw observations passed to combiner
    """
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    start_date = config.get("start_date", "2023-01-01")
    end_date = config.get("end_date", "2023-12-31")
    observation_level = config.get("observation_level", "flash")

    chandipur_box, regional_box = get_bounding_boxes_from_config(config)

    nc_files = glob.glob(os.path.join(input_dir, "**", "*.nc"), recursive=True)
    nc_files.sort()

    if limit_files:
        nc_files = nc_files[:limit_files]

    total_files = len(nc_files)
    if logger:
        logger.info(f"Starting batch extraction on {total_files} granules (Period: {start_date} to {end_date})...")

    manifest_entries = []
    coverage_records = []
    all_observations = []
    chandipur_observations = []
    regional_observations = []

    t_start = time.time()
    for nc_file in tqdm(nc_files, desc="Processing granules", unit="file"):
        obs, cov, mani = process_single_granule(
            nc_path=nc_file,
            chandipur_box=chandipur_box,
            regional_box=regional_box,
            start_date=start_date,
            end_date=end_date,
            observation_level=observation_level,
        )

        manifest_entries.append(mani)
        if cov:
            coverage_records.append(cov)

        for record in obs:
            all_observations.append(record)
            if record["in_chandipur"]:
                chandipur_observations.append(record)
            if record["in_regional"]:
                regional_observations.append(record)

    t_duration = time.time() - t_start

    # Save processed granules manifest
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    manifest_csv = os.path.join(reports_dir, "processed_granules.csv")

    manifest_fields = [
        "filename",
        "granule_id",
        "orbit_number",
        "granule_utc_start",
        "granule_utc_end",
        "total_observations_in_file",
        "extracted_observations",
        "chandipur_observations",
        "regional_observations",
        "invalid_records",
        "chandipur_coverage_observed",
        "status",
        "error_message",
        "filepath",
    ]
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_entries)

    # Save satellite overpass coverage
    coverage_csv = os.path.join(output_dir, "chandipur", "chandipur_satellite_overpasses.csv")
    save_coverage_records(coverage_records, coverage_csv, logger)

    summary = {
        "total_files": total_files,
        "processed_success": sum(1 for m in manifest_entries if m["status"] == "SUCCESS"),
        "processed_failed": sum(1 for m in manifest_entries if m["status"] == "FAILED"),
        "total_observations_extracted": len(all_observations),
        "chandipur_observations_count": len(chandipur_observations),
        "regional_observations_count": len(regional_observations),
        "chandipur_overpasses_count": len(coverage_records),
        "duration_seconds": round(t_duration, 2),
        "manifest_path": manifest_csv,
        "coverage_path": coverage_csv,
        "all_observations": all_observations,
        "chandipur_observations": chandipur_observations,
        "regional_observations": regional_observations,
    }

    if logger:
        logger.info(
            f"Extraction complete in {summary['duration_seconds']}s: "
            f"{summary['processed_success']}/{total_files} granules succeeded. "
            f"Total obs: {len(all_observations)}, Chandipur: {len(chandipur_observations)}, "
            f"Regional: {len(regional_observations)}, Chandipur Overpasses: {len(coverage_records)}."
        )

    return summary
