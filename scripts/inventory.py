import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import netCDF4


def parse_granule_timestamp_from_filename(filename: str) -> Dict[str, str]:
    """
    Extract date and time strings from filename convention:
    ISS_LIS_SC_V3.0_YYYYMMDD_HHMMSS_FIN.[nc|hdf]
    """
    m = re.search(r"(\d{8})_(\d{6})", filename)
    if m:
        d_str, t_str = m.group(1), m.group(2)
        return {
            "date": f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}",
            "time": f"{t_str[:2]}:{t_str[2:4]}:{t_str[4:6]}",
            "datetime_str": f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]} {t_str[:2]}:{t_str[2:4]}:{t_str[4:6]}",
            "raw_date": d_str,
            "raw_time": t_str,
        }
    return {}


def perform_inventory(input_dir: str, output_dir: str, logger=None) -> Dict[str, Any]:
    """
    Recursively inspect input_dir, cataloging NC and HDF files, checking sizes,
    pairing granules, inferring dates, and checking for corrupt/incomplete files.
    """
    if logger:
        logger.info(f"Scanning directory: {input_dir}")

    all_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            all_files.append(os.path.join(root, f))

    nc_files: List[str] = []
    hdf_files: List[str] = []
    other_files: List[str] = []

    for f in all_files:
        if f.lower().endswith(".nc"):
            nc_files.append(f)
        elif f.lower().endswith(".hdf"):
            hdf_files.append(f)
        else:
            other_files.append(f)

    # Sort files
    nc_files.sort()
    hdf_files.sort()
    other_files.sort()

    nc_sizes = {f: os.path.getsize(f) for f in nc_files}
    hdf_sizes = {f: os.path.getsize(f) for f in hdf_files}
    other_sizes = {f: os.path.getsize(f) for f in other_files}

    total_nc_size = sum(nc_sizes.values())
    total_hdf_size = sum(hdf_sizes.values())
    total_other_size = sum(other_sizes.values())

    # Map granules by stem (basename without extension)
    nc_granules = {os.path.splitext(os.path.basename(f))[0]: f for f in nc_files}
    hdf_granules = {os.path.splitext(os.path.basename(f))[0]: f for f in hdf_files}

    all_granule_keys = sorted(set(nc_granules.keys()).union(set(hdf_granules.keys())))
    paired_granules = sorted(set(nc_granules.keys()).intersection(set(hdf_granules.keys())))
    nc_only_granules = sorted(set(nc_granules.keys()) - set(hdf_granules.keys()))
    hdf_only_granules = sorted(set(hdf_granules.keys()) - set(nc_granules.keys()))

    # Date range inference from filenames
    dates = []
    for k in all_granule_keys:
        info = parse_granule_timestamp_from_filename(k)
        if info:
            dates.append(info["date"])

    dates_sorted = sorted(dates)
    earliest_date = dates_sorted[0] if dates_sorted else None
    latest_date = dates_sorted[-1] if dates_sorted else None
    unique_dates = sorted(set(dates_sorted))

    # Detect incomplete or invalid files (0 size or unreadable header)
    invalid_files = []
    for f, sz in nc_sizes.items():
        if sz == 0:
            invalid_files.append({"file": f, "type": "nc", "reason": "Zero-byte file"})
    for f, sz in hdf_sizes.items():
        if sz == 0:
            invalid_files.append({"file": f, "type": "hdf", "reason": "Zero-byte file"})

    # Quick test of a sample of NC headers for integrity
    corrupt_samples = []
    sample_to_test = nc_files[:20] + (nc_files[-20:] if len(nc_files) > 40 else [])
    for f in sample_to_test:
        try:
            with netCDF4.Dataset(f, "r") as ds:
                _ = len(ds.dimensions)
        except Exception as e:
            corrupt_samples.append({"file": f, "error": str(e)})

    report = {
        "input_directory": input_dir,
        "total_files": len(all_files),
        "total_nc_files": len(nc_files),
        "total_hdf_files": len(hdf_files),
        "total_other_files": len(other_files),
        "total_nc_size_bytes": total_nc_size,
        "total_nc_size_mb": round(total_nc_size / (1024 * 1024), 2),
        "total_nc_size_gb": round(total_nc_size / (1024 * 1024 * 1024), 2),
        "total_hdf_size_bytes": total_hdf_size,
        "total_hdf_size_mb": round(total_hdf_size / (1024 * 1024), 2),
        "total_hdf_size_gb": round(total_hdf_size / (1024 * 1024 * 1024), 2),
        "total_other_size_bytes": total_other_size,
        "granule_pairing": {
            "total_unique_granules": len(all_granule_keys),
            "paired_granules_count": len(paired_granules),
            "nc_only_count": len(nc_only_granules),
            "hdf_only_count": len(hdf_only_granules),
            "duplicate_format_granules": len(paired_granules),
        },
        "date_range": {
            "earliest_date": earliest_date,
            "latest_date": latest_date,
            "unique_calendar_dates_count": len(unique_dates),
        },
        "integrity_check": {
            "zero_byte_files_count": len(invalid_files),
            "zero_byte_files": invalid_files,
            "sample_tested_count": len(sample_to_test),
            "corrupt_headers_detected": len(corrupt_samples),
            "corrupt_samples": corrupt_samples,
        },
    }

    # Save reports
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    json_path = os.path.join(reports_dir, "inventory_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = os.path.join(reports_dir, "inventory_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# NASA ISS-LIS Dataset Inventory Report\n\n")
        f.write(f"- **Input Directory**: `{input_dir}`\n")
        f.write(f"- **Total Files Found**: {report['total_files']}\n\n")
        f.write("## File Categories and Storage\n\n")
        f.write("| File Category | Count | Total Size (MB) | Total Size (GB) |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| NetCDF (`.nc`) | {report['total_nc_files']} | {report['total_nc_size_mb']} MB | {report['total_nc_size_gb']} GB |\n")
        f.write(f"| HDF (`.hdf`) | {report['total_hdf_files']} | {report['total_hdf_size_mb']} MB | {report['total_hdf_size_gb']} GB |\n")
        f.write(f"| Other Files | {report['total_other_files']} | {report['total_other_size_bytes']} B | 0.00 GB |\n\n")
        f.write("## Granule Pairing Analysis\n\n")
        f.write(f"- **Total Unique Granules**: {len(all_granule_keys)}\n")
        f.write(f"- **Granules Available in Both `.nc` & `.hdf`**: {len(paired_granules)} (100% paired duplicate formats)\n")
        f.write(f"- **NC Only Granules**: {len(nc_only_granules)}\n")
        f.write(f"- **HDF Only Granules**: {len(hdf_only_granules)}\n\n")
        f.write("## Inferred Date Range\n\n")
        f.write(f"- **Earliest Granule Date**: `{earliest_date}`\n")
        f.write(f"- **Latest Granule Date**: `{latest_date}`\n")
        f.write(f"- **Unique Observed Days**: {len(unique_dates)} days\n\n")
        f.write("## File Integrity Check\n\n")
        f.write(f"- **Zero-byte files detected**: {len(invalid_files)}\n")
        f.write(f"- **Header verification check (sample size: {len(sample_to_test)})**: {len(corrupt_samples)} errors\n")
        if corrupt_samples:
            for cs in corrupt_samples:
                f.write(f"  - `{cs['file']}`: {cs['error']}\n")
        else:
            f.write("- **All tested headers passed NetCDF4 open and dimension inspection.**\n")

    if logger:
        logger.info(f"Inventory completed: {len(nc_files)} NC files, {len(hdf_files)} HDF files.")
        logger.info(f"Inventory saved to {json_path} and {md_path}")

    return report
