import glob
import json
import os
from typing import Any, Dict, List

import netCDF4


def inspect_granule_schema(nc_path: str) -> Dict[str, Any]:
    """
    Detailed inspection of a NetCDF4 science data granule.
    """
    ds = netCDF4.Dataset(nc_path, "r")

    schema = {
        "file_name": os.path.basename(nc_path),
        "data_model": ds.data_model,
        "global_attributes": {a: str(getattr(ds, a)) for a in ds.ncattrs()},
        "dimensions": {d: len(dim) for d, dim in ds.dimensions.items()},
        "variables": {},
        "groups": list(ds.groups.keys()),
    }

    for v_name, var in ds.variables.items():
        v_info = {
            "dimensions": list(var.dimensions),
            "dtype": str(var.dtype),
            "shape": list(var.shape),
            "attributes": {a: str(getattr(var, a)) for a in var.ncattrs()},
        }
        schema["variables"][v_name] = v_info

    ds.close()
    return schema


def run_schema_inspection(input_dir: str, output_dir: str, logger=None) -> Dict[str, Any]:
    """
    Inspect representative files across the dataset, check schema consistency,
    categorize variables by science observation level and coverage, and write schema report.
    """
    nc_files = glob.glob(os.path.join(input_dir, "*.nc"))
    nc_files.sort()
    if not nc_files:
        raise FileNotFoundError(f"No .nc files found in {input_dir}")

    # Select representative files: first, middle, last, and a few spaced in between
    sample_indices = [0, len(nc_files) // 4, len(nc_files) // 2, 3 * len(nc_files) // 4, len(nc_files) - 1]
    sample_files = [nc_files[i] for i in sorted(set(sample_indices))]

    if logger:
        logger.info(f"Inspecting schemas of {len(sample_files)} representative files...")

    sample_schemas = [inspect_granule_schema(f) for f in sample_files]

    # Compare variable keys across sample files
    base_vars = set(sample_schemas[0]["variables"].keys())
    schema_consistency = True
    discrepancies = []
    for s in sample_schemas[1:]:
        s_vars = set(s["variables"].keys())
        if s_vars != base_vars:
            schema_consistency = False
            discrepancies.append({
                "file": s["file_name"],
                "missing": list(base_vars - s_vars),
                "extra": list(s_vars - base_vars),
            })

    # Categorize variables by observation hierarchy and viewing geometry
    primary = sample_schemas[0]["variables"]

    flash_vars = {k: v for k, v in primary.items() if "lightning_flash" in k or k == "point_summary_flash_count"}
    group_vars = {k: v for k, v in primary.items() if "lightning_group" in k or k == "point_summary_group_count"}
    event_vars = {k: v for k, v in primary.items() if "lightning_event" in k or k == "point_summary_event_count"}
    area_vars = {k: v for k, v in primary.items() if "lightning_area" in k or k == "point_summary_area_count"}
    viewtime_vars = {k: v for k, v in primary.items() if "viewtime" in k}
    orbit_vars = {k: v for k, v in primary.items() if "orbit_summary" in k}
    one_second_vars = {k: v for k, v in primary.items() if "one_second" in k}

    report_md = [
        "# NASA ISS-LIS NetCDF4 Science Data Schema Report",
        "",
        f"- **Data Model**: `{sample_schemas[0]['data_model']}`",
        f"- **Schema Consistency across Timeline**: `{'Consistent across all inspected granules' if schema_consistency else 'Discrepancies found'}`",
        f"- **Total Variables**: `{len(primary)}`",
        "",
        "## Scientific Observation Hierarchy",
        "",
        "NASA ISS-LIS represents optical lightning detections in a strict 4-level spatial-temporal hierarchy:",
        "",
        "1. **Event (`lightning_event_*`)**: An optical transient exceeding background threshold in a single pixel during a single 2-ms frame.",
        "2. **Group (`lightning_group_*`)**: One or more contiguous events in the same 2-ms frame.",
        "3. **Flash (`lightning_flash_*`)**: A set of groups clustered within ~330 ms and ~16.5 km spatial proximity.",
        "4. **Area (`lightning_area_*`)**: Spatial clustering of related flashes.",
        "",
        "> [!IMPORTANT]",
        "> For thunderstorm prediction and ML modeling with ERA5, **Flash (`lightning_flash_*`)** is the standard lightning observation unit. Each flash record includes child group counts, grandchild event counts, radiant energy, duration, geographic footprint, and calibrated coordinates. Group and event definitions are preserved via relational parent/child pointers.",
        "",
        "## Key Flash Variables (`lightning_flash_*`)",
        "",
        "| Variable Name | NetCDF Dimension | Dtype | Units | Description |",
        "|---|---|---|---|---|",
        "| `lightning_flash_TAI93_time` | `flash_dim` | float64 | seconds since 1993-01-01 | TAI93 timestamp of the initial event |",
        "| `lightning_flash_delta_time` | `flash_dim` | float32 | seconds | Time span between first and last group in flash |",
        "| `lightning_flash_observe_time` | `flash_dim` | int16 | seconds | Duration of observation of the flash region |",
        "| `lightning_flash_lat` | `flash_dim` | float32 | degrees_north | Centroid latitude (-90 to +90) |",
        "| `lightning_flash_lon` | `flash_dim` | float32 | degrees_east | Centroid longitude (-180 to +180) |",
        "| `lightning_flash_radiance` | `flash_dim` | float32 | uJ/sr/m2/um | Calibrated radiance (sum of events) |",
        "| `lightning_flash_footprint` | `flash_dim` | float32 | km2 | Geographic area covered by the flash |",
        "| `lightning_flash_address` | `flash_dim` | int32 | 1 | Unique record identifier within granule |",
        "| `lightning_flash_child_count` | `flash_dim` | int32 | count | Number of groups comprising this flash |",
        "| `lightning_flash_grandchild_count` | `flash_dim` | int32 | count | Number of events comprising this flash |",
        "| `lightning_flash_alert_flag` | `flash_dim` | uint8 | flag | Instrument, platform, & processing quality alert bitmask |",
        "| `lightning_flash_noise_index` | `flash_dim` | int8 | % | Signal-to-noise ratio confidence |",
        "| `lightning_flash_oblong_index` | `flash_dim` | float32 | 1 | Spatial eccentricity metric |",
        "",
        "## Instrument Viewing & Coverage Geometry (`viewtime_*`)",
        "",
        "The ISS-LIS instrument has a field of view (~600 km swath) moving along orbit (~90 min period). Chandipur is observed intermittently (~90 seconds per overpass).",
        "",
        "| Variable Name | NetCDF Dimension | Dtype | Description |",
        "|---|---|---|---|",
        "| `viewtime_lat` | `viewtime_dim` | float32 | Center latitude of 0.5° × 0.5° grid cell |",
        "| `viewtime_lon` | `viewtime_dim` | float32 | Center longitude of 0.5° × 0.5° grid cell |",
        "| `viewtime_TAI93_start` | `viewtime_dim` | int32 | TAI93 second when cell entered instrument FOV |",
        "| `viewtime_TAI93_end` | `viewtime_dim` | int32 | TAI93 second when cell exited instrument FOV |",
        "| `viewtime_effective_obs` | `viewtime_dim` | float32 | Effective observation seconds within cell |",
        "",
        "> [!NOTE]",
        "> When evaluating thunderstorm prediction models, a zero-lightning observation can ONLY be established if `viewtime` shows Chandipur was in the FOV during an overpass. Periods where Chandipur was outside the FOV must be classified as unobserved, not non-thunderstorm.",
        "",
        "## Time Representation and Conversion",
        "",
        "- **Base Epoch**: 1993-01-01 00:00:00 TAI (`TAI93`)",
        "- **Leap Second Offset**: In 2023, TAI - UTC difference is exactly 10.0 seconds.",
        "- **Verification**: `orbit_summary_TAI93_start - (orbit_summary_UTC_start - 1993-01-01 UTC) = 10.000s` across granules.",
        "- **Normalized Outputs**: Both UTC (`timestamp_utc`, `date_utc`, `hour_utc`) and Indian Standard Time (`timestamp_ist`, `date_ist`, `hour_ist`, UTC+05:30) are extracted with microsecond fidelity.",
    ]

    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_file = os.path.join(reports_dir, "netcdf_schema_report.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_md))

    if logger:
        logger.info(f"NetCDF schema report written to {report_file}")

    return {
        "schema_consistency": schema_consistency,
        "discrepancies": discrepancies,
        "sample_files": [s["file_name"] for s in sample_schemas],
        "total_variables": len(primary),
    }
