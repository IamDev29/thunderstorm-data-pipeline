import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple


OBSERVATION_CSV_FIELDNAMES = [
    "granule_id",
    "orbit_number",
    "observation_type",
    "record_id",
    "parent_area_id",
    "timestamp_utc",
    "timestamp_ist",
    "date_utc",
    "date_ist",
    "hour_utc",
    "hour_ist",
    "latitude",
    "longitude",
    "radiance",
    "footprint_km2",
    "delta_time_seconds",
    "observe_time_seconds",
    "child_group_count",
    "child_event_count",
    "alert_flag",
    "noise_index",
    "oblong_index",
    "in_chandipur",
    "in_regional",
    "tai93_time",
    "source_file",
]


def stable_sort_observations(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort consolidated observations using a stable algorithm by:
    1. timestamp_utc
    2. latitude
    3. longitude
    4. granule_id
    5. record_id
    """
    return sorted(
        records,
        key=lambda r: (
            r["timestamp_utc"],
            r["latitude"],
            r["longitude"],
            r["granule_id"],
            r.get("record_id", 0),
        ),
    )


def analyze_duplicates(records: List[Dict[str, Any]], dataset_name: str = "dataset") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Examine observations for duplicate entries:
    - Exact duplicate records (identical source granule + record_id)
    - Orbit-boundary spatial-temporal duplicates (identical timestamp, coordinates, radiance)
    Returns: (deduplicated_records, duplicate_report)
    """
    seen_unique_ids = set()
    exact_duplicates = []
    orbit_overlap_duplicates = []
    spatial_temporal_seen = {}

    deduped = []

    for r in records:
        # Granule + observation ID key
        uid = (r["granule_id"], r["observation_type"], r.get("record_id", 0))

        # Spatial-temporal key (within microsecond, exact lat/lon, exact radiance)
        st_key = (r["timestamp_utc"], r["latitude"], r["longitude"], r["radiance"])

        if uid in seen_unique_ids:
            exact_duplicates.append(r)
            continue

        if st_key in spatial_temporal_seen:
            orig = spatial_temporal_seen[st_key]
            orbit_overlap_duplicates.append({
                "duplicate_record": r,
                "original_record": orig,
            })
            # We keep the record in deduped unless exact duplicate, but note overlap
            # Depending on policy, we can either keep or filter.
            # To preserve legitimate physics while dropping redundant telemetry:
            continue

        seen_unique_ids.add(uid)
        spatial_temporal_seen[st_key] = r
        deduped.append(r)

    report = {
        "dataset_name": dataset_name,
        "total_input_records": len(records),
        "exact_duplicates_count": len(exact_duplicates),
        "spatial_temporal_overlap_duplicates_count": len(orbit_overlap_duplicates),
        "deduplicated_records_count": len(deduped),
        "deduplication_policy": (
            "Preserves unmodified original dataset. Deduplicated dataset removes exact duplicate telemetry "
            "and orbit boundary cross-granule duplicate flash detections."
        ),
    }

    return deduped, report


def write_csv_dataset(records: List[Dict[str, Any]], output_filepath: str):
    """
    Write a list of observation dicts to CSV.
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OBSERVATION_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)


def write_daily_partitions(
    records: List[Dict[str, Any]],
    daily_output_dir: str,
    partition_key: str = "date_utc",
) -> Dict[str, int]:
    """
    Split records by date and write individual daily CSV files.
    """
    os.makedirs(daily_output_dir, exist_ok=True)
    daily_groups = defaultdict(list)

    for r in records:
        d = r.get(partition_key, "unknown_date")
        daily_groups[d].append(r)

    daily_counts = {}
    for d, recs in sorted(daily_groups.items()):
        daily_file = os.path.join(daily_output_dir, f"{d}.csv")
        write_csv_dataset(recs, daily_file)
        daily_counts[d] = len(recs)

    return daily_counts


def combine_and_save_all(
    all_observations: List[Dict[str, Any]],
    chandipur_observations: List[Dict[str, Any]],
    regional_observations: List[Dict[str, Any]],
    output_dir: str,
    logger=None,
) -> Dict[str, Any]:
    """
    Sort, deduplicate, and write all consolidated and daily datasets.
    """
    if logger:
        logger.info("Sorting and assembling chronological datasets...")

    # Sort all datasets chronologically
    sorted_all = stable_sort_observations(all_observations)
    sorted_chandipur = stable_sort_observations(chandipur_observations)
    sorted_regional = stable_sort_observations(regional_observations)

    # Analyze duplicates for Chandipur and Global
    chandipur_dedup, chandipur_dup_report = analyze_duplicates(sorted_chandipur, "chandipur")
    all_dedup, all_dup_report = analyze_duplicates(sorted_all, "all_global")

    # File paths
    combined_csv = os.path.join(output_dir, "combined", "iss_lis_all_observations.csv")
    chandipur_csv = os.path.join(output_dir, "chandipur", "chandipur_lightning_observations.csv")
    chandipur_dedup_csv = os.path.join(output_dir, "chandipur", "chandipur_lightning_observations_dedup.csv")
    regional_csv = os.path.join(output_dir, "regional", "regional_lightning_observations.csv")
    daily_dir = os.path.join(output_dir, "chandipur", "daily")

    # Write files
    write_csv_dataset(sorted_all, combined_csv)
    write_csv_dataset(sorted_chandipur, chandipur_csv)
    write_csv_dataset(chandipur_dedup, chandipur_dedup_csv)
    write_csv_dataset(sorted_regional, regional_csv)

    # Write daily Chandipur partitions
    daily_counts = write_daily_partitions(sorted_chandipur, daily_dir, partition_key="date_utc")

    # Write deduplication report
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    dup_json_path = os.path.join(reports_dir, "deduplication_report.json")
    with open(dup_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "chandipur_report": chandipur_dup_report,
            "global_report": all_dup_report,
        }, f, indent=2)

    dup_md_path = os.path.join(reports_dir, "deduplication_report.md")
    with open(dup_md_path, "w", encoding="utf-8") as f:
        f.write("# NASA ISS-LIS Deduplication Analysis Report\n\n")
        f.write("## Chandipur Dataset\n\n")
        f.write(f"- **Total Input Observations**: {chandipur_dup_report['total_input_records']}\n")
        f.write(f"- **Exact Duplicates Removed**: {chandipur_dup_report['exact_duplicates_count']}\n")
        f.write(f"- **Orbit Boundary Overlaps**: {chandipur_dup_report['spatial_temporal_overlap_duplicates_count']}\n")
        f.write(f"- **Deduplicated Observations**: {chandipur_dup_report['deduplicated_records_count']}\n\n")
        f.write("## Global Observations Dataset\n\n")
        f.write(f"- **Total Input Observations**: {all_dup_report['total_input_records']}\n")
        f.write(f"- **Exact Duplicates Removed**: {all_dup_report['exact_duplicates_count']}\n")
        f.write(f"- **Orbit Boundary Overlaps**: {all_dup_report['spatial_temporal_overlap_duplicates_count']}\n")
        f.write(f"- **Deduplicated Observations**: {all_dup_report['deduplicated_records_count']}\n\n")
        f.write("> [!NOTE]\n")
        f.write("> The primary `chandipur_lightning_observations.csv` preserves the 100% unmodified observations. A separate `chandipur_lightning_observations_dedup.csv` is provided for ML training.\n")

    summary = {
        "all_observations_file": combined_csv,
        "all_observations_count": len(sorted_all),
        "chandipur_observations_file": chandipur_csv,
        "chandipur_observations_count": len(sorted_chandipur),
        "chandipur_dedup_count": len(chandipur_dedup),
        "regional_observations_file": regional_csv,
        "regional_observations_count": len(sorted_regional),
        "daily_partitions_count": len(daily_counts),
        "daily_dates": list(daily_counts.keys()),
        "deduplication_report_path": dup_json_path,
    }

    if logger:
        logger.info(
            f"Saved: {len(sorted_all)} global observations, {len(sorted_chandipur)} Chandipur observations, "
            f"{len(daily_counts)} daily Chandipur partitions, and deduplication reports."
        )

    return summary
