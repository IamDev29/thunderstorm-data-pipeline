import csv
import datetime
import glob
import json
import os
from typing import Any, Dict, List

from scripts.geo_filter import BoundingBox, get_bounding_boxes_from_config


def validate_csv_dataset(
    csv_path: str,
    target_box: BoundingBox = None,
    must_be_sorted: bool = True,
) -> Dict[str, Any]:
    """
    Run detailed integrity tests on an output CSV file.
    """
    results = {
        "file_exists": os.path.isfile(csv_path),
        "file_path": csv_path,
        "row_count": 0,
        "timestamps_valid": True,
        "chronologically_sorted": True,
        "coords_valid": True,
        "inside_bounding_box": True,
        "provenance_retained": True,
        "errors": [],
        "warnings": [],
    }

    if not results["file_exists"]:
        results["errors"].append(f"File not found: {csv_path}")
        return results

    try:
        prev_utc_dt = None
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            results["row_count"] = len(rows)

            for idx, row in enumerate(rows):
                # 1. Check timestamps
                ts_utc = row.get("timestamp_utc", "")
                try:
                    # e.g. 2023-01-01T02:32:14.300000Z
                    dt_utc = datetime.datetime.fromisoformat(ts_utc.replace("Z", "+00:00"))
                    if must_be_sorted and prev_utc_dt and dt_utc < prev_utc_dt:
                        results["chronologically_sorted"] = False
                        results["errors"].append(
                            f"Row {idx}: Out of chronological order. Prev: {prev_utc_dt}, Current: {dt_utc}"
                        )
                    prev_utc_dt = dt_utc
                except Exception as e:
                    results["timestamps_valid"] = False
                    results["errors"].append(f"Row {idx}: Invalid timestamp '{ts_utc}': {e}")

                # 2. Check coordinates
                try:
                    lat = float(row.get("latitude", 999))
                    lon = float(row.get("longitude", 999))
                    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                        results["coords_valid"] = False
                        results["errors"].append(f"Row {idx}: Coordinate out of bounds: lat={lat}, lon={lon}")

                    # 3. Check bounding box if specified
                    if target_box and not target_box.contains_point(lat, lon):
                        results["inside_bounding_box"] = False
                        results["errors"].append(
                            f"Row {idx}: Observation ({lat}, {lon}) outside {target_box.name} box"
                        )
                except Exception as e:
                    results["coords_valid"] = False
                    results["errors"].append(f"Row {idx}: Non-numeric coordinate: {e}")

                # 4. Check provenance
                if not row.get("granule_id") or not row.get("source_file"):
                    results["provenance_retained"] = False
                    results["errors"].append(f"Row {idx}: Missing provenance (granule_id or source_file)")

    except Exception as e:
        results["errors"].append(f"Failed to read CSV: {str(e)}")

    return results


def run_full_validation(config: Dict[str, Any], logger=None) -> Dict[str, Any]:
    """
    Execute complete end-to-end validation across all generated outputs:
    1. Consolidated files
    2. Daily partitions vs Consolidated agreement
    3. Manifest totals reconciliation
    4. Satellite coverage reconciliation
    Writes validation_report.json and validation_report.txt.
    """
    output_dir = config["output_dir"]
    chandipur_box, regional_box = get_bounding_boxes_from_config(config)

    combined_csv = os.path.join(output_dir, "combined", "iss_lis_all_observations.csv")
    chandipur_csv = os.path.join(output_dir, "chandipur", "chandipur_lightning_observations.csv")
    chandipur_dedup_csv = os.path.join(output_dir, "chandipur", "chandipur_lightning_observations_dedup.csv")
    regional_csv = os.path.join(output_dir, "regional", "regional_lightning_observations.csv")
    manifest_csv = os.path.join(output_dir, "reports", "processed_granules.csv")
    coverage_csv = os.path.join(output_dir, "chandipur", "chandipur_satellite_overpasses.csv")
    daily_dir = os.path.join(output_dir, "chandipur", "daily")

    if logger:
        logger.info("Running output validation checks...")

    # Validate Chandipur consolidated file
    chandipur_val = validate_csv_dataset(chandipur_csv, target_box=chandipur_box, must_be_sorted=True)
    combined_val = validate_csv_dataset(combined_csv, target_box=None, must_be_sorted=True)
    regional_val = validate_csv_dataset(regional_csv, target_box=regional_box, must_be_sorted=True)

    # Validate daily files and check agreement
    daily_files = glob.glob(os.path.join(daily_dir, "*.csv"))
    daily_files.sort()
    daily_row_count = 0
    daily_validations = []
    daily_dates = []

    for df in daily_files:
        d_val = validate_csv_dataset(df, target_box=chandipur_box, must_be_sorted=True)
        daily_validations.append(d_val)
        daily_row_count += d_val["row_count"]
        daily_dates.append(os.path.splitext(os.path.basename(df))[0])

    daily_agreement = (daily_row_count == chandipur_val["row_count"])

    # Reconcile with processed_granules.csv manifest
    manifest_reconciled = False
    manifest_flashes_sum = 0
    manifest_chandipur_sum = 0
    manifest_failed_granules = 0
    total_manifest_granules = 0

    if os.path.isfile(manifest_csv):
        with open(manifest_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_manifest_granules += 1
                manifest_flashes_sum += int(row.get("extracted_observations", 0))
                manifest_chandipur_sum += int(row.get("chandipur_observations", 0))
                if row.get("status") == "FAILED":
                    manifest_failed_granules += 1

        manifest_reconciled = (manifest_chandipur_sum == chandipur_val["row_count"]) and (
            manifest_flashes_sum == combined_val["row_count"]
        )

    # Satellite coverage stats
    total_overpasses = 0
    active_thunderstorm_overpasses = 0
    if os.path.isfile(coverage_csv):
        with open(coverage_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_overpasses += 1
                if int(row.get("is_thunderstorm_active", 0)) == 1:
                    active_thunderstorm_overpasses += 1

    overall_pass = (
        chandipur_val["timestamps_valid"]
        and chandipur_val["chronologically_sorted"]
        and chandipur_val["coords_valid"]
        and chandipur_val["inside_bounding_box"]
        and chandipur_val["provenance_retained"]
        and combined_val["timestamps_valid"]
        and combined_val["chronologically_sorted"]
        and daily_agreement
        and manifest_reconciled
    )

    warnings = []
    if total_overpasses == 0:
        warnings.append("No satellite overpasses were recorded over Chandipur.")
    if chandipur_val["row_count"] == 0:
        warnings.append("Zero lightning observations detected inside Chandipur bounding box.")
    warnings.append(
        "Coverage Note: ISS-LIS operated until November 16, 2023. Missing dates reflect instrument decommissioning and intermittent orbital sampling, NOT continuous absence of lightning."
    )

    report_data = {
        "overall_validation_passed": overall_pass,
        "chandipur_observations": {
            "file": chandipur_csv,
            "total_rows": chandipur_val["row_count"],
            "timestamps_valid": chandipur_val["timestamps_valid"],
            "chronologically_sorted": chandipur_val["chronologically_sorted"],
            "coords_valid": chandipur_val["coords_valid"],
            "inside_bounding_box": chandipur_val["inside_bounding_box"],
            "provenance_retained": chandipur_val["provenance_retained"],
        },
        "daily_reconciliation": {
            "total_daily_files": len(daily_files),
            "sum_daily_rows": daily_row_count,
            "consolidated_chandipur_rows": chandipur_val["row_count"],
            "daily_reconciliation_passed": daily_agreement,
            "active_dates": daily_dates,
        },
        "manifest_reconciliation": {
            "total_granules_in_manifest": total_manifest_granules,
            "failed_granules_count": manifest_failed_granules,
            "manifest_chandipur_flashes_sum": manifest_chandipur_sum,
            "manifest_global_flashes_sum": manifest_flashes_sum,
            "manifest_reconciliation_passed": manifest_reconciled,
        },
        "satellite_coverage_summary": {
            "total_overpasses_over_chandipur": total_overpasses,
            "overpasses_with_detected_lightning": active_thunderstorm_overpasses,
            "overpasses_with_zero_lightning": total_overpasses - active_thunderstorm_overpasses,
        },
        "global_dataset": {
            "file": combined_csv,
            "total_rows": combined_val["row_count"],
            "chronologically_sorted": combined_val["chronologically_sorted"],
        },
        "regional_dataset": {
            "file": regional_csv,
            "total_rows": regional_val["row_count"],
        },
        "warnings": warnings,
        "errors": chandipur_val["errors"] + combined_val["errors"],
    }

    # Save validation_report.json
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    json_path = os.path.join(reports_dir, "validation_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Save validation_report.txt
    txt_path = os.path.join(reports_dir, "validation_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("   NASA ISS-LIS DATA PROCESSING PIPELINE: VALIDATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"OVERALL VALIDATION STATUS: {'PASSED [OK]' if overall_pass else 'FAILED [ERROR]'}\n\n")

        f.write("1. DATASET INTEGRITY & ROW COUNTS:\n")
        f.write(f"   - Global observations (iss_lis_all_observations.csv): {combined_val['row_count']}\n")
        f.write(f"   - Chandipur observations (chandipur_lightning_observations.csv): {chandipur_val['row_count']}\n")
        f.write(f"   - Regional observations (regional_lightning_observations.csv): {regional_val['row_count']}\n")
        f.write(f"   - Daily Chandipur files count: {len(daily_files)} files (Sum rows: {daily_row_count})\n")
        f.write(f"   - Daily reconciliation with consolidated: {'MATCHED' if daily_agreement else 'MISMATCH'}\n\n")

        f.write("2. CHANDIPUR SPATIAL & TEMPORAL CONFORMANCE:\n")
        f.write(f"   - Coordinates valid (-90 to +90, -180 to +180): {chandipur_val['coords_valid']}\n")
        f.write(f"   - Strictly inside Chandipur box (21.25-21.50N, 87.00-87.25E): {chandipur_val['inside_bounding_box']}\n")
        f.write(f"   - UTC and IST Timestamps parseable: {chandipur_val['timestamps_valid']}\n")
        f.write(f"   - Chronologically sorted (timestamp_utc): {chandipur_val['chronologically_sorted']}\n")
        f.write(f"   - Full source provenance retained: {chandipur_val['provenance_retained']}\n\n")

        f.write("3. PROCESSING MANIFEST RECONCILIATION:\n")
        f.write(f"   - Total granules processed: {total_manifest_granules}\n")
        f.write(f"   - Failed granules: {manifest_failed_granules}\n")
        f.write(f"   - Manifest sum equals output row counts: {'YES' if manifest_reconciled else 'NO'}\n\n")

        f.write("4. SATELLITE COVERAGE RECONCILIATION:\n")
        f.write(f"   - Total verified overpasses over Chandipur: {total_overpasses}\n")
        f.write(f"   - Overpasses with lightning: {active_thunderstorm_overpasses}\n")
        f.write(f"   - Overpasses with 0 lightning (True Negative periods): {total_overpasses - active_thunderstorm_overpasses}\n\n")

        if warnings:
            f.write("WARNINGS & COVERAGE NOTES:\n")
            for w in warnings:
                f.write(f"   [!] {w}\n")
            f.write("\n")

        if report_data["errors"]:
            f.write("ERRORS ENCOUNTERED:\n")
            for err in report_data["errors"]:
                f.write(f"   [x] {err}\n")
            f.write("\n")

        f.write("=" * 70 + "\n")

    if logger:
        logger.info(f"Validation {'PASSED' if overall_pass else 'FAILED'}. Reports saved to {json_path} and {txt_path}")

    return report_data
