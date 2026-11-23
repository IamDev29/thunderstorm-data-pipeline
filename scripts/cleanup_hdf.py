import csv
import glob
import os
from pathlib import Path
from typing import Any, Dict, List

import netCDF4


def evaluate_hdf_cleanup(input_dir: str, output_dir: str, dry_run: bool = True, logger=None) -> Dict[str, Any]:
    """
    Evaluate all .hdf files in input_dir for safe deletion eligibility.
    - Requires corresponding .nc file exists
    - Requires .nc file is non-empty and openable with netCDF4
    - Requires HDF file is strictly within input_dir
    - By default dry_run is True (no files deleted)
    - Generates cleanup manifest and markdown report
    """
    input_dir_abs = os.path.abspath(input_dir)
    hdf_files = glob.glob(os.path.join(input_dir, "**", "*.hdf"), recursive=True)
    hdf_files.sort()

    if logger:
        logger.info(f"Evaluating {len(hdf_files)} HDF files for cleanup safety (dry_run={dry_run})...")

    manifest_rows: List[Dict[str, Any]] = []
    total_eligible = 0
    total_ineligible = 0
    reclaimable_bytes = 0
    deleted_count = 0
    deleted_bytes = 0
    failed_deletions = []

    for hdf_path in hdf_files:
        hdf_abs = os.path.abspath(hdf_path)
        hdf_fname = os.path.basename(hdf_abs)
        hdf_size = os.path.getsize(hdf_abs)

        # Check path safety constraint
        is_inside_input = os.path.commonpath([hdf_abs, input_dir_abs]) == input_dir_abs
        is_hdf_ext = hdf_abs.lower().endswith(".hdf")

        # Determine corresponding NC file
        stem = os.path.splitext(hdf_abs)[0]
        nc_abs = stem + ".nc"
        nc_exists = os.path.isfile(nc_abs)
        nc_size = os.path.getsize(nc_abs) if nc_exists else 0

        nc_valid = False
        rejection_reason = ""

        if not is_inside_input:
            rejection_reason = "Security check failed: File is outside input directory"
        elif not is_hdf_ext:
            rejection_reason = "Security check failed: File does not have .hdf extension"
        elif not nc_exists:
            rejection_reason = "Missing corresponding .nc file"
        elif nc_size == 0:
            rejection_reason = "Corresponding .nc file is 0 bytes (corrupt/empty)"
        else:
            # Verify that NC file can be opened and has dimensions
            try:
                with netCDF4.Dataset(nc_abs, "r") as ds:
                    if len(ds.dimensions) > 0:
                        nc_valid = True
                    else:
                        rejection_reason = "Corresponding .nc file has no dimensions"
            except Exception as e:
                rejection_reason = f"Corresponding .nc file failed opening: {str(e)}"

        eligible = (
            is_inside_input
            and is_hdf_ext
            and nc_exists
            and (nc_size > 0)
            and nc_valid
        )

        action = "none"
        if eligible:
            total_eligible += 1
            reclaimable_bytes += hdf_size
            if not dry_run:
                try:
                    os.remove(hdf_abs)
                    action = "deleted"
                    deleted_count += 1
                    deleted_bytes += hdf_size
                except Exception as e:
                    action = f"failed_delete: {str(e)}"
                    failed_deletions.append({"file": hdf_abs, "error": str(e)})
            else:
                action = "dry_run_eligible"
        else:
            total_ineligible += 1
            action = f"skipped: {rejection_reason}"

        manifest_rows.append({
            "hdf_path": hdf_abs,
            "hdf_filename": hdf_fname,
            "hdf_size_bytes": hdf_size,
            "nc_path": nc_abs,
            "nc_exists": nc_exists,
            "nc_size_bytes": nc_size,
            "nc_valid": nc_valid,
            "eligible_for_deletion": eligible,
            "action_taken": action,
            "reason": rejection_reason if not eligible else "Fully verified matching NetCDF granule",
        })

    # Write manifest CSV
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    manifest_csv = os.path.join(reports_dir, "hdf_cleanup_manifest.csv")

    fieldnames = [
        "hdf_filename",
        "hdf_size_bytes",
        "eligible_for_deletion",
        "nc_exists",
        "nc_valid",
        "nc_size_bytes",
        "action_taken",
        "reason",
        "hdf_path",
        "nc_path",
    ]
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Write summary Markdown report
    report_md_path = os.path.join(reports_dir, "hdf_cleanup_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# NASA ISS-LIS HDF Safe Cleanup Report\n\n")
        f.write(f"- **Execution Mode**: `{'DRY RUN (No files modified or deleted)' if dry_run else 'ACTIVE DELETION (--delete-hdf enabled)'}`\n")
        f.write(f"- **Total HDF Files Analyzed**: {len(hdf_files)}\n")
        f.write(f"- **Eligible for Safe Deletion**: {total_eligible} (100% paired with verified NetCDF granules)\n")
        f.write(f"- **Ineligible / Skipped**: {total_ineligible}\n")
        f.write(f"- **Reclaimable Storage Space**: {round(reclaimable_bytes / (1024 * 1024), 2)} MB ({round(reclaimable_bytes / (1024 * 1024 * 1024), 2)} GB)\n")
        if not dry_run:
            f.write(f"- **Files Successfully Deleted**: {deleted_count}\n")
            f.write(f"- **Storage Space Reclaimed**: {round(deleted_bytes / (1024 * 1024), 2)} MB ({round(deleted_bytes / (1024 * 1024 * 1024), 2)} GB)\n")
            if failed_deletions:
                f.write(f"- **Failed Deletions**: {len(failed_deletions)}\n")
                for fd in failed_deletions:
                    f.write(f"  - `{fd['file']}`: {fd['error']}\n")
        else:
            f.write("\n> [!NOTE]\n")
            f.write("> Dry-run mode completed. No `.hdf` or `.nc` files were deleted. To perform the actual deletion, run:\n")
            f.write("> ```powershell\n> python main.py cleanup-hdf --delete-hdf\n> ```\n")

    summary = {
        "dry_run": dry_run,
        "total_hdf_files": len(hdf_files),
        "eligible_for_deletion": total_eligible,
        "ineligible": total_ineligible,
        "reclaimable_bytes": reclaimable_bytes,
        "reclaimable_mb": round(reclaimable_bytes / (1024 * 1024), 2),
        "reclaimable_gb": round(reclaimable_bytes / (1024 * 1024 * 1024), 2),
        "deleted_count": deleted_count,
        "deleted_bytes": deleted_bytes,
        "manifest_path": manifest_csv,
        "report_path": report_md_path,
    }

    if logger:
        logger.info(
            f"HDF cleanup analysis: {total_eligible}/{len(hdf_files)} eligible. "
            f"Reclaimable: {summary['reclaimable_gb']} GB. "
            f"Mode: {'DRY RUN' if dry_run else 'DELETED ' + str(deleted_count) + ' files'}."
        )

    return summary
