import argparse
import sys
from pathlib import Path

from scripts.audit_engine import run_full_audit
from scripts.cleanup_hdf import evaluate_hdf_cleanup
from scripts.combiner import combine_and_save_all
from scripts.config_manager import load_config, setup_logging
from scripts.extractor import run_extraction_pipeline
from scripts.inspect_schema import run_schema_inspection
from scripts.inventory import perform_inventory
from scripts.validator import run_full_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="NASA ISS-LIS Lightning Data Processing Pipeline for Thunderstorm Prediction",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.json file")
    parser.add_argument("--input-dir", type=str, default=None, help="Override NASA input directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")

    subparsers = parser.add_subparsers(dest="command", help="Pipeline subcommands")

    # 1. Inventory
    subparsers.add_parser("inventory", help="Inspect and catalog all .nc and .hdf granules")

    # 2. Inspect NetCDF schema
    subparsers.add_parser("inspect", help="Inspect NetCDF data model, groups, dimensions, and variables")

    # 3. Cleanup HDF files safely
    cleanup_parser = subparsers.add_parser("cleanup-hdf", help="Safely cleanup redundant .hdf granules")
    cleanup_group = cleanup_parser.add_mutually_exclusive_group()
    cleanup_group.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Dry run mode (default: verifies eligibility, deletes nothing)",
    )
    cleanup_group.add_argument(
        "--delete-hdf",
        action="store_true",
        default=False,
        help="DESTRUCTIVE: Delete verified eligible .hdf files only",
    )

    # 4. Extract
    extract_parser = subparsers.add_parser("extract", help="Extract lightning observations from NetCDF files")
    extract_parser.add_argument("--year", type=int, default=None, help="Calendar year to extract (default: 2023)")
    extract_parser.add_argument(
        "--level",
        type=str,
        choices=["flash", "group", "event"],
        default=None,
        help="Observation level (default: flash)",
    )
    extract_parser.add_argument("--limit", type=int, default=None, help="Limit number of granules (for testing)")

    # 5. Filter
    filter_parser = subparsers.add_parser("filter", help="Geographically filter observations")
    filter_parser.add_argument(
        "--region",
        type=str,
        choices=["chandipur", "regional", "all"],
        default="chandipur",
        help="Target region for filtering",
    )

    # 6. Combine
    combine_parser = subparsers.add_parser("combine", help="Combine observations into chronological CSV datasets")
    combine_parser.add_argument("--year", type=int, default=None, help="Calendar year")

    # 7. Validate
    subparsers.add_parser("validate", help="Run comprehensive data integrity and reconciliation validation")

    # 8. Audit
    subparsers.add_parser("audit", help="Run independent scientific audit of lightning, coverage, and labels")

    # 9. Run All
    run_all_parser = subparsers.add_parser("run-all", help="Execute complete pipeline end-to-end")
    run_all_parser.add_argument("--year", type=int, default=None, help="Calendar year (default: 2023)")
    run_all_parser.add_argument("--region", type=str, default="chandipur", help="Study region name")
    run_all_parser.add_argument("--limit", type=int, default=None, help="Limit granules for test execution")
    run_all_parser.add_argument(
        "--delete-hdf",
        action="store_true",
        default=False,
        help="Include safe deletion of eligible .hdf files during run-all",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    overrides = {}
    if args.input_dir:
        overrides["input_dir"] = args.input_dir
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if getattr(args, "year", None):
        overrides["year"] = args.year
        overrides["start_date"] = f"{args.year}-01-01"
        overrides["end_date"] = f"{args.year}-12-31"
    if getattr(args, "level", None):
        overrides["observation_level"] = args.level

    config_path = Path(args.config) if args.config else None
    config = load_config(config_path, overrides)
    logger = setup_logging(config["output_dir"])

    logger.info(f"Command '{args.command}' initiated with input: {config['input_dir']}")

    if args.command == "inventory":
        report = perform_inventory(config["input_dir"], config["output_dir"], logger)
        print("\n" + "=" * 60)
        print("          NASA ISS-LIS INVENTORY SUMMARY")
        print("=" * 60)
        print(f"Total NetCDF (.nc) files: {report['total_nc_files']} ({report['total_nc_size_gb']} GB)")
        print(f"Total HDF (.hdf) files:    {report['total_hdf_files']} ({report['total_hdf_size_gb']} GB)")
        print(f"Paired Granule Formats:   {report['granule_pairing']['paired_granules_count']}")
        print(f"Date Range Inferred:      {report['date_range']['earliest_date']} to {report['date_range']['latest_date']}")
        print(f"Reports Written to:       {config['output_dir']}/reports/")
        print("=" * 60 + "\n")

    elif args.command == "inspect":
        report = run_schema_inspection(config["input_dir"], config["output_dir"], logger)
        print("\n" + "=" * 60)
        print("        NASA ISS-LIS NETCDF SCHEMA INSPECTION")
        print("=" * 60)
        print(f"Total Variables:     {report['total_variables']}")
        print(f"Schema Consistent:   {report['schema_consistency']}")
        print(f"Report Written to:   {config['output_dir']}/reports/netcdf_schema_report.md")
        print("=" * 60 + "\n")

    elif args.command == "cleanup-hdf":
        dry_run = not args.delete_hdf
        summary = evaluate_hdf_cleanup(config["input_dir"], config["output_dir"], dry_run=dry_run, logger=logger)
        print("\n" + "=" * 60)
        print("            NASA ISS-LIS HDF CLEANUP")
        print("=" * 60)
        print(f"Execution Mode:       {'DRY RUN (no deletions performed)' if dry_run else 'ACTIVE DELETION'}")
        print(f"Total HDF Granules:   {summary['total_hdf_files']}")
        print(f"Eligible for Deletion: {summary['eligible_for_deletion']}")
        print(f"Reclaimable Space:    {summary['reclaimable_gb']} GB")
        if not dry_run:
            print(f"Files Deleted:        {summary['deleted_count']}")
        print(f"Manifest written to:  {summary['manifest_path']}")
        print(f"Report written to:    {summary['report_path']}")
        print("=" * 60 + "\n")

    elif args.command == "extract":
        summary = run_extraction_pipeline(config, limit_files=args.limit, logger=logger)
        combine_and_save_all(
            summary["all_observations"],
            summary["chandipur_observations"],
            summary["regional_observations"],
            config["output_dir"],
            logger=logger,
        )

    elif args.command == "filter" or args.command == "combine":
        logger.info(f"Re-running extraction and combination for {config.get('year', 2023)}...")
        summary = run_extraction_pipeline(config, logger=logger)
        combine_and_save_all(
            summary["all_observations"],
            summary["chandipur_observations"],
            summary["regional_observations"],
            config["output_dir"],
            logger=logger,
        )

    elif args.command == "validate":
        report = run_full_validation(config, logger=logger)
        status_str = "PASSED [OK]" if report["overall_validation_passed"] else "FAILED [ERROR]"
        print("\n" + "=" * 60)
        print(f"       VALIDATION STATUS: {status_str}")
        print("=" * 60)
        print(f"Global observations:    {report['global_dataset']['total_rows']}")
        print(f"Chandipur observations: {report['chandipur_observations']['total_rows']}")
        print(f"Daily partitions count: {report['daily_reconciliation']['total_daily_files']}")
        print(f"Daily rows match:       {report['daily_reconciliation']['daily_reconciliation_passed']}")
        print(f"Reports written to:     {config['output_dir']}/reports/validation_report.[json|txt]")
        print("=" * 60 + "\n")

    elif args.command == "audit":
        base_dir = str(Path(config["output_dir"]).parent)
        run_full_audit(base_dir)

    elif args.command == "run-all":
        logger.info("Executing Complete Pipeline (run-all)...")

        # 1. Inventory
        logger.info("Step 1/5: Running Inventory...")
        perform_inventory(config["input_dir"], config["output_dir"], logger)

        # 2. Schema Inspection
        logger.info("Step 2/5: Inspecting NetCDF Schema...")
        run_schema_inspection(config["input_dir"], config["output_dir"], logger)

        # 3. Extraction & Coverage
        logger.info("Step 3/5: Extracting Lightning & Satellite Coverage...")
        summary = run_extraction_pipeline(config, limit_files=args.limit, logger=logger)

        # 4. Combine & Deduplicate
        logger.info("Step 4/5: Combining Datasets and Generating Daily Partitions...")
        combine_and_save_all(
            summary["all_observations"],
            summary["chandipur_observations"],
            summary["regional_observations"],
            config["output_dir"],
            logger=logger,
        )

        # 5. Validation
        logger.info("Step 5/5: Running Data Validation...")
        val_report = run_full_validation(config, logger=logger)

        # Optional Cleanup if explicitly requested
        if getattr(args, "delete_hdf", False):
            logger.info("Optional: Running active HDF cleanup as requested with --delete-hdf...")
            evaluate_hdf_cleanup(config["input_dir"], config["output_dir"], dry_run=False, logger=logger)
        else:
            logger.info("Running HDF cleanup in DRY RUN mode (source files preserved)...")
            evaluate_hdf_cleanup(config["input_dir"], config["output_dir"], dry_run=True, logger=logger)

        print("\n" + "=" * 70)
        print("        PIPELINE EXECUTION COMPLETE (run-all)")
        print("=" * 70)
        print(f"Status:                 {'ALL CHECKS PASSED [OK]' if val_report['overall_validation_passed'] else 'COMPLETED WITH WARNINGS'}")
        print(f"Global Observations:    {summary['total_observations_extracted']}")
        print(f"Chandipur Observations: {summary['chandipur_observations_count']}")
        print(f"Chandipur Overpasses:   {summary['chandipur_overpasses_count']}")
        print(f"Regional Observations:  {summary['regional_observations_count']}")
        print(f"Daily Partitions:       {val_report['daily_reconciliation']['total_daily_files']} days")
        print(f"Output Directory:       {config['output_dir']}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
