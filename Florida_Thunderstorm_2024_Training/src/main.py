"""Unified CLI Pipeline Runner for Florida Thunderstorm Prediction Dataset (2024).

Coordinates:
1. inspect: Audit existing project assets
2. download-noaa: Acquire/reuse 2024 NOAA Storm Events
3. download-era5: Interface with Copernicus CDS for ERA5
4. download-lis: Assess NASA LIS satellite records
5. download-glm: Retrieve NOAA GOES-16 GLM Level 2 LCFA data
6. process-noaa: Filter and clean Florida NOAA events
7. process-era5: Standardize and compute ERA5 meteorological features
8. process-lis: Standardize LIS observations with viewing flags
9. process-glm: Process GOES-16 GLM lightning observations
10. merge: Spatiotemporally align sources and construct training targets
11. validate: Execute comprehensive data quality & readiness suite
12. run-all: Complete automated pipeline execution
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure Florida project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.download_era5 import ERA5Downloader
from src.download_glm import GLMDownloader
from src.download_lis import LISDownloader
from src.download_noaa import NOAADownloader
from src.inspect_existing_data import run_inspection
from src.merge_datasets import DatasetMerger
from src.process_era5 import ERA5Processor
from src.process_glm import GLMProcessor
from src.process_lis import LISProcessor
from src.process_noaa import NOAAProcessor
from src.validate_dataset import DatasetValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "florida_pipeline.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("FloridaPipeline")


def run_all_pipeline(root: Path) -> None:
    """Run full end-to-end reproducible pipeline."""
    logger.info("==================================================================")
    logger.info("STARTING FLORIDA THUNDERSTORM 2024 COMPLETE PIPELINE")
    logger.info("==================================================================")

    # 1. Pre-execution Audit
    logger.info("STEP 1: INSPECTION & SOURCE AUDIT")
    run_inspection(parent_dir=root.parent, reports_dir=root / "reports")

    # 2. Ingest Sources
    logger.info("STEP 2: INGESTING / VERIFYING NOAA STORM EVENTS 2024")
    noaa_dl = NOAADownloader(
        raw_dest_dir=root / "data" / "raw" / "noaa",
        parent_raw_dir=root.parent / "data" / "raw",
    )
    noaa_files = noaa_dl.get_or_download_2024_files()

    logger.info("STEP 2b: INGESTING / VERIFYING ERA5 DATASETS")
    era5_dl = ERA5Downloader(
        raw_single_dir=root / "data" / "raw" / "era5" / "single_levels",
        raw_pressure_dir=root / "data" / "raw" / "era5" / "pressure_levels",
        parent_raw_dir=root.parent / "data" / "raw" / "era5_2024",
    )
    era5_file = era5_dl.import_existing_florida_pilot()

    logger.info("STEP 2c: ASSESSING NASA ISS LIS DATASETS")
    lis_dl = LISDownloader(
        raw_dest_dir=root / "data" / "raw" / "lis",
        parent_lis_dir=root.parent / "NASA LIS DATA",
    )
    lis_status = lis_dl.check_availability(2024)

    logger.info("STEP 2d: INGESTING NOAA GOES-16 GLM SATELLITE LIGHTNING DATA")
    glm_dl = GLMDownloader(raw_dir=root / "data" / "raw" / "glm")
    glm_files = glm_dl.retrieve_sample_granules(year=2024, day_of_year=136, hours=[0, 6, 12, 18], granules_per_hour=3)

    # 3. Process NOAA
    logger.info("STEP 3: PROCESSING NOAA FLORIDA EVENTS")
    noaa_proc = NOAAProcessor(
        raw_dir=root / "data" / "raw" / "noaa",
        boundary_file=root / "data" / "raw" / "geographic" / "florida_boundary.geojson",
        interim_dir=root / "data" / "interim" / "noaa",
        processed_dir=root / "data" / "processed",
    )
    df_noaa = noaa_proc.process_florida_events()

    # 4. Process ERA5
    logger.info("STEP 4: PROCESSING ERA5 MULTIDIMENSIONAL REANALYSIS")
    era5_proc = ERA5Processor(
        raw_single_dir=root / "data" / "raw" / "era5" / "single_levels",
        raw_pressure_dir=root / "data" / "raw" / "era5" / "pressure_levels",
        boundary_file=root / "data" / "raw" / "geographic" / "florida_boundary.geojson",
        interim_dir=root / "data" / "interim" / "era5",
        processed_dir=root / "data" / "processed",
    )
    df_era5 = era5_proc.process_to_dataframe()


    # 5. Process GOES-16 GLM Lightning Observations
    logger.info("STEP 5: PROCESSING GOES-16 GLM SATELLITE LIGHTNING OBSERVATIONS")
    glm_proc = GLMProcessor(
        raw_dir=root / "data" / "raw" / "glm",
        boundary_file=root / "data" / "raw" / "geographic" / "florida_boundary.geojson",
        interim_dir=root / "data" / "interim" / "glm",
        processed_dir=root / "data" / "processed",
    )
    df_glm = glm_proc.process_glm_dataset(target_grid=df_era5)

    # 6. Process LIS (Maintained for scientific audit)
    lis_proc = LISProcessor(
        raw_dir=root / "data" / "raw" / "lis",
        interim_dir=root / "data" / "interim" / "lis",
        processed_dir=root / "data" / "processed",
    )
    df_lis = lis_proc.process_lis_dataset(target_grid=df_era5)

    # 7. Merge Datasets
    logger.info("STEP 6: SPATIOTEMPORAL MERGE & TARGET FORMULATION")
    merger = DatasetMerger(
        processed_dir=root / "data" / "processed",
        training_dir=root / "data" / "training",
        reports_dir=root / "reports",
    )
    df_merged, merge_stats = merger.merge()

    # 8. Validate
    logger.info("STEP 7: DATASET INTEGRITY & VALIDATION SUITE")
    validator = DatasetValidator(
        training_csv=root / "data" / "training" / "florida_thunderstorm_training_2024.csv",
        training_parquet=root / "data" / "training" / "florida_thunderstorm_training_2024.parquet",
        reports_dir=root / "reports",
    )
    val_results = validator.run_all_validations()

    # 9. Update manifest
    manifest_csv = root / "reports" / "download_manifest.csv"
    import csv
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f_m:
        writer = csv.DictWriter(
            f_m,
            fieldnames=["source", "category", "filename", "path", "size_bytes", "sha256", "status", "timestamp", "notes"],
        )
        writer.writeheader()
        all_manifest = noaa_dl.manifest_records + era5_dl.manifest_records + glm_dl.manifest_records
        for r in all_manifest:
            writer.writerow({
                "source": r.get("source", ""),
                "category": r.get("category", ""),
                "filename": r.get("filename", ""),
                "path": r.get("path", ""),
                "size_bytes": r.get("size_bytes", 0),
                "sha256": r.get("sha256", ""),
                "status": r.get("status", ""),
                "timestamp": r.get("timestamp", ""),
                "notes": r.get("notes", ""),
            })

    logger.info("==================================================================")
    logger.info("PIPELINE EXECUTION FINISHED SUCCESSFULLY")
    logger.info(f"Master Merged Dataset: {root / 'data' / 'processed' / 'florida_merged_2024.csv'}")
    logger.info(f"Training CSV: {root / 'data' / 'training' / 'florida_thunderstorm_training_2024.csv'}")
    logger.info(f"Training Parquet: {root / 'data' / 'training' / 'florida_thunderstorm_training_2024.parquet'}")
    logger.info(f"Validation Result: {'ALL CHECKS PASSED' if val_results['all_passed'] else 'ISSUES FOUND'}")
    logger.info("==================================================================")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="FloridaPipeline",
        description="Florida Thunderstorm Prediction Dataset 2024 Pipeline",
    )
    parser.add_argument(
        "command",
        choices=[
            "inspect",
            "download-noaa",
            "download-era5",
            "download-lis",
            "download-glm",
            "process-noaa",
            "process-era5",
            "process-lis",
            "process-glm",
            "merge",
            "validate",
            "run-all",
        ],
        nargs="?",
        default="run-all",
        help="Pipeline stage to execute",
    )

    args = parser.parse_args()
    root = PROJECT_ROOT

    if args.command == "inspect":
        run_inspection(parent_dir=root.parent, reports_dir=root / "reports")
    elif args.command == "download-noaa":
        dl = NOAADownloader(raw_dest_dir=root / "data" / "raw" / "noaa", parent_raw_dir=root.parent / "data" / "raw")
        dl.get_or_download_2024_files()
    elif args.command == "download-era5":
        dl = ERA5Downloader(
            raw_single_dir=root / "data" / "raw" / "era5" / "single_levels",
            raw_pressure_dir=root / "data" / "raw" / "era5" / "pressure_levels",
            parent_raw_dir=root.parent / "data" / "raw" / "era5_2024",
        )
        dl.import_existing_florida_pilot()
    elif args.command == "download-lis":
        dl = LISDownloader(raw_dest_dir=root / "data" / "raw" / "lis", parent_lis_dir=root.parent / "NASA LIS DATA")
        dl.check_availability(2024)
    elif args.command == "download-glm":
        dl = GLMDownloader(raw_dir=root / "data" / "raw" / "glm")
        dl.retrieve_sample_granules()
    elif args.command == "process-noaa":
        proc = NOAAProcessor(
            raw_dir=root / "data" / "raw" / "noaa",
            boundary_file=root / "data" / "raw" / "geographic" / "florida_boundary.geojson",
            interim_dir=root / "data" / "interim" / "noaa",
            processed_dir=root / "data" / "processed",
        )
        proc.process_florida_events()
    elif args.command == "process-era5":
        proc = ERA5Processor(
            raw_dir=root / "data" / "raw" / "era5" / "single_levels",
            boundary_file=root / "data" / "raw" / "geographic" / "florida_boundary.geojson",
            interim_dir=root / "data" / "interim" / "era5",
            processed_dir=root / "data" / "processed",
        )
        proc.process_to_dataframe()
    elif args.command == "process-lis":
        era5_p = root / "data" / "processed" / "era5_florida_2024.csv"
        target_grid = pd.read_csv(era5_p) if era5_p.exists() else None
        proc = LISProcessor(
            raw_dir=root / "data" / "raw" / "lis",
            interim_dir=root / "data" / "interim" / "lis",
            processed_dir=root / "data" / "processed",
        )
        proc.process_lis_dataset(target_grid=target_grid)
    elif args.command == "process-glm":
        era5_p = root / "data" / "processed" / "era5_florida_2024.csv"
        target_grid = pd.read_csv(era5_p) if era5_p.exists() else None
        proc = GLMProcessor(
            raw_dir=root / "data" / "raw" / "glm",
            boundary_file=root / "data" / "raw" / "geographic" / "florida_boundary.geojson",
            interim_dir=root / "data" / "interim" / "glm",
            processed_dir=root / "data" / "processed",
        )
        proc.process_glm_dataset(target_grid=target_grid)
    elif args.command == "merge":
        merger = DatasetMerger(
            processed_dir=root / "data" / "processed",
            training_dir=root / "data" / "training",
            reports_dir=root / "reports",
        )
        merger.merge()
    elif args.command == "validate":
        validator = DatasetValidator(
            training_csv=root / "data" / "training" / "florida_thunderstorm_training_2024.csv",
            training_parquet=root / "data" / "training" / "florida_thunderstorm_training_2024.parquet",
            reports_dir=root / "reports",
        )
        validator.run_all_validations()
    elif args.command == "run-all":
        run_all_pipeline(root)


if __name__ == "__main__":
    main()
