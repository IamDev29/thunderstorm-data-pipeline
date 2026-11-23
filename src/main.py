"""Main CLI entrypoint for NOAA Storm Events Data Collection and Preprocessing Pipeline."""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from src.analysis import NOAAAnalyzer
from src.archive_parser import NOAAArchiveParser
from src.downloader import NOAADownloader
from src.geographic_filter import GeographicFilter
from src.preprocessing import NOAAPreprocessor
from src.reporting import NOAAReporter


def setup_pipeline_logging(config: Dict[str, Any]) -> None:
    """Configure logging format and handlers (console and log file)."""
    log_cfg = config.get("logging", {})
    level_name = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    log_file_str = log_cfg.get("log_file", "data/logs/noaa_pipeline.log")
    log_file = Path(log_file_str)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    handlers: List[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file), encoding="utf-8"),
    ]

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration dictionary from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class NOAADataPipeline:
    """Orchestrator for the automated NOAA Storm Events pipeline."""

    def __init__(self, config_path: Path = Path("config.yaml")):
        self.config_path = config_path
        self.config = load_yaml_config(config_path)
        setup_pipeline_logging(self.config)
        self.logger = logging.getLogger("NOAAPipeline")

        # Resolve directory paths
        dirs_cfg = self.config.get("directories", {})
        self.raw_dirs = {
            "details": Path(dirs_cfg.get("raw", {}).get("details", "data/raw/details")),
            "locations": Path(dirs_cfg.get("raw", {}).get("locations", "data/raw/locations")),
            "fatalities": Path(dirs_cfg.get("raw", {}).get("fatalities", "data/raw/fatalities")),
        }
        self.processed_dir = Path(dirs_cfg.get("processed", {}).get("root", "data/processed"))
        self.reports_dir = Path(dirs_cfg.get("reports", "data/reports"))
        self.logs_dir = Path(dirs_cfg.get("logs", "data/logs"))

        # Create directories
        for d in list(self.raw_dirs.values()) + [self.processed_dir, self.reports_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.archive_cfg = self.config.get("archive", {})
        self.parser = NOAAArchiveParser(
            base_url=self.archive_cfg.get("base_url", "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"),
            timeout=int(self.archive_cfg.get("timeout", 30)),
        )
        self.downloader = NOAADownloader(
            output_dirs=self.raw_dirs,
            reports_dir=self.reports_dir,
            timeout=int(self.archive_cfg.get("timeout", 30)),
            max_retries=int(self.archive_cfg.get("max_retries", 3)),
            backoff_factor=float(self.archive_cfg.get("backoff_factor", 2.0)),
        )
        self.preprocessor = NOAAPreprocessor()
        self.geo_filter = GeographicFilter(self.config)
        self.reporter = NOAAReporter(self.reports_dir)

    def run_download(
        self,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        dataset_types: Optional[List[str]] = None,
    ) -> List[Any]:
        """Download NOAA Storm Events bulk CSV files."""
        self.logger.info("=== STEP 1: DOWNLOADING NOAA STORM EVENTS DATA ===")

        # Determine year range
        if start_year is None or end_year is None:
            hist_cfg = self.config.get("historical_period", {})
            if hist_cfg.get("auto_detect_complete_years", True):
                try:
                    s_yr, e_yr = self.parser.get_complete_calendar_years(count=10)
                    start_year = start_year or s_yr
                    end_year = end_year or e_yr
                except Exception as exc:
                    self.logger.warning(f"Could not auto-detect complete years: {exc}. Using config values.")
                    start_year = start_year or int(hist_cfg.get("start_year", 2016))
                    end_year = end_year or int(hist_cfg.get("end_year", 2025))
            else:
                start_year = start_year or int(hist_cfg.get("start_year", 2016))
                end_year = end_year or int(hist_cfg.get("end_year", 2025))

        types = dataset_types or self.config.get("dataset_types", ["details", "locations", "fatalities"])
        self.logger.info(f"Target Period: {start_year} to {end_year} | Datasets: {types}")

        # Select files
        selected_files = self.parser.select_files(
            start_year=start_year, end_year=end_year, dataset_types=types
        )
        self.logger.info(f"Discovered {len(selected_files)} candidate files to download/verify.")

        # Download batch
        records = self.downloader.download_batch(selected_files)
        success_count = sum(1 for r in records if r.status in ("DOWNLOADED", "CACHED"))
        self.logger.info(f"Download complete: {success_count}/{len(records)} files successfully ready.")
        return records

    def run_preprocessing(self) -> Dict[str, pd.DataFrame]:
        """Process all raw files and save cleaned datasets."""
        self.logger.info("=== STEP 2: PREPROCESSING AND STANDARDIZATION ===")

        def load_and_clean_all(file_pattern: Path, clean_func) -> pd.DataFrame:
            files = sorted(list(file_pattern.glob("*.csv*")))
            if not files:
                self.logger.warning(f"No files found matching {file_pattern}")
                return pd.DataFrame()

            cleaned_dfs = []
            for f in files:
                try:
                    self.logger.info(f"Processing {f.name}...")
                    df = clean_func(f)
                    cleaned_dfs.append(df)
                except Exception as exc:
                    self.logger.error(f"Error processing {f.name}: {exc}")

            if not cleaned_dfs:
                return pd.DataFrame()
            return pd.concat(cleaned_dfs, ignore_index=True)

        # 1. Clean details
        details_df = load_and_clean_all(self.raw_dirs["details"], self.preprocessor.clean_details)
        # 2. Clean locations
        locations_df = load_and_clean_all(self.raw_dirs["locations"], self.preprocessor.clean_locations)
        # 3. Clean fatalities
        fatalities_df = load_and_clean_all(self.raw_dirs["fatalities"], self.preprocessor.clean_fatalities)

        # Save cleaned master datasets
        if not details_df.empty:
            p_details = self.processed_dir / "clean_details.csv"
            details_df.to_csv(p_details, index=False)
            if self.config.get("formats", {}).get("save_parquet", True):
                try:
                    details_df.to_parquet(self.processed_dir / "clean_details.parquet", index=False)
                except Exception as e:
                    self.logger.warning(f"Could not save parquet: {e}")

        if not locations_df.empty:
            p_locations = self.processed_dir / "clean_locations.csv"
            locations_df.to_csv(p_locations, index=False)

        if not fatalities_df.empty:
            p_fatalities = self.processed_dir / "clean_fatalities.csv"
            fatalities_df.to_csv(p_fatalities, index=False)

        self.logger.info(
            f"Cleaned datasets saved: Details ({len(details_df):,} rows), "
            f"Locations ({len(locations_df):,} rows), Fatalities ({len(fatalities_df):,} rows)"
        )

        return {
            "details": details_df,
            "locations": locations_df,
            "fatalities": fatalities_df,
        }

    def run_filtering(
        self,
        details_df: Optional[pd.DataFrame] = None,
        locations_df: Optional[pd.DataFrame] = None,
        fatalities_df: Optional[pd.DataFrame] = None,
        regions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Filter cleaned datasets by US candidate regions and severe event types."""
        self.logger.info("=== STEP 3: GEOGRAPHIC & EVENT-TYPE FILTERING ===")

        if details_df is None:
            p_det = self.processed_dir / "clean_details.csv"
            if not p_det.exists():
                raise FileNotFoundError(f"Clean details not found at {p_det}. Run 'process' first.")
            details_df = pd.read_csv(p_det, low_memory=False)

        if locations_df is None:
            p_loc = self.processed_dir / "clean_locations.csv"
            locations_df = pd.read_csv(p_loc, low_memory=False) if p_loc.exists() else pd.DataFrame()

        if fatalities_df is None:
            p_fat = self.processed_dir / "clean_fatalities.csv"
            fatalities_df = pd.read_csv(p_fat, low_memory=False) if p_fat.exists() else pd.DataFrame()

        filter_results = self.geo_filter.process_all_regions(details_df)

        # Output regional datasets
        dirs_cfg = self.config.get("directories", {}).get("processed", {})
        target_regions = regions or ["florida", "texas", "louisiana", "mississippi"]

        for region in target_regions:
            df_reg = filter_results.get(region)
            if df_reg is not None and not df_reg.empty:
                reg_dir = Path(dirs_cfg.get(region, f"data/processed/{region}"))
                reg_dir.mkdir(parents=True, exist_ok=True)

                out_path = reg_dir / f"details_{region}.csv"
                df_reg.to_csv(out_path, index=False)
                self.logger.info(f"Saved {region} dataset ({len(df_reg):,} events) to {out_path}")

                # Save associated locations and fatalities
                event_ids = set(df_reg["event_id"].dropna().unique())
                if not locations_df.empty:
                    reg_locs = self.geo_filter.filter_associated_locations(locations_df, event_ids)
                    reg_locs.to_csv(reg_dir / f"locations_{region}.csv", index=False)

                if not fatalities_df.empty:
                    reg_fats = self.geo_filter.filter_associated_fatalities(fatalities_df, event_ids)
                    reg_fats.to_csv(reg_dir / f"fatalities_{region}.csv", index=False)

        # Combined Gulf Coast
        combined_df = filter_results.get("combined_gulf_coast", pd.DataFrame())
        combined_dir = Path(dirs_cfg.get("combined", "data/processed/combined"))
        combined_dir.mkdir(parents=True, exist_ok=True)
        if not combined_df.empty:
            out_c = combined_dir / "details_gulf_coast.csv"
            combined_df.to_csv(out_c, index=False)
            self.logger.info(f"Saved combined Gulf Coast dataset ({len(combined_df):,} events) to {out_c}")

            # Associated locations & fatalities for combined
            c_event_ids = set(combined_df["event_id"].dropna().unique())
            if not locations_df.empty:
                c_locs = self.geo_filter.filter_associated_locations(locations_df, c_event_ids)
                c_locs.to_csv(combined_dir / "locations_gulf_coast.csv", index=False)
            if not fatalities_df.empty:
                c_fats = self.geo_filter.filter_associated_fatalities(fatalities_df, c_event_ids)
                c_fats.to_csv(combined_dir / "fatalities_gulf_coast.csv", index=False)

        # Thunderstorm-specific dataset
        ts_df = filter_results.get("thunderstorm_events_combined", pd.DataFrame())
        if not ts_df.empty:
            out_ts = combined_dir / "thunderstorm_events_gulf_coast.csv"
            ts_df.to_csv(out_ts, index=False)
            self.logger.info(
                f"Saved convective/thunderstorm events dataset ({len(ts_df):,} events) to {out_ts}"
            )

        return filter_results

    def run_reporting(
        self,
        details_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Generate statistical summaries, quality reports, and visualizations."""
        self.logger.info("=== STEP 4: DATA QUALITY ANALYSIS & REPORTING ===")

        if details_df is None:
            # Prefer combined gulf coast or clean_details
            p_comb = self.processed_dir / "combined" / "details_gulf_coast.csv"
            p_clean = self.processed_dir / "clean_details.csv"
            if p_comb.exists():
                details_df = pd.read_csv(p_comb, low_memory=False)
            elif p_clean.exists():
                details_df = pd.read_csv(p_clean, low_memory=False)
            else:
                raise FileNotFoundError("No processed dataset found to generate report from.")

        analyzer = NOAAAnalyzer(details_df, self.config)

        # Populate manifest records from disk if empty
        manifest_records = self.downloader.manifest_records
        manifest_json_path = self.reports_dir / "download_manifest.json"
        if not manifest_records and manifest_json_path.exists():
            try:
                import json
                with open(manifest_json_path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                    # Simple object wrapper
                    class Obj:
                        def __init__(self, d):
                            self.__dict__.update(d)
                    manifest_records = [Obj(d) for d in manifest_data]
            except Exception as e:
                self.logger.warning(f"Could not load manifest from disk: {e}")

        # Check total raw records from clean_details.csv if available
        raw_count = self.preprocessor.quality_stats.get("total_raw_records", 0)
        p_clean = self.processed_dir / "clean_details.csv"
        if raw_count == 0 and p_clean.exists():
            try:
                raw_count = len(pd.read_csv(p_clean, usecols=["event_id"]))
            except Exception:
                raw_count = len(details_df)
        if raw_count == 0:
            raw_count = len(details_df)

        # Compile quality summary
        summary = analyzer.get_comprehensive_quality_summary(
            raw_count=raw_count,
            manifest_records=manifest_records,
            preprocessor_stats=self.preprocessor.quality_stats,
            geo_diagnostics=self.geo_filter.filter_diagnostics,
        )

        # Generate markdown report
        md_path = self.reporter.generate_markdown_report(summary, analyzer)
        # Generate JSON summary
        json_path = self.reporter.save_json_summary(summary)
        # Generate summary CSVs
        csv_paths = self.reporter.save_summary_csvs(analyzer)
        # Generate visualizations
        fig_paths = self.reporter.generate_visualizations(details_df, analyzer)

        self.logger.info(f"Report generated: {md_path}")
        self.logger.info(f"Visualizations saved: {[p.name for p in fig_paths]}")

        return summary

    def run_all(
        self,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        dataset_types: Optional[List[str]] = None,
    ) -> None:
        """Run the end-to-end automated pipeline."""
        self.run_download(start_year, end_year, dataset_types)
        cleaned = self.run_preprocessing()
        self.run_filtering(cleaned["details"], cleaned["locations"], cleaned["fatalities"])
        self.run_reporting(cleaned["details"])
        self.logger.info("End-to-end pipeline run completed successfully.")


def build_cli_parser() -> argparse.ArgumentParser:
    """Build command-line interface arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="Automated NOAA Storm Events Data Collection and Preprocessing Pipeline",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Pipeline subcommands")

    # 1. Download
    p_dl = subparsers.add_parser("download", help="Download raw NOAA Storm Events archive files")
    p_dl.add_argument("--start-year", type=int, default=None, help="Start year (default: from config)")
    p_dl.add_argument("--end-year", type=int, default=None, help="End year (default: from config)")
    p_dl.add_argument(
        "--types",
        nargs="+",
        default=None,
        choices=["details", "locations", "fatalities"],
        help="Datasets to download",
    )

    # 2. Process
    subparsers.add_parser("process", help="Clean and standardize downloaded NOAA datasets")

    # 3. Filter
    p_fl = subparsers.add_parser("filter", help="Apply geographic and event type filters")
    p_fl.add_argument(
        "--regions",
        nargs="+",
        default=["florida", "texas", "louisiana", "mississippi"],
        help="Regions to filter (e.g. florida texas louisiana mississippi)",
    )

    # 4. Report
    subparsers.add_parser("report", help="Generate quality reports, CSVs, and visualizations")

    # 5. Run-All
    p_all = subparsers.add_parser("run-all", help="Execute complete end-to-end pipeline")
    p_all.add_argument("--start-year", type=int, default=None, help="Start year")
    p_all.add_argument("--end-year", type=int, default=None, help="End year")
    p_all.add_argument("--types", nargs="+", default=None, help="Datasets to download")

    # 6. ERA5 Download
    p_e_dl = subparsers.add_parser("era5-download", help="Download ERA5 single levels data for 2024 pilot")
    p_e_dl.add_argument("--regions", nargs="+", default=None, help="Regions to download (default: all configured)")
    p_e_dl.add_argument("--months", nargs="+", default=None, help="Months to download (e.g. 05 07 10)")

    # 7. ERA5 Process
    subparsers.add_parser("era5-process", help="Process and standardize downloaded ERA5 NetCDF files")

    # 8. ERA5 Similarity
    subparsers.add_parser("era5-similarity", help="Compute seasonal statistics, climate similarity scores, and generate report")

    # 9. ERA5 Pilot (End-to-End)
    p_e_all = subparsers.add_parser("era5-pilot", help="Run complete ERA5 2024 pilot study pipeline")
    p_e_all.add_argument("--regions", nargs="+", default=None, help="Regions to evaluate")
    p_e_all.add_argument("--months", nargs="+", default=None, help="Months to evaluate")

    return parser


def run_era5_pipeline(config_path: Path, command: str, args: Any) -> None:
    """Execute ERA5 pilot subcommands."""
    from src.era5_downloader import ERA5Downloader
    from src.era5_preprocessing import ERA5Preprocessor
    from src.era5_analysis import ERA5Analyzer
    from src.era5_reporting import ERA5Reporter
    import xarray as xr

    config = load_yaml_config(config_path)
    era5_cfg = config.get("era5_pilot", {})
    raw_dir = Path(era5_cfg.get("directories", {}).get("raw", "data/raw/era5_2024"))
    processed_dir = Path(era5_cfg.get("directories", {}).get("processed", "data/processed/era5_2024"))
    reports_dir = Path(era5_cfg.get("directories", {}).get("reports", "data/reports/climate_similarity_2024"))

    regions = getattr(args, "regions", None) or list(era5_cfg.get("domains", {}).keys())
    months = getattr(args, "months", None) or [f"{m:02d}" for m in range(1, 13)]

    downloader = ERA5Downloader(config, raw_base_dir=raw_dir, reports_dir=reports_dir)
    preprocessor = ERA5Preprocessor(processed_dir=processed_dir)
    reporter = ERA5Reporter(reports_dir=reports_dir)

    if command in ("era5-download", "era5-pilot"):
        print("=== STEP 1: DOWNLOADING ERA5 2024 ATMOSPHERIC DATA ===")
        for reg in regions:
            for m in months:
                downloader.download_region_month(reg, m)
        downloader.save_manifest()

    if command in ("era5-process", "era5-pilot"):
        print("=== STEP 2: PREPROCESSING ERA5 MULTIDIMENSIONAL DATASETS ===")
        for reg in regions:
            reg_files = list((raw_dir / reg).glob("*.nc"))
            if reg_files:
                preprocessor.load_and_process_region_files(reg, reg_files)

    if command in ("era5-similarity", "era5-pilot"):
        print("=== STEP 3: COMPUTING CLIMATE SIMILARITY SCORES & REPORTING ===")
        datasets = {}
        for reg in regions:
            p_nc = processed_dir / f"era5_processed_{reg}_2024.nc"
            if p_nc.exists():
                datasets[reg] = xr.open_dataset(p_nc)

        if not datasets:
            print("No processed ERA5 datasets found to analyze.")
            return

        analyzer = ERA5Analyzer(datasets, config)
        stats_df = analyzer.compute_all_regional_seasonal_stats()
        sim_df, rankings = analyzer.compute_climate_similarity(ref_region="chandipur")

        reporter.save_summary_csvs(stats_df, sim_df)
        fig_paths = reporter.generate_visualizations(stats_df, sim_df)
        md_report = reporter.generate_markdown_report(stats_df, sim_df, rankings, downloader.manifest_records)

        print(f"Climate Similarity Report generated at: {md_report}")
        print(f"Visualizations saved: {[p.name for p in fig_paths]}")


def main() -> None:
    """Main CLI entry point."""
    parser = build_cli_parser()
    args = parser.parse_args()

    config_path = Path(args.config)

    if args.command in ("era5-download", "era5-process", "era5-similarity", "era5-pilot"):
        run_era5_pipeline(config_path, args.command, args)
        return

    pipeline = NOAADataPipeline(config_path)

    if args.command == "download":
        pipeline.run_download(
            start_year=args.start_year,
            end_year=args.end_year,
            dataset_types=args.types,
        )
    elif args.command == "process":
        pipeline.run_preprocessing()
    elif args.command == "filter":
        pipeline.run_filtering(regions=args.regions)
    elif args.command == "report":
        pipeline.run_reporting()
    elif args.command == "run-all":
        pipeline.run_all(
            start_year=args.start_year,
            end_year=args.end_year,
            dataset_types=args.types,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

