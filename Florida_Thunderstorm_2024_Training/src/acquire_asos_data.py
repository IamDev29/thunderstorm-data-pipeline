"""
acquire_asos_data.py
====================
Automated historical ASOS/METAR acquisition pipeline for Florida in 2024.

Features:
- Automated station inventory retrieval from Iowa Environmental Mesonet (IEM) FL_ASOS network.
- Identification and spatial filtering of stations operating during 2024.
- Concurrency-safe, rate-limited, retry-safe acquisition across full year 2024 in 5-day batches.
- Parsing of original METAR text for TS, VCTS, and lightning remarks (LTG, TSB, TSE, CB).
- Precise UTC timestamp handling and snapping to half-open (t, t+6h] forecasting windows.
- Strict coverage status accounting (preventing missing reports from being imputed as zero).
- Persistent manifest tracking with automatic resumption and deduplication.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("acquire_asos_data")

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
IEM_GEOJSON_URL = "https://mesonet.agron.iastate.edu/geojson/network/FL_ASOS.geojson"
DEFAULT_USER_AGENT = "FloridaThunderstorm2024Research/1.0"


from src.parse_metar_robust import parse_metar_thunderstorm_robust


def parse_metar_thunderstorm(metar_text: str, wxcodes_text: str = "", station_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Robust METAR parser that isolates body from remarks and excludes station callsigns.
    """
    res = parse_metar_thunderstorm_robust(metar_text, wxcodes_text, station_id=station_id)
    return {
        "has_ts": res["has_thunderstorm_ts"],
        "has_vcts": res["has_vicinity_thunderstorm_vcts"],
        "has_lightning_remark": res["has_genuine_lightning_remark"] or res["has_cumulonimbus_cb"],
        "thunderstorm_reported": res["strict_thunderstorm_reported"] or res["has_cumulonimbus_cb"],
        "lightning_remarks": res["lightning_remarks"],
        "has_thunderstorm_ts": res["has_thunderstorm_ts"],
        "has_vicinity_thunderstorm_vcts": res["has_vicinity_thunderstorm_vcts"],
        "has_genuine_lightning_remark": res["has_genuine_lightning_remark"],
        "has_cumulonimbus_cb": res["has_cumulonimbus_cb"],
        "strict_thunderstorm_reported": res["strict_thunderstorm_reported"],
        "cb_alone": res["cb_alone"],
    }


class ASOSAcquisitionPipeline:
    """Manages the full-year Florida ASOS data acquisition and harmonization workflow."""

    def __init__(
        self,
        output_dir: Path = Path("data/processed/asos"),
        active_grid_reference_path: Optional[Path] = Path("data/processed/florida_thunderstorm_2024_forecasting_6h.csv"),
    ):
        self.output_dir = output_dir
        self.raw_batches_dir = output_dir / "raw_batches"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_batches_dir.mkdir(parents=True, exist_ok=True)

        self.active_grid_reference_path = active_grid_reference_path
        self._active_grid_cells: Set[Tuple[float, float]] = set()
        self._load_active_grid()

        self.stations_df: Optional[pd.DataFrame] = None
        self.manifest_path = self.output_dir / "asos_download_manifest.csv"

    def _load_active_grid(self) -> None:
        """Load 218 terrestrial active Florida grid cells."""
        if self.active_grid_reference_path and self.active_grid_reference_path.exists():
            df_ref = pd.read_csv(self.active_grid_reference_path, usecols=["latitude", "longitude"]).drop_duplicates()
            for _, r in df_ref.iterrows():
                self._active_grid_cells.add((round(float(r["latitude"]), 4), round(float(r["longitude"]), 4)))
            logger.info(f"Loaded {len(self._active_grid_cells)} active terrestrial grid cells.")

    def fetch_station_inventory(self) -> pd.DataFrame:
        """Fetch and filter Florida ASOS stations active during 2024."""
        logger.info(f"Querying station inventory from {IEM_GEOJSON_URL}...")
        resp = requests.get(IEM_GEOJSON_URL, headers={"User-Agent": DEFAULT_USER_AGENT}, timeout=30)
        resp.raise_for_status()

        features = resp.json()["features"]
        records = []
        for feat in features:
            props = feat["properties"]
            coords = feat["geometry"]["coordinates"]
            lon, lat = coords[0], coords[1]
            st_id = feat["id"]

            b_str = str(props.get("archive_begin", ""))
            e_str = str(props.get("archive_end", ""))
            on = props.get("online", True)

            b_dt = pd.to_datetime(b_str, errors="coerce")
            e_dt = pd.to_datetime(e_str, errors="coerce")

            # Check overlap with 2024
            overlaps_2024 = (b_dt <= pd.Timestamp("2024-12-31")) and (
                e_dt >= pd.Timestamp("2024-01-01") or pd.isna(e_dt) or on
            )

            glat = round(lat * 4) / 4.0
            glon = round(lon * 4) / 4.0
            in_terrestrial = (round(glat, 4), round(glon, 4)) in self._active_grid_cells
            grid_cell_id = f"FL_{glat}_{glon}" if in_terrestrial else f"COASTAL_MARINE_{glat}_{glon}"

            records.append({
                "station_id": st_id,
                "station_name": props.get("sname", ""),
                "state": props.get("state", "FL"),
                "network": props.get("network", "FL_ASOS"),
                "latitude": lat,
                "longitude": lon,
                "grid_lat": glat,
                "grid_lon": glon,
                "in_active_terrestrial_grid": in_terrestrial,
                "grid_cell_id": grid_cell_id,
                "elevation_m": props.get("elevation", None),
                "archive_begin": b_str,
                "archive_end": e_str,
                "online": on,
                "eligible_2024": overlaps_2024,
            })

        df_all = pd.DataFrame(records)
        df_eligible = df_all[df_all["eligible_2024"] == True].copy().sort_values("station_id").reset_index(drop=True)

        inv_csv = self.output_dir / "florida_asos_stations_2024.csv"
        df_eligible.to_csv(inv_csv, index=False)
        logger.info(f"Saved {len(df_eligible)} eligible 2024 Florida ASOS stations to {inv_csv}")
        self.stations_df = df_eligible
        return df_eligible

    @staticmethod
    def generate_date_batches(year: int = 2024, batch_days: int = 5) -> List[Dict[str, Any]]:
        """Generate contiguous non-overlapping 5-day date ranges spanning the entire year."""
        batches = []
        d_start = datetime(year, 1, 1, 0, 0, 0)
        d_year_end = datetime(year, 12, 31, 23, 59, 59)

        batch_id = 1
        curr = d_start
        while curr <= d_year_end:
            next_end = min(curr + timedelta(days=batch_days - 1), d_year_end)
            batches.append({
                "batch_id": batch_id,
                "start_date": curr.strftime("%Y-%m-%d"),
                "end_date": next_end.strftime("%Y-%m-%d"),
                "year1": curr.year, "month1": curr.month, "day1": curr.day,
                "year2": next_end.year, "month2": next_end.month, "day2": next_end.day,
            })
            batch_id += 1
            curr = next_end + timedelta(days=1)

        return batches

    def download_and_parse_batch(self, batch_info: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """Download a single date batch for all FL_ASOS stations, parse METARs, and save to parquet."""
        bid = batch_info["batch_id"]
        parquet_path = self.raw_batches_dir / f"florida_asos_batch_{bid:03d}.parquet"

        # Check if already processed
        if parquet_path.exists() and parquet_path.stat().st_size > 0:
            try:
                existing_df = pd.read_parquet(parquet_path)
                return {
                    "batch_id": bid,
                    "start_date": batch_info["start_date"],
                    "end_date": batch_info["end_date"],
                    "status": "SUCCESS",
                    "row_count": len(existing_df),
                    "ts_count": int(existing_df["has_thunderstorm_ts"].sum()),
                    "vcts_count": int(existing_df["has_vicinity_thunderstorm_vcts"].sum()),
                    "duration_seconds": 0.0,
                    "error_message": None,
                }
            except Exception:
                pass  # Re-download if corrupted

        params = {
            "network": "FL_ASOS",
            "data": ["metar", "wxcodes", "tmpf", "dwpf", "sknt", "drct", "gust", "vsby", "p01i", "alti"],
            "year1": str(batch_info["year1"]),
            "month1": str(batch_info["month1"]),
            "day1": str(batch_info["day1"]),
            "year2": str(batch_info["year2"]),
            "month2": str(batch_info["month2"]),
            "day2": str(batch_info["day2"]),
            "tz": "Etc/UTC",
            "format": "onlycomma",
            "latlon": "yes",
            "elev": "yes",
            "missing": "null",
            "trace": "T",
            "direct": "yes",
            "report_type": ["1", "2"]
        }

        headers = {"User-Agent": DEFAULT_USER_AGENT}
        attempt = 0
        while attempt < max_retries:
            attempt += 1
            t0 = time.time()
            try:
                logger.info(f"Downloading Batch {bid} ({batch_info['start_date']} to {batch_info['end_date']}) - Attempt {attempt}...")
                resp = requests.get(IEM_ASOS_URL, params=params, headers=headers, timeout=90)
                dur = time.time() - t0

                if resp.status_code == 200:
                    text_data = resp.text.strip()
                    if not text_data or text_data.startswith("ERROR"):
                        raise ValueError(f"Server returned empty or error response: {text_data[:100]}")

                    df_batch = pd.read_csv(io.StringIO(text_data), low_memory=False)
                    if df_batch.empty:
                        logger.warning(f"Batch {bid} returned 0 rows.")
                        df_batch.to_parquet(parquet_path, index=False)
                        return {
                            "batch_id": bid, "start_date": batch_info["start_date"], "end_date": batch_info["end_date"],
                            "status": "SUCCESS", "row_count": 0, "ts_count": 0, "vcts_count": 0,
                            "duration_seconds": round(dur, 2), "error_message": None,
                        }

                    # Coerce numeric columns to float (handles mixed types like '1/4' or 'null')
                    num_cols = ["tmpf", "dwpf", "sknt", "drct", "gust", "vsby", "p01i", "alti", "elevation", "lat", "lon"]
                    for col in num_cols:
                        if col in df_batch.columns:
                            df_batch[col] = pd.to_numeric(df_batch[col], errors="coerce")

                    # Harmonize column names and types
                    df_batch["valid_time_utc"] = pd.to_datetime(df_batch["valid"], utc=True)
                    df_batch["station_id"] = df_batch["station"].astype(str).str.strip()

                    # Merge station spatial grid info
                    if self.stations_df is not None:
                        st_lookup = self.stations_df[["station_id", "station_name", "grid_lat", "grid_lon", "in_active_terrestrial_grid", "grid_cell_id"]].drop_duplicates()
                        df_batch = pd.merge(df_batch, st_lookup, on="station_id", how="left")
                    else:
                        df_batch["grid_lat"] = df_batch["lat"].round(2)
                        df_batch["grid_lon"] = df_batch["lon"].round(2)
                        df_batch["in_active_terrestrial_grid"] = True
                        df_batch["grid_cell_id"] = "FL_" + df_batch["grid_lat"].astype(str) + "_" + df_batch["grid_lon"].astype(str)

                    # Parse METAR thunderstorm and lightning evidence
                    metar_col = df_batch["metar"].fillna("")
                    wx_col = df_batch["wxcodes"].fillna("")

                    parsed_records = [
                        parse_metar_thunderstorm(m, w)
                        for m, w in zip(metar_col, wx_col)
                    ]
                    df_parsed = pd.DataFrame(parsed_records)

                    df_batch["has_thunderstorm_ts"] = df_parsed["has_ts"]
                    df_batch["has_vicinity_thunderstorm_vcts"] = df_parsed["has_vcts"]
                    df_batch["has_lightning_remark"] = df_parsed["has_lightning_remark"]
                    df_batch["thunderstorm_reported"] = df_parsed["thunderstorm_reported"]
                    df_batch["lightning_remarks"] = df_parsed["lightning_remarks"]

                    # Deduplicate exact station + timestamp
                    n_before = len(df_batch)
                    df_batch = df_batch.drop_duplicates(subset=["station_id", "valid_time_utc"], keep="last")
                    n_dups = n_before - len(df_batch)

                    df_batch.to_parquet(parquet_path, index=False, engine="pyarrow")
                    logger.info(f"Batch {bid} SUCCESS: {len(df_batch):,} rows ({df_batch['has_thunderstorm_ts'].sum():,} TS, {df_batch['has_vicinity_thunderstorm_vcts'].sum():,} VCTS) in {dur:.1f}s. Dups dropped: {n_dups}")

                    return {
                        "batch_id": bid,
                        "start_date": batch_info["start_date"],
                        "end_date": batch_info["end_date"],
                        "status": "SUCCESS",
                        "row_count": len(df_batch),
                        "ts_count": int(df_batch["has_thunderstorm_ts"].sum()),
                        "vcts_count": int(df_batch["has_vicinity_thunderstorm_vcts"].sum()),
                        "duration_seconds": round(dur, 2),
                        "error_message": None,
                    }
                else:
                    logger.warning(f"Batch {bid} HTTP error {resp.status_code}. Retrying in {attempt * 3}s...")
                    time.sleep(attempt * 3)

            except Exception as exc:
                logger.error(f"Batch {bid} Attempt {attempt} failed: {exc}")
                time.sleep(attempt * 3)

        return {
            "batch_id": bid,
            "start_date": batch_info["start_date"],
            "end_date": batch_info["end_date"],
            "status": "FAILED",
            "row_count": 0,
            "ts_count": 0,
            "vcts_count": 0,
            "duration_seconds": 0.0,
            "error_message": f"Failed after {max_retries} attempts",
        }

    def run_full_year_acquisition(self, max_workers: int = 2) -> pd.DataFrame:
        """Execute acquisition for all batches in 2024 with polite concurrency."""
        if self.stations_df is None:
            self.fetch_station_inventory()

        batches = self.generate_date_batches(2024, batch_days=5)
        logger.info(f"Generated {len(batches)} five-day batches for full year 2024.")

        # Load existing manifest if any
        manifest_records = []
        if self.manifest_path.exists():
            try:
                manifest_records = pd.read_csv(self.manifest_path).to_dict(orient="records")
                logger.info(f"Loaded existing manifest with {len(manifest_records)} entries.")
            except Exception:
                pass

        completed_batch_ids = {r["batch_id"] for r in manifest_records if r.get("status") == "SUCCESS"}
        pending_batches = [b for b in batches if b["batch_id"] not in completed_batch_ids]
        logger.info(f"{len(completed_batch_ids)} batches already complete. {len(pending_batches)} batches remaining.")

        if pending_batches:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch = {
                    executor.submit(self.download_and_parse_batch, b): b
                    for b in pending_batches
                }
                for future in as_completed(future_to_batch):
                    res = future.result()
                    manifest_records.append(res)
                    # Persist manifest dynamically
                    pd.DataFrame(manifest_records).to_csv(self.manifest_path, index=False)
                    time.sleep(1.0)  # Rate limiting sleep between requests

        df_manifest = pd.DataFrame(manifest_records).sort_values("batch_id").reset_index(drop=True)
        df_manifest.to_csv(self.manifest_path, index=False)
        return df_manifest

    def combine_and_align_forecast_windows(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Combine all raw batch files into a unified master observation dataset and
        aggregate into strictly half-open (t, t+6h] forecasting windows.
        """
        batch_files = sorted(list(self.raw_batches_dir.glob("florida_asos_batch_*.parquet")))
        logger.info(f"Combining {len(batch_files)} batch parquet files...")

        dfs = []
        for bf in batch_files:
            try:
                df = pd.read_parquet(bf)
                if not df.empty:
                    dfs.append(df)
            except Exception as exc:
                logger.error(f"Error reading {bf}: {exc}")

        if not dfs:
            raise RuntimeError("No valid batch files available to combine.")

        df_all = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total raw observations combined: {len(df_all):,}")

        # Global deduplication across all batches
        init_len = len(df_all)
        df_all = df_all.drop_duplicates(subset=["station_id", "valid_time_utc"]).reset_index(drop=True)
        logger.info(f"Deduplicated combined observations: {len(df_all):,} (dropped {init_len - len(df_all):,})")

        # Coerce numeric columns and string columns to uniform types for PyArrow
        numeric_cols = ["tmpf", "dwpf", "sknt", "drct", "gust", "vsby", "p01i", "alti", "elevation", "lat", "lon"]
        for col in numeric_cols:
            if col in df_all.columns:
                df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

        str_cols = ["station_id", "station_name", "state", "presentwx", "metar_raw"]
        for col in str_cols:
            if col in df_all.columns:
                df_all[col] = df_all[col].fillna("").astype(str)

        # 1. Master observations Parquet
        obs_parquet = self.output_dir / "florida_asos_observations_2024.parquet"
        df_all.to_parquet(obs_parquet, index=False, engine="pyarrow")
        logger.info(f"Saved master observations to {obs_parquet} ({len(df_all):,} rows)")

        # 2. Strict half-open target window snapping (t, t+6h]
        # Mathematical rule: cycle t is (valid_time_utc - 1us).floor('6h')
        df_all["forecast_init_time_utc"] = (df_all["valid_time_utc"] - pd.Timedelta(microseconds=1)).dt.floor("6h")

        # Aggregate by (forecast_init_time_utc, station_id, grid_cell_id)
        agg_windows = df_all.groupby(["forecast_init_time_utc", "station_id", "grid_cell_id", "in_active_terrestrial_grid"]).agg(
            observation_count=("valid_time_utc", "count"),
            ts_report_count=("has_thunderstorm_ts", "sum"),
            vcts_report_count=("has_vicinity_thunderstorm_vcts", "sum"),
            lightning_remark_count=("has_lightning_remark", "sum"),
            thunderstorm_any_count=("thunderstorm_reported", "sum"),
            first_observation_utc=("valid_time_utc", "min"),
            last_observation_utc=("valid_time_utc", "max"),
        ).reset_index()

        # Strict classification of reporting completeness:
        # Minimum 4 observations in a 6-hour window required to declare adequately monitored
        def assign_window_status(row):
            obs_cnt = row["observation_count"]
            has_ts = row["thunderstorm_any_count"] > 0
            if has_ts:
                return 1, "COVERED_THUNDERSTORM"
            elif obs_cnt >= 4:
                return 0, "COVERED_NO_THUNDERSTORM"
            elif obs_cnt > 0:
                return np.nan, "INCOMPLETE_MONITORING"
            else:
                return np.nan, "UNOBSERVED"

        statuses = [assign_window_status(r) for _, r in agg_windows.iterrows()]
        agg_windows["asos_thunderstorm_target"] = [s[0] for s in statuses]
        agg_windows["coverage_status"] = [s[1] for s in statuses]

        windows_parquet = self.output_dir / "florida_asos_forecast_windows_2024.parquet"
        agg_windows.to_parquet(windows_parquet, index=False, engine="pyarrow")
        logger.info(f"Saved forecast window aggregations to {windows_parquet} ({len(agg_windows):,} window-station records)")

        return df_all, agg_windows


def main():
    parser = argparse.ArgumentParser(description="Automated Florida ASOS/METAR 2024 Acquisition.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/asos"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    pipeline = ASOSAcquisitionPipeline(output_dir=args.output_dir)
    pipeline.fetch_station_inventory()
    df_manifest = pipeline.run_full_year_acquisition(max_workers=args.workers)
    logger.info(f"Acquisition manifest summary: {df_manifest['status'].value_counts().to_dict()}")

    if (df_manifest["status"] == "SUCCESS").all():
        pipeline.combine_and_align_forecast_windows()
        logger.info("Complete 2024 Florida ASOS acquisition and harmonization finished successfully!")
    else:
        logger.warning("Some batches failed; check manifest for retry.")


if __name__ == "__main__":
    main()
