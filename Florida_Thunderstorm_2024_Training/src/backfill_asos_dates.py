"""
backfill_asos_dates.py
======================
Targeted acquisition and data repair pipeline to backfill the 73 missing
calendar dates from the 2024 Florida ASOS observations.

Workflow:
1. Identify 73 missing dates from batch boundary truncation.
2. Query IEM ASOS API with inclusive hour2=23, minute2=59.
3. Save raw backfill files to data/processed/asos_repair/raw_backfill_batches/.
4. Combine original + backfill into florida_asos_observations_repaired_2024.parquet.
5. Re-run Criterion C forecast window aggregation (incorporating unobserved windows).
6. Merge with ERA5 predictors into florida_asos_modeling_ready_repaired_2024.parquet.
7. Re-evaluate baseline models and save metrics.
"""

from __future__ import annotations

import argparse
import io
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, brier_score_loss
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("backfill_asos_dates")

ROOT_DIR = Path(".").resolve()
ORIG_ASOS_DIR = ROOT_DIR / "data" / "processed" / "asos"
REPAIR_DIR = ROOT_DIR / "data" / "processed" / "asos_repair"
RAW_BACKFILL_DIR = REPAIR_DIR / "raw_backfill_batches"

REPAIR_DIR.mkdir(parents=True, exist_ok=True)
RAW_BACKFILL_DIR.mkdir(parents=True, exist_ok=True)

ORIG_OBS_PARQUET = ORIG_ASOS_DIR / "florida_asos_observations_2024.parquet"
STATIONS_CSV = ORIG_ASOS_DIR / "florida_asos_stations_2024.csv"
FORECAST_6H_CSV = ROOT_DIR / "data" / "processed" / "florida_thunderstorm_2024_forecasting_6h.csv"
REPAIR_MANIFEST_CSV = REPAIR_DIR / "asos_missing_dates_manifest.csv"

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
DEFAULT_USER_AGENT = "FloridaThunderstormAudit/1.0"


def parse_metar_strict(metar_text: str, wxcodes_text: str = "") -> Dict[str, Any]:
    """Parse METAR report strictly separating TS, VCTS, genuine LTG remarks, and CB."""
    metar = str(metar_text) if pd.notna(metar_text) else ""
    wx = str(wxcodes_text) if pd.notna(wxcodes_text) else ""

    parts = metar.split(" RMK ")
    body = parts[0]
    remarks = parts[1] if len(parts) > 1 else ""

    # 1. In-station TS
    has_ts = False
    for code in wx.split():
        if "TS" in code and not code.startswith("VC"):
            has_ts = True
            break
    if not has_ts:
        for token in body.split():
            if "TS" in token and not token.startswith("VC") and token not in ["AUTO", "METAR", "SPECI"]:
                has_ts = True
                break

    # 2. Vicinity VCTS
    has_vcts = False
    for code in wx.split():
        if code.startswith("VC") and "TS" in code:
            has_vcts = True
            break
    if not has_vcts:
        for token in (body + " " + remarks).split():
            if token.startswith("VCTS") or "VCTS" in token:
                has_vcts = True
                break

    # 3. Genuine lightning remarks (LTG, TSB, TSE)
    ltg_matches = []
    m_ltg = re.findall(r"\b(?:OCNL|FRQ|CONS)?\s*LTG[A-Z]*(?:\s+[A-Z0-9\-\/]+){0,4}", remarks)
    if m_ltg:
        ltg_matches.extend([m.strip() for m in m_ltg])

    m_tsb = re.findall(r"\bTSB\d{2,4}\b", remarks)
    if m_tsb:
        ltg_matches.extend(m_tsb)

    m_tse = re.findall(r"\bTSE\d{2,4}\b", remarks)
    if m_tse:
        ltg_matches.extend(m_tse)

    has_genuine_ltg = len(ltg_matches) > 0

    # 4. Cumulonimbus (CB) in body or remarks
    cb_matches = re.findall(r"\bCB\b|\bCB\s+[A-Z]|\d{3}CB\b", metar)
    has_cb = len(cb_matches) > 0

    # Strict thunderstorm = TS | VCTS | Genuine LTG (excluding CB alone)
    strict_ts = has_ts or has_vcts or has_genuine_ltg
    cb_alone = has_cb and not strict_ts

    return {
        "has_thunderstorm_ts": bool(has_ts),
        "has_vicinity_thunderstorm_vcts": bool(has_vcts),
        "has_genuine_lightning_remark": bool(has_genuine_ltg),
        "has_cumulonimbus_cb": bool(has_cb),
        "strict_thunderstorm_reported": bool(strict_ts),
        "cb_alone": bool(cb_alone),
        "lightning_remarks": "; ".join(ltg_matches) if ltg_matches else None,
    }


class ASOSBackfillPipeline:
    def __init__(self):
        self.df_stations = pd.read_csv(STATIONS_CSV)
        self.station_map = {}
        for _, r in self.df_stations.iterrows():
            st_id = str(r["station_id"]).strip()
            self.station_map[st_id] = {
                "station_id": st_id,
                "station_name": r["station_name"],
                "grid_lat": r["grid_lat"],
                "grid_lon": r["grid_lon"],
                "in_active_terrestrial_grid": r["in_active_terrestrial_grid"],
                "grid_cell_id": r["grid_cell_id"],
            }
        logger.info(f"Loaded {len(self.df_stations)} station definitions.")

    def identify_missing_dates(self) -> List[date]:
        """Identify missing calendar dates from original observations."""
        logger.info("Reading original observations to identify present dates...")
        df_orig = pd.read_parquet(ORIG_OBS_PARQUET, columns=["valid_time_utc"])
        present_dates = set(df_orig["valid_time_utc"].dt.date.unique())
        all_dates = set(pd.date_range("2024-01-01", "2024-12-31", freq="D").date)
        missing = sorted(list(all_dates - present_dates))
        logger.info(f"Identified {len(missing)} missing calendar dates in 2024.")
        return missing

    def download_date(self, target_date: date, max_retries: int = 3) -> Dict[str, Any]:
        """Download observations for all FL_ASOS stations for a single missing date."""
        d_str = target_date.strftime("%Y-%m-%d")
        parquet_path = RAW_BACKFILL_DIR / f"backfill_{d_str}.parquet"

        # Check existing
        if parquet_path.exists():
            try:
                df_ex = pd.read_parquet(parquet_path)
                return {
                    "date": d_str,
                    "status": "SUCCESS",
                    "row_count": len(df_ex),
                    "station_count": df_ex["station_id"].nunique(),
                    "min_time": str(df_ex["valid_time_utc"].min()),
                    "max_time": str(df_ex["valid_time_utc"].max()),
                    "duration_sec": 0.0,
                    "error_message": None,
                }
            except Exception:
                pass

        params = {
            "network": "FL_ASOS",
            "data": ["metar", "wxcodes", "tmpf", "dwpf", "sknt", "drct", "gust", "vsby", "p01i", "alti"],
            "year1": str(target_date.year),
            "month1": str(target_date.month),
            "day1": str(target_date.day),
            "year2": str(target_date.year),
            "month2": str(target_date.month),
            "day2": str(target_date.day),
            "hour2": "23",
            "minute2": "59",
            "tz": "Etc/UTC",
            "format": "onlycomma",
            "latlon": "yes",
            "elev": "yes",
            "missing": "null",
            "trace": "T",
            "direct": "yes",
            "report_type": ["1", "2"],
        }
        headers = {"User-Agent": DEFAULT_USER_AGENT}

        for attempt in range(1, max_retries + 1):
            t0 = time.time()
            try:
                logger.info(f"Downloading backfill for {d_str} (Attempt {attempt}/{max_retries})...")
                resp = requests.get(IEM_ASOS_URL, params=params, headers=headers, timeout=60)
                dur = time.time() - t0

                if resp.status_code == 200:
                    text_data = resp.text.strip()
                    if not text_data or text_data.startswith("ERROR"):
                        raise ValueError(f"Server error: {text_data[:100]}")

                    df_raw = pd.read_csv(io.StringIO(text_data), low_memory=False)
                    if df_raw.empty:
                        logger.warning(f"Date {d_str} returned 0 rows.")
                        df_raw.to_parquet(parquet_path, index=False)
                        return {
                            "date": d_str, "status": "SUCCESS", "row_count": 0, "station_count": 0,
                            "min_time": None, "max_time": None, "duration_sec": round(dur, 2), "error_message": None,
                        }

                    # Filter to eligible stations
                    df_raw["station"] = df_raw["station"].str.strip()
                    df_filt = df_raw[df_raw["station"].isin(self.station_map.keys())].copy()

                    # Deduplicate station-valid
                    df_filt["valid_time_utc"] = pd.to_datetime(df_filt["valid"], utc=True)
                    df_filt = df_filt.drop_duplicates(subset=["station", "valid_time_utc"]).reset_index(drop=True)

                    # Annotate station metadata
                    df_filt["station_id"] = df_filt["station"]
                    df_filt["station_name"] = df_filt["station_id"].apply(lambda s: self.station_map[s]["station_name"])
                    df_filt["grid_lat"] = df_filt["station_id"].apply(lambda s: self.station_map[s]["grid_lat"])
                    df_filt["grid_lon"] = df_filt["station_id"].apply(lambda s: self.station_map[s]["grid_lon"])
                    df_filt["in_active_terrestrial_grid"] = df_filt["station_id"].apply(lambda s: self.station_map[s]["in_active_terrestrial_grid"])
                    df_filt["grid_cell_id"] = df_filt["station_id"].apply(lambda s: self.station_map[s]["grid_cell_id"])

                    # Parse METAR semantics
                    parsed = [parse_metar_strict(r["metar"], r.get("wxcodes", "")) for _, r in df_filt.iterrows()]
                    df_filt["has_thunderstorm_ts"] = [p["has_thunderstorm_ts"] for p in parsed]
                    df_filt["has_vicinity_thunderstorm_vcts"] = [p["has_vicinity_thunderstorm_vcts"] for p in parsed]
                    df_filt["has_genuine_lightning_remark"] = [p["has_genuine_lightning_remark"] for p in parsed]
                    df_filt["has_cumulonimbus_cb"] = [p["has_cumulonimbus_cb"] for p in parsed]
                    df_filt["strict_thunderstorm_reported"] = [p["strict_thunderstorm_reported"] for p in parsed]
                    df_filt["cb_alone"] = [p["cb_alone"] for p in parsed]
                    df_filt["lightning_remarks"] = [p["lightning_remarks"] for p in parsed]

                    # Coerce numeric columns
                    num_cols = ["tmpf", "dwpf", "sknt", "drct", "gust", "vsby", "p01i", "alti", "elevation", "lat", "lon"]
                    for c in num_cols:
                        if c in df_filt.columns:
                            df_filt[c] = pd.to_numeric(df_filt[c], errors="coerce")

                    str_cols = ["station_id", "station_name", "wxcodes", "metar"]
                    for c in str_cols:
                        if c in df_filt.columns:
                            df_filt[c] = df_filt[c].fillna("").astype(str)

                    # Save to parquet
                    df_filt.to_parquet(parquet_path, index=False, engine="pyarrow")
                    logger.info(f"Saved backfill {d_str}: {len(df_filt):,} rows ({df_filt['station_id'].nunique()} stations) in {dur:.2f}s")

                    return {
                        "date": d_str,
                        "status": "SUCCESS",
                        "row_count": len(df_filt),
                        "station_count": df_filt["station_id"].nunique(),
                        "min_time": str(df_filt["valid_time_utc"].min()),
                        "max_time": str(df_filt["valid_time_utc"].max()),
                        "duration_sec": round(dur, 2),
                        "error_message": None,
                    }
                else:
                    raise RuntimeError(f"HTTP {resp.status_code}")
            except Exception as exc:
                logger.warning(f"Attempt {attempt} failed for {d_str}: {exc}")
                time.sleep(3.0 * attempt)

        return {
            "date": d_str, "status": "FAILED", "row_count": 0, "station_count": 0,
            "min_time": None, "max_time": None, "duration_sec": 0.0, "error_message": str(exc),
        }

    def run_backfill(self, max_workers: int = 2) -> pd.DataFrame:
        """Run backfill across all 73 missing dates."""
        missing_dates = self.identify_missing_dates()
        logger.info(f"Starting targeted backfill of {len(missing_dates)} missing dates with {max_workers} workers...")

        manifest_records = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_date = {executor.submit(self.download_date, d): d for d in missing_dates}
            for future in as_completed(future_to_date):
                rec = future.result()
                manifest_records.append(rec)
                time.sleep(0.5)

        df_man = pd.DataFrame(manifest_records).sort_values("date").reset_index(drop=True)
        df_man.to_csv(REPAIR_MANIFEST_CSV, index=False)
        logger.info(f"Backfill manifest saved to {REPAIR_MANIFEST_CSV}. Status: {df_man['status'].value_counts().to_dict()}")
        return df_man

    def combine_repaired_dataset(self) -> pd.DataFrame:
        """Combine original observations with all 73 backfill batches."""
        logger.info("Combining original observations with backfill batches...")
        df_orig = pd.read_parquet(ORIG_OBS_PARQUET)
        logger.info(f"Loaded original observations: {len(df_orig):,} rows")

        # Parse original if missing audited columns
        if "strict_thunderstorm_reported" not in df_orig.columns:
            logger.info("Adding audited convective flags to original dataset...")
            parsed = [parse_metar_strict(r["metar"], r.get("wxcodes", "")) for _, r in df_orig.iterrows()]
            df_orig["has_genuine_lightning_remark"] = [p["has_genuine_lightning_remark"] for p in parsed]
            df_orig["has_cumulonimbus_cb"] = [p["has_cumulonimbus_cb"] for p in parsed]
            df_orig["strict_thunderstorm_reported"] = [p["strict_thunderstorm_reported"] for p in parsed]
            df_orig["cb_alone"] = [p["cb_alone"] for p in parsed]

        # Load all backfill batches
        backfill_files = sorted(list(RAW_BACKFILL_DIR.glob("backfill_*.parquet")))
        logger.info(f"Found {len(backfill_files)} backfill parquet files.")
        bf_dfs = []
        for bf in backfill_files:
            b_df = pd.read_parquet(bf)
            if not b_df.empty:
                bf_dfs.append(b_df)

        df_bf = pd.concat(bf_dfs, ignore_index=True)
        logger.info(f"Total backfill rows: {len(df_bf):,}")

        # Combine
        common_cols = [c for c in df_orig.columns if c in df_bf.columns]
        df_comb = pd.concat([df_orig[common_cols], df_bf[common_cols]], ignore_index=True)
        logger.info(f"Total combined rows before deduplication: {len(df_comb):,}")

        # Deduplicate
        df_repaired = df_comb.drop_duplicates(subset=["station_id", "valid_time_utc"]).sort_values(["station_id", "valid_time_utc"]).reset_index(drop=True)
        logger.info(f"Total repaired observations after deduplication: {len(df_repaired):,}")

        # Verify date coverage
        dates_covered = df_repaired["valid_time_utc"].dt.date.nunique()
        logger.info(f"Repaired dataset unique calendar dates: {dates_covered} / 366")

        repaired_obs_path = REPAIR_DIR / "florida_asos_observations_repaired_2024.parquet"
        df_repaired.to_parquet(repaired_obs_path, index=False, engine="pyarrow")
        logger.info(f"Saved repaired observations to {repaired_obs_path}")

        return df_repaired

    def rebuild_forecast_windows(self, df_obs: pd.DataFrame) -> pd.DataFrame:
        """Aggregate repaired observations into 6h forecast windows under Criterion C."""
        logger.info("Snapping observations to half-open (t, t+6h] forecast windows...")
        df_obs["forecast_init_time_utc"] = (df_obs["valid_time_utc"] - pd.Timedelta(microseconds=1)).dt.floor("6h")

        # Sort for gap calculation
        df_obs = df_obs.sort_values(["station_id", "valid_time_utc"]).reset_index(drop=True)
        df_obs["prev_time"] = df_obs.groupby(["station_id", "forecast_init_time_utc"])["valid_time_utc"].shift(1)
        df_obs["gap_minutes"] = (df_obs["valid_time_utc"] - df_obs["prev_time"]).dt.total_seconds() / 60.0

        agg_windows = df_obs.groupby(["forecast_init_time_utc", "station_id", "grid_cell_id", "in_active_terrestrial_grid"]).agg(
            obs_count=("valid_time_utc", "count"),
            ts_count=("has_thunderstorm_ts", "sum"),
            vcts_count=("has_vicinity_thunderstorm_vcts", "sum"),
            ltg_count=("has_genuine_lightning_remark", "sum"),
            cb_alone_count=("cb_alone", "sum"),
            strict_ts_count=("strict_thunderstorm_reported", "sum"),
            first_obs=("valid_time_utc", "min"),
            last_obs=("valid_time_utc", "max"),
            max_gap_minutes=("gap_minutes", "max"),
        ).reset_index()

        agg_windows["max_gap_minutes"] = agg_windows["max_gap_minutes"].fillna(360.0)
        agg_windows["obs_span_hours"] = (agg_windows["last_obs"] - agg_windows["first_obs"]).dt.total_seconds() / 3600.0

        # Apply Criterion C
        def assign_status_crit_c(row):
            has_ts = row["strict_ts_count"] > 0
            if has_ts:
                return 1.0, "COVERED_THUNDERSTORM"
            elif row["obs_count"] >= 6 and row["max_gap_minutes"] <= 75.0 and row["obs_span_hours"] >= 5.0:
                return 0.0, "COVERED_NO_THUNDERSTORM"
            else:
                return np.nan, "INCOMPLETE_MONITORING"

        statuses = [assign_status_crit_c(r) for _, r in agg_windows.iterrows()]
        agg_windows["target_crit_c"] = [s[0] for s in statuses]
        agg_windows["status_crit_c"] = [s[1] for s in statuses]

        windows_path = REPAIR_DIR / "florida_asos_forecast_windows_repaired_2024.parquet"
        agg_windows.to_parquet(windows_path, index=False, engine="pyarrow")
        logger.info(f"Saved repaired forecast windows to {windows_path} ({len(agg_windows):,} rows)")

        return agg_windows

    def rebuild_modeling_dataset_and_evaluate(self, df_windows: pd.DataFrame):
        """Merge repaired forecast windows with ERA5 predictors and evaluate baselines."""
        logger.info("Merging repaired forecast windows with ERA5 predictors...")
        feature_cols = [
            "forecast_init_time_utc", "grid_cell_id", "latitude", "longitude",
            "month", "hour_utc", "temperature_2m_degC", "dewpoint_2m_degC",
            "dewpoint_depression_degC", "surface_pressure_hPa", "mslp_hPa",
            "wind_speed_10m_ms", "wind_direction_10m_deg", "total_precipitation_mm",
            "convective_precipitation_mm", "cape_J_kg", "cin_J_kg", "cloud_cover_fraction",
            "tcwv_kg_m2", "temperature_850hPa_degC", "relative_humidity_850hPa_pct",
            "wind_speed_850hPa_ms", "temperature_500hPa_degC", "relative_humidity_500hPa_pct",
            "bulk_shear_850_300_ms", "temp_diff_850_500_degC", "era5_thunderstorm_favorability",
            "future_noaa_event_6h", "true_prior_event_6h"
        ]
        df_fc = pd.read_csv(FORECAST_6H_CSV, usecols=feature_cols)

        df_windows["init_str"] = df_windows["forecast_init_time_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df_merged = pd.merge(
            df_windows, df_fc,
            left_on=["init_str", "grid_cell_id"],
            right_on=["forecast_init_time_utc", "grid_cell_id"],
            how="inner",
            suffixes=("", "_era5")
        )

        df_terr = df_merged[df_merged["in_active_terrestrial_grid"] == True].copy()
        logger.info(f"Repaired terrestrial station-windows with ERA5 features: {len(df_terr):,} rows (was 98,353 prior to repair)")

        # Disaggregated targets
        df_terr["target_asos_strict_crit_c"] = df_terr["target_crit_c"]
        df_terr["target_asos_explicit_ts"] = np.where(
            df_terr["ts_count"] > 0, 1.0,
            np.where(df_terr["status_crit_c"] == "COVERED_NO_THUNDERSTORM", 0.0, np.nan)
        )
        df_terr["target_asos_aerodrome_ts_vcts"] = np.where(
            (df_terr["ts_count"] > 0) | (df_terr["vcts_count"] > 0), 1.0,
            np.where(df_terr["status_crit_c"] == "COVERED_NO_THUNDERSTORM", 0.0, np.nan)
        )
        df_terr["target_noaa_severe_6h"] = df_terr["future_noaa_event_6h"]

        # Save modeling ready dataset
        modeling_path = REPAIR_DIR / "florida_asos_modeling_ready_repaired_2024.parquet"
        df_terr.to_parquet(modeling_path, index=False, engine="pyarrow")
        logger.info(f"Saved repaired modeling dataset to {modeling_path}")

        # Baseline evaluation on chronological split
        df_eval = df_terr[df_terr["target_asos_strict_crit_c"].notna()].copy()
        df_eval["target"] = df_eval["target_asos_strict_crit_c"].astype(int)

        df_eval = df_eval.sort_values(["station_id", "forecast_init_time_utc"]).reset_index(drop=True)
        df_eval["prev_init"] = df_eval.groupby("station_id")["forecast_init_time_utc"].shift(1)
        df_eval["prev_target"] = df_eval.groupby("station_id")["target"].shift(1)
        df_eval["cycle_diff_hours"] = (df_eval["forecast_init_time_utc"] - df_eval["prev_init"]).dt.total_seconds() / 3600.0
        # Strict persistence requires exactly 6h prior cycle
        df_eval["pred_persistence"] = np.where(df_eval["cycle_diff_hours"] == 6.0, df_eval["prev_target"], 0).astype(int)

        train_mask = df_eval["forecast_init_time_utc"] < "2024-08-01"
        val_mask = (df_eval["forecast_init_time_utc"] >= "2024-08-01") & (df_eval["forecast_init_time_utc"] < "2024-10-01")
        test_mask = df_eval["forecast_init_time_utc"] >= "2024-10-01"

        logger.info(f"Repaired Train samples: {train_mask.sum():,}, Pos={df_eval.loc[train_mask, 'target'].sum():,} ({df_eval.loc[train_mask, 'target'].mean()*100:.2f}%)")
        logger.info(f"Repaired Val samples:   {val_mask.sum():,}, Pos={df_eval.loc[val_mask, 'target'].sum():,} ({df_eval.loc[val_mask, 'target'].mean()*100:.2f}%)")
        logger.info(f"Repaired Test samples:  {test_mask.sum():,}, Pos={df_eval.loc[test_mask, 'target'].sum():,} ({df_eval.loc[test_mask, 'target'].mean()*100:.2f}%)")

        y_test = df_eval.loc[test_mask, "target"].values
        y_train = df_eval.loc[train_mask, "target"].values

        # 1. Climatology
        p_clim = y_train.mean()
        y_prob_clim = np.full_like(y_test, fill_value=p_clim, dtype=float)

        # 2. Persistence
        y_pred_persist = df_eval.loc[test_mask, "pred_persistence"].values
        prec_p = precision_score(y_test, y_pred_persist, zero_division=0)
        rec_p = recall_score(y_test, y_pred_persist, zero_division=0)
        f1_p = f1_score(y_test, y_pred_persist, zero_division=0)
        prauc_p = average_precision_score(y_test, y_pred_persist)
        roc_p = roc_auc_score(y_test, y_pred_persist)
        brier_p = brier_score_loss(y_test, y_pred_persist)

        # 3. CAPE Threshold
        cape_test = df_eval.loc[test_mask, "cape_J_kg"].fillna(0).values
        y_pred_cape = (cape_test >= 1000.0).astype(int)
        prec_c = precision_score(y_test, y_pred_cape, zero_division=0)
        rec_c = recall_score(y_test, y_pred_cape, zero_division=0)
        f1_c = f1_score(y_test, y_pred_cape, zero_division=0)
        prauc_c = average_precision_score(y_test, cape_test)
        roc_c = roc_auc_score(y_test, cape_test)
        brier_c = brier_score_loss(y_test, y_pred_cape)

        # 4. Logistic Regression
        lr_features = [
            "cape_J_kg", "cin_J_kg", "dewpoint_depression_degC",
            "temperature_2m_degC", "bulk_shear_850_300_ms",
            "total_precipitation_mm", "tcwv_kg_m2"
        ]
        X_train = df_eval.loc[train_mask, lr_features].fillna(0).values
        X_test = df_eval.loc[test_mask, lr_features].fillna(0).values

        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        y_prob_lr = lr.predict_proba(X_test)[:, 1]
        y_pred_lr = (y_prob_lr >= 0.5).astype(int)

        prec_lr = precision_score(y_test, y_pred_lr, zero_division=0)
        rec_lr = recall_score(y_test, y_pred_lr, zero_division=0)
        f1_lr = f1_score(y_test, y_pred_lr, zero_division=0)
        prauc_lr = average_precision_score(y_test, y_prob_lr)
        roc_lr = roc_auc_score(y_test, y_prob_lr)
        brier_lr = brier_score_loss(y_test, y_prob_lr)

        baselines = [
            {
                "baseline": "Climatology / Constant Prevalence",
                "precision": float(p_clim), "recall": 1.0,
                "f1": float(2 * p_clim / (1.0 + p_clim)),
                "pr_auc": float(average_precision_score(y_test, y_prob_clim)),
                "roc_auc": 0.5000, "brier_score": float(brier_score_loss(y_test, y_prob_clim)),
            },
            {
                "baseline": "Persistence (Thunderstorm at t-6h)",
                "precision": float(prec_p), "recall": float(rec_p), "f1": float(f1_p),
                "pr_auc": float(prauc_p), "roc_auc": float(roc_p), "brier_score": float(brier_p),
            },
            {
                "baseline": "CAPE Threshold (CAPE >= 1000 J/kg)",
                "precision": float(prec_c), "recall": float(rec_c), "f1": float(f1_c),
                "pr_auc": float(prauc_c), "roc_auc": float(roc_c), "brier_score": float(brier_c),
            },
            {
                "baseline": "Simple Logistic Regression (7 ERA5 Predictors)",
                "precision": float(prec_lr), "recall": float(rec_lr), "f1": float(f1_lr),
                "pr_auc": float(prauc_lr), "roc_auc": float(roc_lr), "brier_score": float(brier_lr),
            },
        ]
        df_base = pd.DataFrame(baselines)
        metrics_path = REPAIR_DIR / "asos_repaired_baseline_metrics.csv"
        df_base.to_csv(metrics_path, index=False)
        logger.info(f"Saved repaired baseline metrics to {metrics_path}")


def main():
    pipeline = ASOSBackfillPipeline()
    # Phase 2: Backfill
    pipeline.run_backfill(max_workers=2)
    # Phase 3: Combine & rebuild
    df_obs = pipeline.combine_repaired_dataset()
    df_windows = pipeline.rebuild_forecast_windows(df_obs)
    pipeline.rebuild_modeling_dataset_and_evaluate(df_windows)
    logger.info("Complete ASOS data repair and revalidation completed successfully!")


if __name__ == "__main__":
    main()
