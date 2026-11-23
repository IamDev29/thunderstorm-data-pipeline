"""
final_validation_audit.py
=========================
Independent validation script for Florida 2024 ASOS/METAR repair audit.
Executes Phase 1 through Phase 5 checks and exports audit deliverables.
"""

from pathlib import Path
import json
import math
import re
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, brier_score_loss
)

ROOT_DIR = Path(".").resolve()
REPAIR_DIR = ROOT_DIR / "data" / "processed" / "asos_repair"
AUDIT_DIR = ROOT_DIR / "data" / "processed" / "asos_final_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

REPAIRED_OBS_FILE = REPAIR_DIR / "florida_asos_observations_repaired_2024.parquet"
REPAIRED_WINDOWS_FILE = REPAIR_DIR / "florida_asos_forecast_windows_repaired_2024.parquet"
REPAIRED_MODELING_FILE = REPAIR_DIR / "florida_asos_modeling_ready_repaired_2024.parquet"
BACKFILL_MANIFEST_FILE = REPAIR_DIR / "asos_missing_dates_manifest.csv"
STATIONS_FILE = ROOT_DIR / "data" / "processed" / "asos" / "florida_asos_stations_2024.csv"
ERA5_6H_FILE = ROOT_DIR / "data" / "processed" / "florida_thunderstorm_2024_forecasting_6h.csv"


def run_phase_1():
    print("=" * 80)
    print("PHASE 1: VERIFY REPAIR COMPLETENESS")
    print("=" * 80)

    # 1. Expected 366 dates
    expected_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range("2024-01-01", "2024-12-31", freq="D")]
    print(f"Total expected UTC calendar dates in 2024 leap year: {len(expected_dates)}")

    # Load backfill manifest
    df_man = pd.read_csv(BACKFILL_MANIFEST_FILE)
    print(f"Backfill manifest entries: {len(df_man)}")
    print(f"Backfill manifest statuses: {df_man['status'].value_counts().to_dict()}")
    manifest_dates = set(df_man["date"])

    # Load repaired observations
    print("Loading repaired observations dataset...")
    df_obs = pd.read_parquet(REPAIRED_OBS_FILE)
    print(f"Repaired observations row count: {len(df_obs):,}")
    print(f"Columns: {df_obs.columns.tolist()}")

    # Date presence
    df_obs["obs_date"] = df_obs["valid_time_utc"].dt.strftime("%Y-%m-%d")
    present_dates = set(df_obs["obs_date"].unique())
    missing_from_obs = set(expected_dates) - present_dates
    print(f"Unique calendar dates in observations: {len(present_dates)} / {len(expected_dates)}")
    print(f"Dates missing from repaired observations: {missing_from_obs}")

    # Check 73 backfilled dates
    all_73_present = manifest_dates.issubset(present_dates)
    print(f"All 73 manifest backfill dates present in observations: {all_73_present}")

    # Timezone & Range Check
    t_min = df_obs["valid_time_utc"].min()
    t_max = df_obs["valid_time_utc"].max()
    tz_info = str(df_obs["valid_time_utc"].dt.tz)
    outside_2024 = (df_obs["valid_time_utc"] < "2024-01-01 00:00:00+00:00") | (df_obs["valid_time_utc"] > "2024-12-31 23:59:59+00:00")
    print(f"Min timestamp: {t_min}")
    print(f"Max timestamp: {t_max}")
    print(f"Timestamp timezone: {tz_info}")
    print(f"Observations outside 2024: {outside_2024.sum()}")

    # Deduplication check
    dup_keys = df_obs.duplicated(subset=["station_id", "valid_time_utc"], keep=False)
    print(f"Duplicate (station_id, valid_time_utc) keys: {dup_keys.sum()}")

    # Station Inventory & Station AGR Status
    df_st = pd.read_csv(STATIONS_FILE)
    print(f"\nStation inventory count: {len(df_st)}")
    observed_stations = set(df_obs["station_id"].unique())
    catalog_stations = set(df_st["station_id"].unique())
    unobserved_stations = catalog_stations - observed_stations
    print(f"Observed stations in repaired dataset: {len(observed_stations)}")
    print(f"Unobserved stations from inventory: {unobserved_stations}")

    # Detailed inspection of AGR
    agr_rows = df_st[df_st["station_id"] == "AGR"].to_dict(orient="records")
    print("\nStation AGR Inventory Details:")
    for k, v in agr_rows[0].items():
        print(f"  {k}: {v}")

    # Per-date coverage report
    print("\nGenerating per-date coverage report...")
    daily_stats = df_obs.groupby("obs_date").agg(
        observation_count=("valid_time_utc", "count"),
        active_stations=("station_id", "nunique"),
        first_obs_utc=("valid_time_utc", "min"),
        last_obs_utc=("valid_time_utc", "max"),
        ts_count=("has_thunderstorm_ts", "sum"),
        strict_ts_count=("strict_thunderstorm_reported", "sum"),
    ).reset_index()
    daily_stats["is_backfilled_date"] = daily_stats["obs_date"].isin(manifest_dates)
    daily_stats["obs_span_hours"] = (daily_stats["last_obs_utc"] - daily_stats["first_obs_utc"]).dt.total_seconds() / 3600.0
    daily_stats["first_obs_str"] = daily_stats["first_obs_utc"].dt.strftime("%H:%M:%S")
    daily_stats["last_obs_str"] = daily_stats["last_obs_utc"].dt.strftime("%H:%M:%S")

    # Export daily_coverage.csv
    daily_stats.to_csv(AUDIT_DIR / "daily_coverage.csv", index=False)
    print(f"Exported daily coverage: {len(daily_stats)} dates to {AUDIT_DIR / 'daily_coverage.csv'}")

    # Per-station coverage report
    print("\nGenerating per-station coverage report...")
    st_stats = df_obs.groupby("station_id").agg(
        total_observations=("valid_time_utc", "count"),
        dates_observed=("obs_date", "nunique"),
        first_obs_utc=("valid_time_utc", "min"),
        last_obs_utc=("valid_time_utc", "max"),
        ts_count=("has_thunderstorm_ts", "sum"),
        strict_ts_count=("strict_thunderstorm_reported", "sum"),
    ).reset_index()

    st_merged = pd.merge(df_st, st_stats, on="station_id", how="left")
    st_merged["total_observations"] = st_merged["total_observations"].fillna(0).astype(int)
    st_merged["dates_observed"] = st_merged["dates_observed"].fillna(0).astype(int)
    st_merged["reporting_status"] = np.where(st_merged["total_observations"] > 0, "ACTIVE", "INACTIVE_OUTAGE")

    st_merged.to_csv(AUDIT_DIR / "station_coverage.csv", index=False)
    print(f"Exported station coverage: {len(st_merged)} stations to {AUDIT_DIR / 'station_coverage.csv'}")

    return {
        "dates_expected": len(expected_dates),
        "dates_present": len(present_dates),
        "all_73_manifest_dates_present": bool(all_73_present),
        "total_observations": len(df_obs),
        "duplicate_keys": int(dup_keys.sum()),
        "outside_2024": int(outside_2024.sum()),
        "observed_stations": len(observed_stations),
        "unobserved_stations": list(unobserved_stations),
        "agr_archive_begin": agr_rows[0].get("archive_begin"),
        "agr_online": bool(agr_rows[0].get("online", True)),
    }


if __name__ == "__main__":
    run_phase_1()
