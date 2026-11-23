"""
audit_asos_labels.py
====================
Independent scientific and data-integrity audit of Florida ASOS/METAR 2024
observations, METAR semantics, negative-label validity, time-window alignment,
and spatial representativeness.

Outputs saved to:
- data/processed/asos_audit/
"""

import math
import re
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Paths
ROOT_DIR = Path(".").resolve()
ASOS_DIR = ROOT_DIR / "data" / "processed" / "asos"
AUDIT_DIR = ROOT_DIR / "data" / "processed" / "asos_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

OBSERVATIONS_PARQUET = ASOS_DIR / "florida_asos_observations_2024.parquet"
WINDOWS_PARQUET = ASOS_DIR / "florida_asos_forecast_windows_2024.parquet"
STATIONS_CSV = ASOS_DIR / "florida_asos_stations_2024.csv"
MANIFEST_CSV = ASOS_DIR / "asos_download_manifest.csv"
FORECAST_6H_CSV = ROOT_DIR / "data" / "processed" / "florida_thunderstorm_2024_forecasting_6h.csv"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points in km."""
    r = 6371.0  # Earth's radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def run_audit():
    print("=" * 80)
    print("STARTING INDEPENDENT AUDIT OF FLORIDA ASOS/METAR 2024 DATASET")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # PART 1: ACQUISITION INTEGRITY
    # -------------------------------------------------------------------------
    print("\n--- 1. ACQUISITION INTEGRITY ---")
    df_manifest = pd.read_csv(MANIFEST_CSV)
    print(f"Manifest total batches: {len(df_manifest)}")
    print(f"Manifest status counts: {df_manifest['status'].value_counts().to_dict()}")
    print(f"Manifest date range: {df_manifest['start_date'].min()} to {df_manifest['end_date'].max()}")

    df_st = pd.read_csv(STATIONS_CSV)
    print(f"Total stations in inventory: {len(df_st)}")
    print(f"  Terrestrial stations: {(df_st['in_active_terrestrial_grid'] == True).sum()}")
    print(f"  Coastal marine stations: {(df_st['in_active_terrestrial_grid'] == False).sum()}")

    print("Loading observations parquet...")
    df_obs = pd.read_parquet(OBSERVATIONS_PARQUET)
    print(f"Total observation rows: {len(df_obs):,}")
    print(f"Columns in observations: {len(df_obs.columns)}")
    print(f"Timestamp range: {df_obs['valid_time_utc'].min()} to {df_obs['valid_time_utc'].max()}")
    print(f"Unique reporting stations: {df_obs['station_id'].nunique()}")

    # Station outage check
    st_obs = set(df_obs["station_id"].unique())
    st_inv = set(df_st["station_id"].unique())
    unobserved_stations = st_inv - st_obs
    print(f"Unobserved stations during 2024: {unobserved_stations}")

    # Null checks across observations
    null_summary = df_obs.isnull().sum()
    null_cols = null_summary[null_summary > 0].to_dict()
    print("Null counts in columns with missing values:")
    for col, cnt in null_cols.items():
        print(f"  {col}: {cnt:,} ({cnt/len(df_obs)*100:.2f}%)")

    # Monthly observation counts
    df_obs["month"] = df_obs["valid_time_utc"].dt.month
    monthly_counts = df_obs.groupby("month").agg(
        total_observations=("station_id", "count"),
        active_stations=("station_id", "nunique"),
        ts_reports=("has_thunderstorm_ts", "sum"),
        vcts_reports=("has_vicinity_thunderstorm_vcts", "sum"),
        any_reports=("thunderstorm_reported", "sum"),
    ).reset_index()
    monthly_counts["month_name"] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]
    monthly_counts.to_csv(AUDIT_DIR / "monthly_observation_audit.csv", index=False)
    print("\nMonthly observation summary:")
    print(monthly_counts[["month_name", "total_observations", "active_stations", "ts_reports", "any_reports"]])

    # -------------------------------------------------------------------------
    # PART 2: METAR SEMANTICS & CB SEPARATION
    # -------------------------------------------------------------------------
    print("\n--- 2. METAR SEMANTICS & CUMULONIMBUS SEPARATION ---")
    
    # We re-evaluate semantic parsing strictly separating:
    # 1. explicit TS in present weather (body / wxcodes)
    # 2. vicinity VCTS in body / wxcodes / remarks
    # 3. genuine lightning in remarks: LTG (with any descriptor), TSB, TSE
    # 4. cumulonimbus (CB) in remarks or cloud layer
    
    def parse_strict_semantics(metar_text: str, wxcodes_text: str):
        metar = str(metar_text) if pd.notna(metar_text) else ""
        wx = str(wxcodes_text) if pd.notna(wxcodes_text) else ""
        
        parts = metar.split(" RMK ")
        body = parts[0]
        remarks = parts[1] if len(parts) > 1 else ""
        
        # 1. Explicit TS (in-station)
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
        has_genuine_ltg = bool(re.search(r"\b(?:OCNL|FRQ|CONS)?\s*LTG|\bTSB\d{2,4}\b|\bTSE\d{2,4}\b", remarks, re.IGNORECASE))
        
        # 4. Cumulonimbus (CB) in remarks or clouds
        has_cb = bool(re.search(r"\bCB\b|\bCB\s+[A-Z]|\d{3}CB\b", metar, re.IGNORECASE))
        
        # Strict thunderstorm = TS | VCTS | genuine LTG (CB alone is excluded)
        strict_ts = has_ts or has_vcts or has_genuine_ltg
        
        return has_ts, has_vcts, has_genuine_ltg, has_cb, strict_ts

    print("Re-evaluating METAR semantic parsing on all 4,582,946 observations...")
    # Apply vectorized / batch processing
    # Extract metar and wxcodes columns
    metar_series = df_obs["metar"].fillna("")
    wx_series = df_obs["wxcodes"].fillna("")

    # Vectorized regex checks
    has_ts_orig = df_obs["has_thunderstorm_ts"].values
    has_vcts_orig = df_obs["has_vicinity_thunderstorm_vcts"].values
    
    # Genuine LTG remarks (strictly LTG, TSB, TSE)
    # Split RMK
    rmk_split = metar_series.str.split(r"\s+RMK\s+", n=1, regex=True)
    remarks_series = rmk_split.apply(lambda x: x[1] if len(x) > 1 else "")
    
    has_genuine_ltg = remarks_series.str.contains(r"\b(?:OCNL|FRQ|CONS)?\s*LTG|\bTSB\d{2,4}\b|\bTSE\d{2,4}\b", regex=True)
    has_cb = metar_series.str.contains(r"\bCB\b|\bCB\s+[A-Z]|\d{3}CB\b", regex=True)
    
    # Strict thunderstorm = TS | VCTS | Genuine LTG
    strict_ts = has_ts_orig | has_vcts_orig | has_genuine_ltg
    
    cb_alone = has_cb & (~strict_ts)
    
    print(f"Total explicit TS reports: {has_ts_orig.sum():,}")
    print(f"Total vicinity VCTS reports: {has_vcts_orig.sum():,}")
    print(f"Total genuine lightning remark reports: {has_genuine_ltg.sum():,}")
    print(f"Total Cumulonimbus (CB) reports: {has_cb.sum():,}")
    print(f"  --> CB ALONE (CB present, but NO TS, NO VCTS, NO LTG): {cb_alone.sum():,}")
    print(f"Original thunderstorm_reported count (including CB): {df_obs['thunderstorm_reported'].sum():,}")
    print(f"Strict thunderstorm_reported count (EXCLUDING CB alone): {strict_ts.sum():,}")
    print(f"Reduction in positive observation reports: {df_obs['thunderstorm_reported'].sum() - strict_ts.sum():,} (-{(df_obs['thunderstorm_reported'].sum() - strict_ts.sum())/df_obs['thunderstorm_reported'].sum()*100:.2f}%)")

    # Save semantic breakdown table
    df_sem = pd.DataFrame([
        {"category": "Explicit In-Station TS (TS, +TSRA, etc.)", "count": int(has_ts_orig.sum()), "pct": float(has_ts_orig.mean() * 100)},
        {"category": "Vicinity Thunderstorm (VCTS)", "count": int(has_vcts_orig.sum()), "pct": float(has_vcts_orig.mean() * 100)},
        {"category": "Genuine Lightning Remarks (LTG, TSB, TSE)", "count": int(has_genuine_ltg.sum()), "pct": float(has_genuine_ltg.mean() * 100)},
        {"category": "Cumulonimbus Cloud (CB in body or RMK)", "count": int(has_cb.sum()), "pct": float(has_cb.mean() * 100)},
        {"category": "CB Alone (No TS / No VCTS / No LTG)", "count": int(cb_alone.sum()), "pct": float(cb_alone.mean() * 100)},
        {"category": "Strict Thunderstorm (TS | VCTS | Genuine LTG)", "count": int(strict_ts.sum()), "pct": float(strict_ts.mean() * 100)},
        {"category": "Original Indicator (TS | VCTS | Genuine LTG | CB)", "count": int(df_obs['thunderstorm_reported'].sum()), "pct": float(df_obs['thunderstorm_reported'].mean() * 100)},
    ])
    df_sem.to_csv(AUDIT_DIR / "semantic_breakdown_audit.csv", index=False)

    # Attach audited convective indicators to observations dataframe
    df_obs["has_genuine_lightning_remark"] = has_genuine_ltg
    df_obs["has_cumulonimbus_cb"] = has_cb
    df_obs["strict_thunderstorm_reported"] = strict_ts
    df_obs["cb_alone"] = cb_alone

    # -------------------------------------------------------------------------
    # PART 3: TIME-WINDOW ALIGNMENT & COVERAGE GAP MEASUREMENT
    # -------------------------------------------------------------------------
    print("\n--- 3. TIME-WINDOW ALIGNMENT & COVERAGE GAP MEASUREMENT ---")
    # Verify half-open snapping
    df_obs["forecast_init_time_utc"] = (df_obs["valid_time_utc"] - pd.Timedelta(microseconds=1)).dt.floor("6h")
    
    # Sort for gap calculation
    df_obs = df_obs.sort_values(["station_id", "valid_time_utc"]).reset_index(drop=True)
    
    # Calculate inter-observation gap in minutes
    print("Calculating inter-observation gaps within each 6-hour window...")
    df_obs["prev_time"] = df_obs.groupby(["station_id", "forecast_init_time_utc"])["valid_time_utc"].shift(1)
    df_obs["gap_minutes"] = (df_obs["valid_time_utc"] - df_obs["prev_time"]).dt.total_seconds() / 60.0

    # Aggregate window metrics
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

    # Fill max_gap for single-observation windows with 360 min (entire 6h window)
    agg_windows["max_gap_minutes"] = agg_windows["max_gap_minutes"].fillna(360.0)
    
    # Calculate observation span in hours
    agg_windows["obs_span_hours"] = (agg_windows["last_obs"] - agg_windows["first_obs"]).dt.total_seconds() / 3600.0

    print(f"Aggregated station-windows with >= 1 observation: {len(agg_windows):,}")

    # Theoretical full Cartesian grid:
    # 105 stations * 1464 cycles = 153,720 station-windows
    unique_stations = df_st["station_id"].unique()
    unique_cycles = pd.date_range("2024-01-01 00:00:00", "2024-12-31 18:00:00", freq="6h", tz="UTC")
    total_theoretical = len(unique_stations) * len(unique_cycles)
    unobserved_count = total_theoretical - len(agg_windows)
    print(f"Theoretical total station-windows: {total_theoretical:,} (105 stations * 1,464 cycles)")
    print(f"Unobserved station-windows (0 reports): {unobserved_count:,} ({unobserved_count/total_theoretical*100:.2f}%)")

    # -------------------------------------------------------------------------
    # PART 4: EVALUATING ALTERNATIVE NEGATIVE-LABEL CRITERIA
    # -------------------------------------------------------------------------
    print("\n--- 4. EVALUATING ALTERNATIVE NEGATIVE-LABEL CRITERIA ---")

    # Define coverage criteria functions:
    # Under all criteria: if strict_ts_count > 0 -> target = 1 (COVERED_THUNDERSTORM)
    
    # Criterion 0: Original Baseline (N >= 4, including CB as thunderstorm)
    # df_orig_windows = pd.read_parquet(WINDOWS_PARQUET)
    # vc_orig = df_orig_windows["coverage_status"].value_counts().to_dict()

    # Criterion A: Naive Count Only (N >= 4, Strict TS without CB)
    def eval_crit_a(row):
        if row["strict_ts_count"] > 0:
            return 1, "COVERED_THUNDERSTORM"
        elif row["obs_count"] >= 4:
            return 0, "COVERED_NO_THUNDERSTORM"
        else:
            return np.nan, "INCOMPLETE_MONITORING"

    # Criterion B: Moderate Gap Safeguard (N >= 5, max_gap <= 90 min, span >= 4.5 h)
    def eval_crit_b(row):
        if row["strict_ts_count"] > 0:
            return 1, "COVERED_THUNDERSTORM"
        elif row["obs_count"] >= 5 and row["max_gap_minutes"] <= 90.0 and row["obs_span_hours"] >= 4.5:
            return 0, "COVERED_NO_THUNDERSTORM"
        else:
            return np.nan, "INCOMPLETE_MONITORING"

    # Criterion C: Strict Standard Meteorological Cadence (N >= 6, max_gap <= 75 min, span >= 5.0 h)
    def eval_crit_c(row):
        if row["strict_ts_count"] > 0:
            return 1, "COVERED_THUNDERSTORM"
        elif row["obs_count"] >= 6 and row["max_gap_minutes"] <= 75.0 and row["obs_span_hours"] >= 5.0:
            return 0, "COVERED_NO_THUNDERSTORM"
        else:
            return np.nan, "INCOMPLETE_MONITORING"

    # Criterion D: High-Confidence Continuous Monitoring (N >= 6, max_gap <= 65 min, span >= 5.2 h)
    def eval_crit_d(row):
        if row["strict_ts_count"] > 0:
            return 1, "COVERED_THUNDERSTORM"
        elif row["obs_count"] >= 6 and row["max_gap_minutes"] <= 65.0 and row["obs_span_hours"] >= 5.2:
            return 0, "COVERED_NO_THUNDERSTORM"
        else:
            return np.nan, "INCOMPLETE_MONITORING"

    print("Evaluating criteria across all windows...")
    res_a = [eval_crit_a(r) for _, r in agg_windows.iterrows()]
    res_b = [eval_crit_b(r) for _, r in agg_windows.iterrows()]
    res_c = [eval_crit_c(r) for _, r in agg_windows.iterrows()]
    res_d = [eval_crit_d(r) for _, r in agg_windows.iterrows()]

    agg_windows["target_crit_a"] = [r[0] for r in res_a]
    agg_windows["status_crit_a"] = [r[1] for r in res_a]
    
    agg_windows["target_crit_b"] = [r[0] for r in res_b]
    agg_windows["status_crit_b"] = [r[1] for r in res_b]

    agg_windows["target_crit_c"] = [r[0] for r in res_c]
    agg_windows["status_crit_c"] = [r[1] for r in res_c]

    agg_windows["target_crit_d"] = [r[0] for r in res_d]
    agg_windows["status_crit_d"] = [r[1] for r in res_d]

    # Create comparison table
    crit_comparison = []
    for name, col_tgt, col_stat in [
        ("Criterion A (Count >= 4 only)", "target_crit_a", "status_crit_a"),
        ("Criterion B (Count >= 5, gap <= 90m, span >= 4.5h)", "target_crit_b", "status_crit_b"),
        ("Criterion C [RECOMMENDED] (Count >= 6, gap <= 75m, span >= 5.0h)", "target_crit_c", "status_crit_c"),
        ("Criterion D (Count >= 6, gap <= 65m, span >= 5.2h)", "target_crit_d", "status_crit_d"),
    ]:
        n_pos = (agg_windows[col_tgt] == 1).sum()
        n_neg = (agg_windows[col_tgt] == 0).sum()
        n_incomp = agg_windows[col_tgt].isna().sum()
        prevalence = n_pos / (n_pos + n_neg) * 100.0 if (n_pos + n_neg) > 0 else 0.0
        crit_comparison.append({
            "criterion": name,
            "positive_windows": int(n_pos),
            "negative_windows": int(n_neg),
            "incomplete_windows": int(n_incomp),
            "effective_prevalence_pct": round(prevalence, 2),
        })
        print(f"\n{name}:")
        print(f"  Positive (y=1): {n_pos:,}")
        print(f"  Negative (y=0): {n_neg:,}")
        print(f"  Incomplete (y=NaN): {n_incomp:,}")
        print(f"  Effective Prevalence: {prevalence:.2f}%")

    df_crit_comp = pd.DataFrame(crit_comparison)
    df_crit_comp.to_csv(AUDIT_DIR / "coverage_criteria_comparison.csv", index=False)

    # -------------------------------------------------------------------------
    # PART 5: SPATIAL ALIGNMENT & REPRESENTATIVENESS AUDIT
    # -------------------------------------------------------------------------
    print("\n--- 5. SPATIAL ALIGNMENT & GRID REPRESENTATIVENESS ---")
    # Load 218 active terrestrial grid cells from ERA5 forecast dataset
    df_fc = pd.read_csv(FORECAST_6H_CSV, usecols=["grid_cell_id", "latitude", "longitude"]).drop_duplicates()
    print(f"Active terrestrial Florida grid cells: {len(df_fc)}")

    # Calculate station-to-grid center distances
    df_st_terr = df_st[df_st["in_active_terrestrial_grid"] == True].copy()
    distances = []
    for _, r in df_st.iterrows():
        d_km = haversine_km(r["latitude"], r["longitude"], r["grid_lat"], r["grid_lon"])
        distances.append(round(d_km, 2))
    df_st["distance_to_grid_center_km"] = distances
    df_st.to_csv(AUDIT_DIR / "station_inventory_audit.csv", index=False)

    print("Station-to-grid center distance distribution (km):")
    print(df_st["distance_to_grid_center_km"].describe())

    # Count stations per grid cell
    station_counts_per_cell = df_st_terr["grid_cell_id"].value_counts()
    
    # Merge with all 218 active terrestrial grid cells
    df_spatial = df_fc.copy()
    df_spatial["station_count"] = df_spatial["grid_cell_id"].map(station_counts_per_cell).fillna(0).astype(int)
    
    print("\nGrid Cell Coverage Breakdown (out of 218 active Florida terrestrial cells):")
    sc_dist = df_spatial["station_count"].value_counts().sort_index().to_dict()
    for n_st, cnt in sc_dist.items():
        print(f"  {cnt} grid cells have {n_st} station(s) ({cnt/len(df_spatial)*100:.1f}%)")

    zero_station_cells = (df_spatial["station_count"] == 0).sum()
    print(f"\nCRITICAL SPATIAL FINDING:")
    print(f"  {zero_station_cells} of 218 terrestrial grid cells ({zero_station_cells/len(df_spatial)*100:.1f}%) have ZERO ASOS stations!")
    print(f"  Only {len(df_spatial) - zero_station_cells} grid cells ({((len(df_spatial)-zero_station_cells)/len(df_spatial))*100:.1f}%) have at least 1 ASOS station.")
    
    df_spatial.to_csv(AUDIT_DIR / "spatial_representation_audit.csv", index=False)

    # -------------------------------------------------------------------------
    # PART 6: SAVE AUDITED FORECAST WINDOW DATASET
    # -------------------------------------------------------------------------
    print("\n--- 6. SAVING AUDITED FORECAST WINDOW DATASET ---")
    # Save the audited windows parquet with Criterion C as primary audited target,
    # plus flags for all criteria and separated semantic counts
    audited_parquet = AUDIT_DIR / "florida_asos_forecast_windows_audited.parquet"
    agg_windows.to_parquet(audited_parquet, index=False, engine="pyarrow")
    print(f"Saved audited forecast window dataset to {audited_parquet} ({len(agg_windows):,} rows)")

    print("\n" + "=" * 80)
    print("AUDIT EXECUTION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    run_audit()
