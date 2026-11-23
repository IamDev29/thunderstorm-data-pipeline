"""
Diagnostic audit script for Phase 2 and Phase 3:
1. Target reconciliation: source events, eligible events, matched events, unique forecast positive keys.
2. Temporal alignment audit: exactly what events map into future_noaa_event_6h and prior_event_at_init.
3. Event ID overlap check: did the same event ID create both prior_event_at_init=1 and future_noaa_event_6h=1?
"""

from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

def haversine_np(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))

def main():
    print("=== AUDIT OF FORECAST TARGET AND PERSISTENCE BASELINE ===")
    
    noaa_csv = Path("Florida_Thunderstorm_2024_Training/data/processed/noaa_florida_2024.csv")
    df_noaa = pd.read_csv(noaa_csv)
    print(f"Total raw processed NOAA rows: {len(df_noaa)}")
    
    conv_types = {"Thunderstorm Wind", "Hail", "Tornado", "Lightning"}
    df_conv = df_noaa[df_noaa["event_type"].isin(conv_types)].copy()
    print(f"Convective type events: {len(df_conv)}")
    
    df_coords = df_conv[df_conv["coordinates_available"]].copy()
    print(f"Convective events with coordinates: {len(df_coords)}")
    
    master_csv = Path("data/processed/florida_thunderstorm_2024_master.csv")
    df_master = pd.read_csv(master_csv, low_memory=False)
    grid_coords = df_master[["latitude", "longitude"]].drop_duplicates().values
    print(f"Unique Florida grid cells in master: {len(grid_coords)}")
    
    df_coords["timestamp_utc"] = pd.to_datetime(df_coords["timestamp_utc"])
    
    # Check spatial distance and window mapping
    fwd_records = []
    same_time_records = []
    
    for _, r in df_coords.iterrows():
        e_t = r["timestamp_utc"]
        e_lat, e_lon = float(r["latitude"]), float(r["longitude"])
        eid = str(r["event_id"])
        etype = str(r["event_type"])
        
        # Spatial match
        dists = haversine_np(grid_coords[:, 0], grid_coords[:, 1], e_lat, e_lon)
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        grid_lat = grid_coords[min_idx, 0]
        grid_lon = grid_coords[min_idx, 1]
        
        # 1. Forward 6h window mapping: (t, t + 6h]
        t_floor = e_t.floor("6h")
        if e_t == t_floor:
            t_init = t_floor - timedelta(hours=6)
        else:
            t_init = t_floor
        t_end = t_init + timedelta(hours=6)
        
        # 2. Master same-time mapping: synoptic_t = round("6h")
        synoptic_t = e_t.round("6h")
        t_diff_min = abs((e_t - synoptic_t).total_seconds()) / 60.0
        
        fwd_records.append({
            "event_id": eid,
            "event_type": etype,
            "timestamp_utc": e_t,
            "min_dist_km": min_dist,
            "grid_lat": grid_lat,
            "grid_lon": grid_lon,
            "fwd_t_init": t_init,
            "fwd_t_end": t_end,
            "synoptic_t": synoptic_t,
            "synoptic_diff_min": t_diff_min,
            "is_fwd_matched": (min_dist <= 28.0),
            "is_synoptic_matched": (min_dist <= 28.0) and (t_diff_min <= 180.0)
        })
        
    df_ev = pd.DataFrame(fwd_records)
    print(f"\n--- Event Matching Reconciliation ---")
    print(f"Total convective coordinate events: {len(df_ev)}")
    print(f"Forward matched (dist <= 28km): {df_ev['is_fwd_matched'].sum()}")
    print(f"Forward excluded (dist > 28km): {(~df_ev['is_fwd_matched']).sum()}")
    print(f"Synoptic matched (dist <= 28km & dt <= 3h): {df_ev['is_synoptic_matched'].sum()}")
    
    # Unique positive keys for forward
    fwd_matched = df_ev[df_ev["is_fwd_matched"]].copy()
    fwd_keys = fwd_matched.groupby(["fwd_t_init", "grid_lat", "grid_lon"]).agg(
        event_count=("event_id", "count"),
        event_ids=("event_id", list)
    ).reset_index()
    print(f"Unique forward forecast positive keys (t_init, lat, lon): {len(fwd_keys)}")
    
    # Check overlap between prior_event_at_init and future_noaa_event_6h
    # In master, target_noaa_event is at synoptic_t
    syn_matched = df_ev[df_ev["is_synoptic_matched"]].copy()
    syn_keys = syn_matched.groupby(["synoptic_t", "grid_lat", "grid_lon"]).agg(
        event_count=("event_id", "count"),
        event_ids=("event_id", list)
    ).reset_index()
    print(f"Unique synoptic positive keys (synoptic_t, lat, lon): {len(syn_keys)}")
    
    # Now check in the actual forecasting dataset
    fc_csv = Path("data/processed/florida_thunderstorm_2024_forecasting_6h.csv")
    df_fc = pd.read_csv(fc_csv, parse_dates=["forecast_init_time_utc"], low_memory=False)
    
    # Filter to test split (Months 9-12)
    test_fc = df_fc[df_fc["month"].isin([9, 10, 11, 12]) & (df_fc["forecast_window_status"] != "TRUNCATED_YEAR_END")].copy()
    print(f"\n--- Test Split (Months 09-12) Persistence Analysis ---")
    print(f"Total test rows: {len(test_fc)}")
    print(f"Test positive future targets (future_noaa_event_6h == 1): {(test_fc['future_noaa_event_6h'] == 1).sum()}")
    print(f"Test prior events at init (prior_event_at_init == 1): {(test_fc['prior_event_at_init'] == 1).sum()}")
    
    # Overlap
    tp_pers = ((test_fc["future_noaa_event_6h"] == 1) & (test_fc["prior_event_at_init"] == 1)).sum()
    fp_pers = ((test_fc["future_noaa_event_6h"] == 0) & (test_fc["prior_event_at_init"] == 1)).sum()
    fn_pers = ((test_fc["future_noaa_event_6h"] == 1) & (test_fc["prior_event_at_init"] == 0)).sum()
    tn_pers = ((test_fc["future_noaa_event_6h"] == 0) & (test_fc["prior_event_at_init"] == 0)).sum()
    print(f"Persistence Confusion Matrix on Test Split: TP={tp_pers}, FP={fp_pers}, FN={fn_pers}, TN={tn_pers}")
    
    # Inspect TP cases: where both prior_event_at_init == 1 and future_noaa_event_6h == 1
    tp_rows = test_fc[(test_fc["future_noaa_event_6h"] == 1) & (test_fc["prior_event_at_init"] == 1)]
    print(f"\nAnalyzing the {len(tp_rows)} Persistence Hits (True Positives)...")
    
    # For each TP row, check the event IDs in master at that time vs event IDs in forward target
    same_id_count = 0
    different_id_count = 0
    detailed_overlap = []
    
    for _, r in tp_rows.iterrows():
        t_init = r["forecast_init_time_utc"]
        lat = r["latitude"]
        lon = r["longitude"]
        fwd_ids = set(str(r["future_noaa_event_ids"]).split(";"))
        
        # What events caused prior_event_at_init=1?
        # That was target_noaa_event at valid_time_utc = t_init in master!
        m_row = df_master[(df_master["valid_time_utc"] == str(t_init)) & (df_master["latitude"] == lat) & (df_master["longitude"] == lon)]
        if not m_row.empty:
            prior_ids = set(str(m_row.iloc[0]["noaa_event_ids"]).split(";"))
        else:
            prior_ids = set()
            
        common = fwd_ids.intersection(prior_ids)
        if len(common) > 0:
            same_id_count += 1
        else:
            different_id_count += 1
            
        detailed_overlap.append({
            "t_init": t_init,
            "lat": lat,
            "lon": lon,
            "fwd_ids": fwd_ids,
            "prior_ids": prior_ids,
            "common_ids": common
        })
        
    print(f"Of {len(tp_rows)} persistence hits:")
    print(f"  - Hits sharing the EXACT SAME NOAA event ID: {same_id_count} ({(same_id_count/len(tp_rows)*100):.1f}%)")
    print(f"  - Hits with DIFFERENT event IDs (continuation/separate reports): {different_id_count} ({(different_id_count/len(tp_rows)*100):.1f}%)")
    
    if detailed_overlap:
        print("\nSample Overlap Events:")
        for sample in detailed_overlap[:5]:
            print(f"  t_init={sample['t_init']}, cell=({sample['lat']}, {sample['lon']}):")
            print(f"    Forward IDs: {sample['fwd_ids']}")
            print(f"    Prior IDs:   {sample['prior_ids']}")
            print(f"    Shared IDs:  {sample['common_ids']}")
            # Look up timestamp of shared ID
            for cid in sample['common_ids']:
                ev_row = df_coords[df_coords["event_id"].astype(str) == str(cid)]
                if not ev_row.empty:
                    print(f"    Event {cid} actual timestamp_utc: {ev_row.iloc[0]['timestamp_utc']}")

if __name__ == "__main__":
    main()
