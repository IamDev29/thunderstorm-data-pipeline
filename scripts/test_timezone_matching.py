import pandas as pd
import numpy as np
from datetime import timezone, timedelta
from pathlib import Path

# Load Florida grid cells
master_cells = pd.read_csv('data/processed/florida_thunderstorm_2024_master.csv', usecols=['latitude', 'longitude']).drop_duplicates()
grid_coords = master_cells[['latitude', 'longitude']].values
print(f"Total Florida grid cells: {len(grid_coords)}")

df_raw = pd.read_csv('data/raw/details/StormEvents_details-ftp_v1.0_d2024_c20260728.csv.gz', low_memory=False)
fl = df_raw[df_raw['STATE'].str.upper() == 'FLORIDA'].copy()
CONVECTIVE_TYPES = {'Thunderstorm Wind', 'Hail', 'Tornado', 'Lightning'}
fl_conv = fl[fl['EVENT_TYPE'].isin(CONVECTIVE_TYPES)].copy()

tz_offsets = {'EST-5': 5, 'CST-6': 6, 'EDT-4': 4, 'CDT-5': 5}

# Compute uncorrected vs corrected
records = []
for _, r in fl_conv.iterrows():
    local_str = r['BEGIN_DATE_TIME']
    tz_str = r['CZ_TIMEZONE']
    e_lat = float(r['BEGIN_LAT'])
    e_lon = float(r['BEGIN_LON'])
    
    dt_local = pd.to_datetime(local_str, format='%d-%b-%y %H:%M:%S')
    
    # Old uncorrected method: naive local as UTC
    old_utc = dt_local
    old_anchor = int(np.round(old_utc.hour / 6.0) * 6)
    if old_anchor == 24:
        old_era5_t = old_utc.floor('D') + timedelta(days=1)
    else:
        old_era5_t = old_utc.floor('D') + timedelta(hours=old_anchor)
        
    # Corrected method:
    offset_h = tz_offsets[tz_str]
    tz_obj = timezone(timedelta(hours=-offset_h), name=tz_str[:3])
    aware_dt = dt_local.replace(tzinfo=tz_obj)
    utc_aware = aware_dt.astimezone(timezone.utc)
    utc_naive = utc_aware.replace(tzinfo=None)
    
    # Nearest 6h timestamp:
    new_era5_t = utc_naive.round('6h')
    time_diff_min = abs((utc_naive - new_era5_t).total_seconds()) / 60.0
    
    # Spatial distance to nearest ERA5 cell (Haversine)
    dlat = np.radians(grid_coords[:, 0] - e_lat)
    dlon = np.radians(grid_coords[:, 1] - e_lon)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(e_lat)) * np.cos(np.radians(grid_coords[:, 0])) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    dist_km = 6371.0 * c
    min_idx = np.argmin(dist_km)
    min_dist = dist_km[min_idx]
    matched_lat = grid_coords[min_idx, 0]
    matched_lon = grid_coords[min_idx, 1]
    
    time_changed = (old_era5_t != new_era5_t)
    
    records.append({
        'event_id': r['EVENT_ID'],
        'event_type': r['EVENT_TYPE'],
        'local_timestamp': dt_local,
        'cz_timezone': tz_str,
        'old_era5_t': old_era5_t,
        'new_utc': utc_naive,
        'new_era5_t': new_era5_t,
        'time_diff_min': time_diff_min,
        'dist_km': min_dist,
        'matched_lat': matched_lat,
        'matched_lon': matched_lon,
        'time_changed': time_changed
    })

df_comp = pd.DataFrame(records)
print(f"Total convective events analyzed: {len(df_comp)}")
changed_count = df_comp['time_changed'].sum()
print(f"Events with ERA5 timestamp changed by timezone fix: {changed_count} ({changed_count/len(df_comp)*100:.1f}%)")
print("\nSample comparisons:")
print(df_comp[['event_id', 'cz_timezone', 'local_timestamp', 'old_era5_t', 'new_utc', 'new_era5_t', 'time_changed']].head(10))

# Check how many distinct (new_era5_t, matched_lat, matched_lon) pairs are formed
unique_old_pairs = df_comp[['old_era5_t', 'matched_lat', 'matched_lon']].drop_duplicates()
unique_new_pairs = df_comp[['new_era5_t', 'matched_lat', 'matched_lon']].drop_duplicates()
print(f"\nUnique (timestamp, cell) pairs: Old = {len(unique_old_pairs)}, New = {len(unique_new_pairs)}")
