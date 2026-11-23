import pandas as pd

df_obs = pd.read_parquet("data/processed/asos_repair/florida_asos_observations_repaired_2024.parquet")
print("Columns:", df_obs.columns.tolist())
for st in ["DTS", "TTS"]:
    sub = df_obs[df_obs["station_id"] == st]
    print(f"\nStation {st}: obs={len(sub)}, ts_count={sub['has_thunderstorm_ts'].sum()}, strict_ts={sub['strict_thunderstorm_reported'].sum()}")
    for idx, r in sub.head(5).iterrows():
        print(f"  valid={r['valid_time_utc']}")
        for col in ["wxcodes", "metar", "presentwx"]:
            if col in r:
                print(f"    {col}: {r[col]}")
