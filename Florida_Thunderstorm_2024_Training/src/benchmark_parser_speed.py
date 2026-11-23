import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0, str(Path("Florida_Thunderstorm_2024_Training").resolve()))
sys.path.insert(0, str(Path(".").resolve()))
from Florida_Thunderstorm_2024_Training.src.parse_metar_robust import parse_metar_thunderstorm_robust

print("Loading 50,000 row sample...")
df_sample = pd.read_parquet(
    "data/processed/asos_repair/florida_asos_observations_repaired_2024.parquet",
    columns=["metar", "wxcodes", "station_id"]
).head(50000)

t0 = time.time()
metar_col = df_sample["metar"].values
wx_col = df_sample["wxcodes"].values
st_col = df_sample["station_id"].values

results = [
    parse_metar_thunderstorm_robust(m, w, s)
    for m, w, s in zip(metar_col, wx_col, st_col)
]
dur = time.time() - t0
rate = len(df_sample) / dur
print(f"Parsed 50,000 rows in {dur:.2f}s ({rate:,.0f} rows/s).")
est_full = 5728432 / rate
print(f"Estimated time for 5.7M rows: {est_full:.1f}s (~{est_full/60:.1f} minutes).")
