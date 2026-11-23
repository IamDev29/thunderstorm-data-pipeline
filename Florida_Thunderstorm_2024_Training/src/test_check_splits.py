import pandas as pd
import numpy as np

df = pd.read_parquet("data/processed/asos_repair/florida_asos_modeling_ready_repaired_2024.parquet")
print("Total rows:", len(df))

# Monitored rows under criterion c
df_eval = df[df["target_asos_strict_crit_c"].notna()].copy()
df_eval["target"] = df_eval["target_asos_strict_crit_c"].astype(int)
print("Monitored rows (target not null):", len(df_eval))

# Persistence calculation
df_eval = df_eval.sort_values(["station_id", "forecast_init_time_utc"]).reset_index(drop=True)
df_eval["prev_init"] = df_eval.groupby("station_id")["forecast_init_time_utc"].shift(1)
df_eval["prev_target"] = df_eval.groupby("station_id")["target"].shift(1)
df_eval["cycle_diff_hours"] = (df_eval["forecast_init_time_utc"] - df_eval["prev_init"]).dt.total_seconds() / 3600.0
df_eval["pred_persistence"] = np.where(df_eval["cycle_diff_hours"] == 6.0, df_eval["prev_target"], 0).astype(int)

# Check splits
train_mask = df_eval["forecast_init_time_utc"] < "2024-08-01"
val_mask = (df_eval["forecast_init_time_utc"] >= "2024-08-01") & (df_eval["forecast_init_time_utc"] < "2024-10-01")
test_mask = df_eval["forecast_init_time_utc"] >= "2024-10-01"

print(f"Train (Jan-Jul): {train_mask.sum()} rows, Pos={df_eval.loc[train_mask, 'target'].sum()} ({df_eval.loc[train_mask, 'target'].mean()*100:.2f}%)")
print(f"Val   (Aug-Sep): {val_mask.sum()} rows, Pos={df_eval.loc[val_mask, 'target'].sum()} ({df_eval.loc[val_mask, 'target'].mean()*100:.2f}%)")
print(f"Test  (Oct-Dec): {test_mask.sum()} rows, Pos={df_eval.loc[test_mask, 'target'].sum()} ({df_eval.loc[test_mask, 'target'].mean()*100:.2f}%)")
print(f"Total evaluated rows: {train_mask.sum() + val_mask.sum() + test_mask.sum()}")
