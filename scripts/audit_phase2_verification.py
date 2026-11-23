import pandas as pd
import numpy as np

def run_phase2_audit():
    print("=" * 80)
    print("PHASE 2: VERIFICATION OF THE FINAL MODELING DATASET")
    print("=" * 80)

    # 1. Observations dataset audit
    obs_path = "data/processed/asos_parser_fixed/florida_asos_observations_fixed_2024.parquet"
    print(f"\n1. Auditing observations dataset: {obs_path}")
    df_obs = pd.read_parquet(obs_path)
    total_obs = len(df_obs)
    print(f"   Total observation rows: {total_obs:,}")

    # Calendar dates
    df_obs["valid_dt"] = pd.to_datetime(df_obs["valid"])
    dates = df_obs["valid_dt"].dt.strftime("%Y-%m-%d").unique()
    n_dates = len(dates)
    print(f"   Unique calendar dates: {n_dates} (Expected: 366)")
    assert n_dates == 366, f"Expected 366 dates, found {n_dates}"

    # Duplicate keys
    dups = df_obs.duplicated(subset=["station_id", "valid"]).sum()
    print(f"   Duplicate station-timestamp keys: {dups} (Expected: 0)")
    assert dups == 0, f"Found {dups} duplicate observations"

    # DTS and TTS check
    dts_obs = df_obs[df_obs["station_id"] == "DTS"]
    dts_ts = int(dts_obs["strict_thunderstorm_reported"].sum())
    tts_obs = df_obs[df_obs["station_id"] == "TTS"]
    tts_ts = int(tts_obs["strict_thunderstorm_reported"].sum())
    print(f"   DTS: {dts_ts:,} TS / {len(dts_obs):,} total obs ({dts_ts / len(dts_obs) * 100:.2f}%)")
    print(f"   TTS: {tts_ts:,} TS / {len(tts_obs):,} total obs ({tts_ts / len(tts_obs) * 100:.2f}%)")
    assert dts_ts < len(dts_obs), "DTS still 100% positive"
    assert tts_ts < len(tts_obs), "TTS still 100% positive"

    # CB alone check
    cb_alone_ts = df_obs[(df_obs["cb_alone"] == True) & (df_obs["strict_thunderstorm_reported"] == 1)]
    print(f"   CB-alone observations marked as strict TS: {len(cb_alone_ts)} (Expected: 0)")
    assert len(cb_alone_ts) == 0, f"Found {len(cb_alone_ts)} CB-alone rows marked as TS"

    # 2. Modeling table audit
    mod_path = "data/processed/asos_parser_fixed/florida_asos_modeling_ready_fixed_2024.parquet"
    print(f"\n2. Auditing modeling dataset: {mod_path}")
    df_mod = pd.read_parquet(mod_path)
    total_mod = len(df_mod)
    print(f"   Total rows in modeling table: {total_mod:,}")

    # Row multiplication / join check
    dups_mod = df_mod.duplicated(subset=["station_id", "forecast_init_time_utc"]).sum()
    print(f"   Duplicate station + forecast_init_time_utc: {dups_mod} (Expected: 0)")
    assert dups_mod == 0, f"Found {dups_mod} duplicated keys in modeling table"

    # Missing / Incomplete windows check
    incomplete_mask = df_mod["target_asos_strict_crit_c"].isna()
    n_incomplete = incomplete_mask.sum()
    print(f"   Incomplete / unmonitored windows (NaN count): {n_incomplete:,} ({n_incomplete / total_mod * 100:.2f}%)")
    assert n_incomplete > 0, "Incomplete windows were silently zero-filled!"
    status_counts = df_mod.loc[incomplete_mask, "status_crit_c"].value_counts()
    print("   Status breakdown of unmonitored windows:")
    for status, cnt in status_counts.items():
        print(f"     - {status}: {cnt:,}")

    # Monitored rows
    df_mon = df_mod[~incomplete_mask].copy()
    n_mon = len(df_mon)
    print(f"   Monitored windows: {n_mon:,}")

    # 3. Predictor leakage audit
    print("\n3. Auditing predictor leakage:")
    # Check predictor timestamps and definitions
    # Forecast init is t. ERA5 features must be from t.
    # Check if ERA5 time matches forecast_init_time_utc
    t_era5 = pd.to_datetime(df_mod["forecast_init_time_utc_era5"], utc=True)
    era5_time_diff = (t_era5 - df_mod["forecast_init_time_utc"]).dt.total_seconds().abs().max()
    print(f"   Max absolute difference between ERA5 time and forecast_init_time_utc: {era5_time_diff}s")
    assert era5_time_diff == 0.0, "ERA5 predictors are not aligned with forecast initialization time!"

    # Check persistence lag feature
    # Must use prior_strict_ts_6h which looks back to [t-6h, t]
    print(f"   Persistence lag feature: 'prior_strict_ts_6h'")
    print(f"   Prior positive count in monitored data: {df_mon['prior_strict_ts_6h'].sum():,}")

    # 4. Chronological Splits audit
    print("\n4. Auditing chronological train/validation/test split:")
    train_mask = (df_mon["forecast_init_time_utc"] < "2024-08-01")
    val_mask = (df_mon["forecast_init_time_utc"] >= "2024-08-01") & (df_mon["forecast_init_time_utc"] < "2024-10-01")
    test_mask = (df_mon["forecast_init_time_utc"] >= "2024-10-01")

    n_train = train_mask.sum()
    n_val = val_mask.sum()
    n_test = test_mask.sum()
    assert n_train + n_val + n_test == n_mon, "Splits do not partition monitored rows!"

    y_train = df_mon.loc[train_mask, "target_asos_strict_crit_c"]
    y_val = df_mon.loc[val_mask, "target_asos_strict_crit_c"]
    y_test = df_mon.loc[test_mask, "target_asos_strict_crit_c"]

    print(f"   Train split (Jan 1 - Jul 31): {n_train:,} rows, Positives={int(y_train.sum()):,} ({y_train.mean()*100:.2f}%)")
    print(f"   Val split   (Aug 1 - Sep 30): {n_val:,} rows, Positives={int(y_val.sum()):,} ({y_val.mean()*100:.2f}%)")
    print(f"   Test split  (Oct 1 - Dec 31): {n_test:,} rows, Positives={int(y_test.sum()):,} ({y_test.mean()*100:.2f}%)")

    # 5. Station representation in splits
    st_train = df_mon.loc[train_mask, "station_id"].nunique()
    st_val = df_mon.loc[val_mask, "station_id"].nunique()
    st_test = df_mon.loc[test_mask, "station_id"].nunique()
    print(f"   Active stations: Train={st_train}, Val={st_val}, Test={st_test}")

    print("\nALL PHASE 2 CHECKS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_phase2_audit()
