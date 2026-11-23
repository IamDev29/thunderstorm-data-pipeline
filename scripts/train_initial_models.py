import os
import sys
import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.calibration import calibration_curve
import lightgbm as lgb
import xgboost as xgb

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    return {
        "threshold": round(threshold, 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision": round(prec, 6), "recall": round(rec, 6), "f1": round(f1, 6),
        "pr_auc": round(pr_auc, 6), "roc_auc": round(roc_auc, 6),
        "brier_score": round(brier, 6)
    }

def find_optimal_threshold(y_true, y_prob, metric="f1"):
    best_thresh = 0.5
    best_score = -1.0
    for thresh in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= thresh).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "youden":
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sens + spec - 1.0
        if score > best_score:
            best_score = score
            best_thresh = thresh
    return float(best_thresh), float(best_score)

def main():
    print("=" * 80)
    print("PHASE 3: TRAINING INITIAL SUPERVISED MODELS FOR FLORIDA ASOS FORECASTING")
    print("=" * 80)

    # Directories
    root_dir = Path(".").resolve()
    models_dir = root_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = root_dir / "data" / "processed" / "model_evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    data_path = root_dir / "data" / "processed" / "asos_parser_fixed" / "florida_asos_modeling_ready_fixed_2024.parquet"
    print(f"Loading modeling dataset from {data_path}...")
    df = pd.read_parquet(data_path)

    # Strict Criterion C filtering
    df_mon = df[df["target_asos_strict_crit_c"].notna()].copy()
    df_mon["target"] = df_mon["target_asos_strict_crit_c"].astype(int)
    print(f"Total monitored rows: {len(df_mon):,}")

    # Feature engineering: Diurnal cycle encoding
    df_mon["sin_hour"] = np.sin(2.0 * np.pi * df_mon["hour_utc"] / 24.0)
    df_mon["cos_hour"] = np.cos(2.0 * np.pi * df_mon["hour_utc"] / 24.0)

    feature_cols = [
        # Thermodynamic & Instability
        "cape_J_kg", "cin_J_kg", "dewpoint_depression_degC", "tcwv_kg_m2",
        "total_precipitation_mm", "convective_precipitation_mm", "cloud_cover_fraction",
        # Surface observations & Pressure
        "temperature_2m_degC", "dewpoint_2m_degC", "surface_pressure_hPa", "mslp_hPa",
        "wind_speed_10m_ms", "wind_direction_10m_deg",
        # Upper Air / Vertical Profile
        "temperature_850hPa_degC", "relative_humidity_850hPa_pct", "wind_speed_850hPa_ms",
        "temperature_500hPa_degC", "relative_humidity_500hPa_pct",
        "bulk_shear_850_300_ms", "temp_diff_850_500_degC", "era5_thunderstorm_favorability",
        # Spatial & Diurnal
        "latitude", "longitude", "sin_hour", "cos_hour",
        # Convective Station State / Persistence Lag
        "prior_strict_ts_6h"
    ]
    print(f"Selected {len(feature_cols)} predictor features: {feature_cols}")

    # Chronological Splits
    train_mask = (df_mon["forecast_init_time_utc"] < "2024-08-01")
    val_mask = (df_mon["forecast_init_time_utc"] >= "2024-08-01") & (df_mon["forecast_init_time_utc"] < "2024-10-01")
    test_mask = (df_mon["forecast_init_time_utc"] >= "2024-10-01")

    df_train = df_mon[train_mask].copy()
    df_val = df_mon[val_mask].copy()
    df_test = df_mon[test_mask].copy()

    X_train = df_train[feature_cols].copy()
    y_train = df_train["target"].values

    X_val = df_val[feature_cols].copy()
    y_val = df_val["target"].values

    X_test = df_test[feature_cols].copy()
    y_test = df_test["target"].values

    print(f"Train set: {len(X_train):,} rows, {y_train.sum():,} positives ({y_train.mean()*100:.2f}%)")
    print(f"Val set:   {len(X_val):,} rows, {y_val.sum():,} positives ({y_val.mean()*100:.2f}%)")
    print(f"Test set:  {len(X_test):,} rows, {y_test.sum():,} positives ({y_test.mean()*100:.2f}%)")

    # Verify no missing values
    assert X_train.isna().sum().sum() == 0, "Train features have NaNs"
    assert X_val.isna().sum().sum() == 0, "Val features have NaNs"
    assert X_test.isna().sum().sum() == 0, "Test features have NaNs"

    # Preprocessing for linear model: StandardScaler fitted STRICTLY on train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, models_dir / "scaler_logistic_regression.joblib")

    # -------------------------------------------------------------
    # 1. Model 1: Logistic Regression
    # -------------------------------------------------------------
    print("\n--- Training Model 1: Logistic Regression ---")
    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42, solver="lbfgs")
    lr.fit(X_train_scaled, y_train)

    val_prob_lr = lr.predict_proba(X_val_scaled)[:, 1]
    test_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

    opt_thresh_lr, val_f1_lr = find_optimal_threshold(y_val, val_prob_lr, metric="f1")
    print(f"Logistic Regression: Optimal Val Threshold={opt_thresh_lr:.4f} (Val F1={val_f1_lr:.4f})")

    joblib.dump(lr, models_dir / "logistic_regression.joblib")

    # -------------------------------------------------------------
    # 2. Model 2: LightGBM
    # -------------------------------------------------------------
    print("\n--- Training Model 2: LightGBM ---")
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_estimators": 1000,
        "verbose": -1
    }

    lgb_clf = lgb.LGBMClassifier(**lgb_params)
    lgb_clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    print(f"LightGBM best iteration: {lgb_clf.best_iteration_}")

    val_prob_lgb = lgb_clf.predict_proba(X_val)[:, 1]
    test_prob_lgb = lgb_clf.predict_proba(X_test)[:, 1]

    opt_thresh_lgb, val_f1_lgb = find_optimal_threshold(y_val, val_prob_lgb, metric="f1")
    print(f"LightGBM: Optimal Val Threshold={opt_thresh_lgb:.4f} (Val F1={val_f1_lgb:.4f})")

    joblib.dump(lgb_clf, models_dir / "lightgbm_model.joblib")
    lgb_clf.booster_.save_model(str(models_dir / "lightgbm_model.txt"))

    # -------------------------------------------------------------
    # 3. Model 3: XGBoost
    # -------------------------------------------------------------
    print("\n--- Training Model 3: XGBoost ---")
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_estimators": 1000,
        "early_stopping_rounds": 50
    }

    xgb_clf = xgb.XGBClassifier(**xgb_params)
    xgb_clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    print(f"XGBoost best iteration: {xgb_clf.best_iteration}")

    val_prob_xgb = xgb_clf.predict_proba(X_val)[:, 1]
    test_prob_xgb = xgb_clf.predict_proba(X_test)[:, 1]

    opt_thresh_xgb, val_f1_xgb = find_optimal_threshold(y_val, val_prob_xgb, metric="f1")
    print(f"XGBoost: Optimal Val Threshold={opt_thresh_xgb:.4f} (Val F1={val_f1_xgb:.4f})")

    joblib.dump(xgb_clf, models_dir / "xgboost_model.joblib")
    xgb_clf.save_model(str(models_dir / "xgboost_model.json"))

    # -------------------------------------------------------------
    # 4. Persistence Baseline Predictions
    # -------------------------------------------------------------
    test_persistence = df_test["prior_strict_ts_6h"].values
    val_persistence = df_val["prior_strict_ts_6h"].values

    # -------------------------------------------------------------
    # 5. Compile Test Set Predictions & Export CSV
    # -------------------------------------------------------------
    df_preds = pd.DataFrame({
        "forecast_init_time_utc": df_test["forecast_init_time_utc"].values,
        "station_id": df_test["station_id"].values,
        "month": df_test["month"].values,
        "hour_utc": df_test["hour_utc"].values,
        "actual_target": y_test,
        "pred_persistence": test_persistence,
        "prob_logistic_regression": test_prob_lr,
        "pred_bin_logistic_regression_050": (test_prob_lr >= 0.50).astype(int),
        "pred_bin_logistic_regression_opt": (test_prob_lr >= opt_thresh_lr).astype(int),
        "prob_lightgbm": test_prob_lgb,
        "pred_bin_lightgbm_050": (test_prob_lgb >= 0.50).astype(int),
        "pred_bin_lightgbm_opt": (test_prob_lgb >= opt_thresh_lgb).astype(int),
        "prob_xgboost": test_prob_xgb,
        "pred_bin_xgboost_050": (test_prob_xgb >= 0.50).astype(int),
        "pred_bin_xgboost_opt": (test_prob_xgb >= opt_thresh_xgb).astype(int)
    })
    preds_csv = eval_dir / "test_predictions.csv"
    df_preds.to_csv(preds_csv, index=False)
    print(f"\nSaved test predictions to {preds_csv} ({len(df_preds):,} rows)")

    # -------------------------------------------------------------
    # 6. Comprehensive Test Set Metrics Evaluation
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION RESULTS (OCT 1 - DEC 31, 2024; N = 28,073)")
    print("=" * 80)

    # Persistence metrics
    tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_test, test_persistence).ravel()
    prec_p = precision_score(y_test, test_persistence)
    rec_p = recall_score(y_test, test_persistence)
    f1_p = f1_score(y_test, test_persistence)
    pr_p = average_precision_score(y_test, test_persistence)
    roc_p = roc_auc_score(y_test, test_persistence)
    brier_p = brier_score_loss(y_test, test_persistence)

    eval_summary = [
        {
            "model": "Prior 6h Persistence",
            "threshold_rule": "Prior cycle TS == 1",
            "tp": int(tp_p), "fp": int(fp_p), "tn": int(tn_p), "fn": int(fn_p),
            "precision": round(prec_p, 4), "recall": round(rec_p, 4), "f1": round(f1_p, 4),
            "pr_auc": round(pr_p, 4), "roc_auc": round(roc_p, 4), "brier_score": round(brier_p, 4)
        }
    ]

    # Evaluate ML models at 0.50 and validation-optimal threshold
    model_configs = [
        ("Logistic Regression (tau=0.50)", test_prob_lr, 0.50),
        (f"Logistic Regression (tau={opt_thresh_lr:.2f})", test_prob_lr, opt_thresh_lr),
        ("LightGBM (tau=0.50)", test_prob_lgb, 0.50),
        (f"LightGBM (tau={opt_thresh_lgb:.2f})", test_prob_lgb, opt_thresh_lgb),
        ("XGBoost (tau=0.50)", test_prob_xgb, 0.50),
        (f"XGBoost (tau={opt_thresh_xgb:.2f})", test_prob_xgb, opt_thresh_xgb),
    ]

    for name, probs, thresh in model_configs:
        m = compute_metrics(y_test, probs, threshold=thresh)
        eval_summary.append({
            "model": name,
            "threshold_rule": f"Prob >= {thresh:.2f}",
            "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
            "precision": round(m["precision"], 4), "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4), "pr_auc": round(m["pr_auc"], 4),
            "roc_auc": round(m["roc_auc"], 4), "brier_score": round(m["brier_score"], 4)
        })

    df_eval_summary = pd.DataFrame(eval_summary)
    print(df_eval_summary.to_string(index=False))

    # -------------------------------------------------------------
    # 7. Monthly Performance Breakdown
    # -------------------------------------------------------------
    print("\n--- Computing Monthly Performance Breakdown ---")
    monthly_records = []
    months = [10, 11, 12]
    month_names = {10: "October", 11: "November", 12: "December"}

    for m in months:
        mask_m = df_preds["month"] == m
        sub = df_preds[mask_m]
        y_m = sub["actual_target"].values
        n_m = len(y_m)
        n_pos_m = int(y_m.sum())
        prev_m = n_pos_m / n_m if n_m > 0 else 0

        # Persistence
        p_m = sub["pred_persistence"].values
        tn, fp, fn, tp = confusion_matrix(y_m, p_m).ravel()
        monthly_records.append({
            "month": m, "month_name": month_names[m], "model": "Persistence",
            "samples": n_m, "positives": n_pos_m, "prevalence": round(prev_m, 4),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "precision": round(precision_score(y_m, p_m, zero_division=0), 4),
            "recall": round(recall_score(y_m, p_m, zero_division=0), 4),
            "f1": round(f1_score(y_m, p_m, zero_division=0), 4),
            "pr_auc": round(average_precision_score(y_m, p_m), 4),
            "roc_auc": round(roc_auc_score(y_m, p_m), 4),
            "brier_score": round(brier_score_loss(y_m, p_m), 4)
        })

        # ML Models (using val-optimal thresholds)
        for m_name, prob_col, thresh in [
            ("Logistic Regression", "prob_logistic_regression", opt_thresh_lr),
            ("LightGBM", "prob_lightgbm", opt_thresh_lgb),
            ("XGBoost", "prob_xgboost", opt_thresh_xgb)
        ]:
            probs_m = sub[prob_col].values
            preds_m = (probs_m >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_m, preds_m).ravel()
            monthly_records.append({
                "month": m, "month_name": month_names[m], "model": m_name,
                "samples": n_m, "positives": n_pos_m, "prevalence": round(prev_m, 4),
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
                "precision": round(precision_score(y_m, preds_m, zero_division=0), 4),
                "recall": round(recall_score(y_m, preds_m, zero_division=0), 4),
                "f1": round(f1_score(y_m, preds_m, zero_division=0), 4),
                "pr_auc": round(average_precision_score(y_m, probs_m), 4),
                "roc_auc": round(roc_auc_score(y_m, probs_m), 4),
                "brier_score": round(brier_score_loss(y_m, probs_m), 4)
            })

    df_monthly = pd.DataFrame(monthly_records)
    monthly_csv = eval_dir / "metrics_by_month.csv"
    df_monthly.to_csv(monthly_csv, index=False)
    print(f"Saved monthly metrics to {monthly_csv}")

    # -------------------------------------------------------------
    # 8. Station-Level Performance Breakdown
    # -------------------------------------------------------------
    print("\n--- Computing Station-Level Performance Breakdown ---")
    st_records = []
    unique_stations = df_preds["station_id"].unique()

    for st in sorted(unique_stations):
        sub_st = df_preds[df_preds["station_id"] == st]
        y_st = sub_st["actual_target"].values
        n_st = len(y_st)
        n_pos_st = int(y_st.sum())
        prev_st = n_pos_st / n_st if n_st > 0 else 0

        # Only evaluate detailed metrics if station has at least 1 positive in test set
        if n_pos_st >= 5:
            # Persistence
            p_st = sub_st["pred_persistence"].values
            f1_p_st = f1_score(y_st, p_st, zero_division=0)
            brier_p_st = brier_score_loss(y_st, p_st)

            # LightGBM
            prob_lgb_st = sub_st["prob_lightgbm"].values
            pred_lgb_st = (prob_lgb_st >= opt_thresh_lgb).astype(int)
            f1_lgb_st = f1_score(y_st, pred_lgb_st, zero_division=0)
            roc_lgb_st = roc_auc_score(y_st, prob_lgb_st) if len(np.unique(y_st)) > 1 else np.nan
            pr_lgb_st = average_precision_score(y_st, prob_lgb_st)
            brier_lgb_st = brier_score_loss(y_st, prob_lgb_st)

            # XGBoost
            prob_xgb_st = sub_st["prob_xgboost"].values
            pred_xgb_st = (prob_xgb_st >= opt_thresh_xgb).astype(int)
            f1_xgb_st = f1_score(y_st, pred_xgb_st, zero_division=0)
            roc_xgb_st = roc_auc_score(y_st, prob_xgb_st) if len(np.unique(y_st)) > 1 else np.nan
            pr_xgb_st = average_precision_score(y_st, prob_xgb_st)
            brier_xgb_st = brier_score_loss(y_st, prob_xgb_st)

            st_records.append({
                "station_id": st, "samples": n_st, "positives": n_pos_st,
                "prevalence": round(prev_st, 4),
                "f1_persistence": round(f1_p_st, 4), "brier_persistence": round(brier_p_st, 4),
                "f1_lightgbm": round(f1_lgb_st, 4), "roc_lightgbm": round(roc_lgb_st, 4) if not np.isnan(roc_lgb_st) else 0.5,
                "pr_lightgbm": round(pr_lgb_st, 4), "brier_lightgbm": round(brier_lgb_st, 4),
                "f1_xgboost": round(f1_xgb_st, 4), "roc_xgboost": round(roc_xgb_st, 4) if not np.isnan(roc_xgb_st) else 0.5,
                "pr_xgboost": round(pr_xgb_st, 4), "brier_xgboost": round(brier_xgb_st, 4),
            })

    df_station = pd.DataFrame(st_records)
    station_csv = eval_dir / "metrics_by_station.csv"
    df_station.to_csv(station_csv, index=False)
    print(f"Saved station metrics ({len(df_station)} qualifying stations) to {station_csv}")

    # -------------------------------------------------------------
    # 9. Feature Importance Analysis
    # -------------------------------------------------------------
    lgb_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": lgb_clf.booster_.feature_importance(importance_type="gain"),
        "importance_split": lgb_clf.booster_.feature_importance(importance_type="split")
    }).sort_values("importance_gain", ascending=False).reset_index(drop=True)

    xgb_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance_weight": [xgb_clf.get_booster().get_score(importance_type="weight").get(f, 0) for f in feature_cols],
        "importance_gain": [xgb_clf.get_booster().get_score(importance_type="gain").get(f, 0) for f in feature_cols]
    }).sort_values("importance_gain", ascending=False).reset_index(drop=True)

    print("\nTop 10 Features by LightGBM Gain:")
    for idx, r in lgb_imp.head(10).iterrows():
        print(f"  {idx+1:2d}. {r['feature']:30s} Gain={r['importance_gain']:10.1f} Split={int(r['importance_split']):5d}")

    # -------------------------------------------------------------
    # 10. Calibration Assessment
    # -------------------------------------------------------------
    cal_lgb = calibration_curve(y_test, test_prob_lgb, n_bins=10)
    cal_xgb = calibration_curve(y_test, test_prob_xgb, n_bins=10)
    print("\nCalibration Bins (LightGBM):")
    for f_p, m_p in zip(cal_lgb[0], cal_lgb[1]):
        print(f"  Predicted Prob: {m_p:.4f} -> Empirical Frequency: {f_p:.4f}")

    # -------------------------------------------------------------
    # 11. Save Training Configuration Record
    # -------------------------------------------------------------
    config_record = {
        "training_timestamp_utc": "2026-10-27T22:45:00Z",
        "random_seed": 42,
        "splits": {
            "train": {"period": "2024-01-01 to 2024-07-31", "samples": len(df_train), "positives": int(y_train.sum()), "prevalence": float(y_train.mean())},
            "validation": {"period": "2024-08-01 to 2024-09-30", "samples": len(df_val), "positives": int(y_val.sum()), "prevalence": float(y_val.mean())},
            "test": {"period": "2024-10-01 to 2024-12-31", "samples": len(df_test), "positives": int(y_test.sum()), "prevalence": float(y_test.mean())}
        },
        "features": {
            "count": len(feature_cols),
            "feature_list": feature_cols,
            "preprocessing": "StandardScaler for Logistic Regression fit strictly on train; tree models fit on unscaled continuous + lag features"
        },
        "model_parameters": {
            "logistic_regression": {"C": 0.1, "max_iter": 1000, "solver": "lbfgs", "random_state": 42, "optimal_val_threshold": opt_thresh_lr},
            "lightgbm": {**lgb_params, "best_iteration": int(lgb_clf.best_iteration_), "optimal_val_threshold": opt_thresh_lgb},
            "xgboost": {**xgb_params, "best_iteration": int(xgb_clf.best_iteration), "optimal_val_threshold": opt_thresh_xgb}
        },
        "feature_importances_top10": lgb_imp.head(10).to_dict(orient="records"),
        "test_benchmark_results": eval_summary
    }

    config_json = models_dir / "model_training_config.json"
    with open(config_json, "w") as f:
        json.dump(config_record, f, indent=2)
    print(f"\nSaved training configuration to {config_json}")

    print("\nALL PHASE 3 TRAINING AND EVALUATION TASKS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
