import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, brier_score_loss
)

df = pd.read_parquet("data/processed/asos_repair/florida_asos_modeling_ready_repaired_2024.parquet")
df_eval = df[df["target_asos_strict_crit_c"].notna()].copy()
df_eval["target"] = df_eval["target_asos_strict_crit_c"].astype(int)

# Sort and compute persistence
df_eval = df_eval.sort_values(["station_id", "forecast_init_time_utc"]).reset_index(drop=True)
df_eval["prev_init"] = df_eval.groupby("station_id")["forecast_init_time_utc"].shift(1)
df_eval["prev_target"] = df_eval.groupby("station_id")["target"].shift(1)
df_eval["cycle_diff_hours"] = (df_eval["forecast_init_time_utc"] - df_eval["prev_init"]).dt.total_seconds() / 3600.0
df_eval["pred_persistence"] = np.where(df_eval["cycle_diff_hours"] == 6.0, df_eval["prev_target"], 0).astype(int)

# Evaluate on full dataset (current state) vs excluding DTS and TTS
for label, df_sub in [("ALL 87 STATIONS (WITH DTS/TTS ARTIFACT)", df_eval),
                      ("85 STATIONS (EXCLUDING CORRUPTED DTS/TTS)", df_eval[~df_eval["station_id"].isin(["DTS", "TTS"])])]:
    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)
    train_mask = df_sub["forecast_init_time_utc"] < "2024-08-01"
    val_mask = (df_sub["forecast_init_time_utc"] >= "2024-08-01") & (df_sub["forecast_init_time_utc"] < "2024-10-01")
    test_mask = df_sub["forecast_init_time_utc"] >= "2024-10-01"

    y_train = df_sub.loc[train_mask, "target"].values
    y_val = df_sub.loc[val_mask, "target"].values
    y_test = df_sub.loc[test_mask, "target"].values

    print(f"Train: N={len(y_train):,}, Pos={y_train.sum():,} ({y_train.mean()*100:.2f}%)")
    print(f"Val:   N={len(y_val):,}, Pos={y_val.sum():,} ({y_val.mean()*100:.2f}%)")
    print(f"Test:  N={len(y_test):,}, Pos={y_test.sum():,} ({y_test.mean()*100:.2f}%)")

    # Persistence
    y_pred_p = df_sub.loc[test_mask, "pred_persistence"].values
    tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_test, y_pred_p).ravel()
    prec_p = precision_score(y_test, y_pred_p, zero_division=0)
    rec_p = recall_score(y_test, y_pred_p, zero_division=0)
    f1_p = f1_score(y_test, y_pred_p, zero_division=0)
    brier_p = brier_score_loss(y_test, y_pred_p)
    roc_p = 0.5 * (rec_p + (tn_p / (tn_p + fp_p)))
    pr_p = prec_p * rec_p
    print(f"Persistence: TP={tp_p}, FP={fp_p}, TN={tn_p}, FN={fn_p}")
    print(f"  Prec={prec_p:.4f}, Rec={rec_p:.4f}, F1={f1_p:.4f}, PR-AUC={pr_p:.4f}, ROC-AUC={roc_p:.4f}, Brier={brier_p:.4f}")

    # CAPE >= 1000
    cape_test = df_sub.loc[test_mask, "cape_J_kg"].fillna(0).values
    y_pred_cape = (cape_test >= 1000.0).astype(int)
    tn_c, fp_c, fn_c, tp_c = confusion_matrix(y_test, y_pred_cape).ravel()
    prec_c = precision_score(y_test, y_pred_cape, zero_division=0)
    rec_c = recall_score(y_test, y_pred_cape, zero_division=0)
    f1_c = f1_score(y_test, y_pred_cape, zero_division=0)
    pr_c = average_precision_score(y_test, cape_test)
    roc_c = roc_auc_score(y_test, cape_test)
    brier_c = brier_score_loss(y_test, y_pred_cape)
    print(f"CAPE >= 1000: TP={tp_c}, FP={fp_c}, TN={tn_c}, FN={fn_c}")
    print(f"  Prec={prec_c:.4f}, Rec={rec_c:.4f}, F1={f1_c:.4f}, PR-AUC={pr_c:.4f}, ROC-AUC={roc_c:.4f}, Brier={brier_c:.4f}")

    # Logistic Regression
    lr_features = [
        "cape_J_kg", "cin_J_kg", "dewpoint_depression_degC",
        "temperature_2m_degC", "bulk_shear_850_300_ms",
        "total_precipitation_mm", "tcwv_kg_m2"
    ]
    X_train = df_sub.loc[train_mask, lr_features].fillna(0).values
    X_test = df_sub.loc[test_mask, lr_features].fillna(0).values
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    y_pred_lr = (y_prob_lr >= 0.5).astype(int)
    tn_lr, fp_lr, fn_lr, tp_lr = confusion_matrix(y_test, y_pred_lr).ravel()
    prec_lr = precision_score(y_test, y_pred_lr, zero_division=0)
    rec_lr = recall_score(y_test, y_pred_lr, zero_division=0)
    f1_lr = f1_score(y_test, y_pred_lr, zero_division=0)
    pr_lr = average_precision_score(y_test, y_prob_lr)
    roc_lr = roc_auc_score(y_test, y_prob_lr)
    brier_lr = brier_score_loss(y_test, y_prob_lr)
    print(f"Logistic Regression: TP={tp_lr}, FP={fp_lr}, TN={tn_lr}, FN={fn_lr}")
    print(f"  Prec={prec_lr:.4f}, Rec={rec_lr:.4f}, F1={f1_lr:.4f}, PR-AUC={pr_lr:.4f}, ROC-AUC={roc_lr:.4f}, Brier={brier_lr:.4f}")
