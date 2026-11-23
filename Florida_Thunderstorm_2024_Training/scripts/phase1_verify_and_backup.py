"""
Phase 1: Verification and Backup before MLP Benchmark & Episode Split Correction.
Records SHA-256 hashes, feature lists, target definitions, and creates versioned backup.
"""
import shutil
import hashlib
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

BASE_DIR = Path(r"c:\Users\KIIT0001\Desktop\Data Collection")
PROJECT_DIR = BASE_DIR / "Florida_Thunderstorm_2024_Training"
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP_DIR = BASE_DIR / "data" / "backups" / f"mlp_benchmark_backup_{TS}"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

fc_csv = BASE_DIR / "data" / "processed" / "florida_thunderstorm_2024_forecasting_6h.csv"
splits_csv = BASE_DIR / "data" / "processed" / "audit_v2" / "split_assignments.csv"
metrics_csv = BASE_DIR / "data" / "processed" / "audit_v2" / "reconciled_experiment_metrics.csv"

files_to_backup = [fc_csv, splits_csv, metrics_csv]
files_to_backup.extend(list((BASE_DIR / "data" / "processed" / "audit_v2").glob("*.csv")))
files_to_backup.extend(list((BASE_DIR / "data" / "processed" / "audit_v2").glob("*.json")))

print("=== BACKING UP CRITICAL FILES ===")
manifest = []
manifest.append(f"Backup Timestamp: {datetime.now().isoformat()}")
manifest.append(f"Backup Directory: {BACKUP_DIR}")
manifest.append("=" * 140)
header = "{:<75} | {:>14} | {:<64}".format("File", "Size (Bytes)", "SHA-256")
manifest.append(header)
manifest.append("-" * 160)

for p in sorted(set(files_to_backup)):
    if p.exists() and p.is_file():
        rel = p.relative_to(BASE_DIR)
        dest = BACKUP_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        h = sha256(p)
        entry = "{:<75} | {:>14,} | {:<64}".format(str(rel), p.stat().st_size, h)
        manifest.append(entry)
        print(f"Backed up: {rel}")

(BACKUP_DIR / "manifest.txt").write_text("\n".join(manifest), encoding="utf-8")
print(f"Saved backup manifest to {BACKUP_DIR / 'manifest.txt'}")

print("\n=== VERIFYING AUTHORITATIVE FORECAST DATASET & TARGET ===")
fc_hash = sha256(fc_csv)
df = pd.read_csv(fc_csv, nrows=10)
print(f"Dataset path: {fc_csv}")
print(f"SHA-256: {fc_hash}")
print(f"File size: {fc_csv.stat().st_size:,} bytes")

# Target check
print(f"Target column 'future_noaa_event_6h' present: {'future_noaa_event_6h' in df.columns}")
print(f"Historical prior 'true_prior_event_6h' present: {'true_prior_event_6h' in df.columns}")
print(f"Leaked feature 'prior_event_at_init' present: {'prior_event_at_init' in df.columns}")

# Feature list verification
exclude_cols = [
    "forecast_init_time_utc", "latitude", "longitude", "grid_cell_id", "month", "hour_utc",
    "future_noaa_event_count", "future_noaa_event_ids", "future_noaa_event_types",
    "future_noaa_min_distance_km", "future_noaa_event_6h", "forecast_window_start_utc",
    "forecast_window_end_utc", "forecast_window_status", "prior_event_at_init",
    "true_prior_event_count", "true_prior_event_ids", "true_prior_episode_ids",
    "future_noaa_episode_ids", "glm_coverage_status", "pressure_levels_complete",
    "features_complete", "audit_category", "number", "expver", "init_dt", "date", "doy"
]

feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ["float64", "int64"]]
print(f"\nTotal verified feature columns ({len(feature_cols)}):")
for i, f in enumerate(feature_cols):
    print(f"  {i+1:2d}. {f}")

assert "prior_event_at_init" not in feature_cols, "ERROR: Leaked prior_event_at_init is in feature list!"
assert "true_prior_event_6h" in feature_cols, "ERROR: true_prior_event_6h missing!"
assert len(feature_cols) == 48, f"Expected exactly 48 features, got {len(feature_cols)}"

info = {
    "dataset_path": str(fc_csv),
    "dataset_sha256": fc_hash,
    "feature_count": len(feature_cols),
    "feature_cols": feature_cols,
    "target_col": "future_noaa_event_6h",
    "target_window": "(t, t+6h]",
    "historical_convection_col": "true_prior_event_6h",
    "historical_window": "(t-6h, t]",
    "excluded_leaked_feature": "prior_event_at_init"
}

out_meta = BASE_DIR / "data" / "processed" / "mlp_benchmark" / "feature_metadata.json"
out_meta.parent.mkdir(parents=True, exist_ok=True)
with open(out_meta, "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2)
print(f"\nSaved feature metadata to {out_meta}")
