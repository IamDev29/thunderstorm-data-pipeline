"""
Backup script for Phase 1 of Persistence and Event Dependence Audit.
Computes SHA-256 hashes and creates backups of master, forecast datasets,
feature dictionaries, reports, and scripts.
"""

import hashlib
from pathlib import Path
import shutil

FILES_TO_BACKUP = [
    Path("data/processed/florida_thunderstorm_2024_master.csv"),
    Path("data/processed/florida_thunderstorm_2024_forecasting_6h.csv"),
    Path("data/processed/florida_thunderstorm_2024_label_audit.csv"),
    Path("reports/thunderstorm_label_quality_audit.md"),
    Path("data/processed/feature_dictionary.json"),
    Path("reports/full_year_3d_model_readiness.md"),
    Path("reports/six_hour_ahead_forecast_evaluation.md"),
    Path("scripts/build_experiments_and_forecast_datasets.py"),
    Path("scripts/build_full_year_3d_master.py"),
]

BACKUP_DIR = Path("data/backups/phase_audit_backup")
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    manifest_lines = []
    print("=== PHASE 1: BACKUP AND SHA-256 HASH RECORDING ===")
    
    for f in FILES_TO_BACKUP:
        if not f.exists():
            # Check child project directory
            alt_f = Path("Florida_Thunderstorm_2024_Training") / f
            if alt_f.exists():
                f = alt_f
            else:
                print(f"[WARNING] File not found: {f}")
                continue
                
        h = sha256_file(f)
        dst = BACKUP_DIR / f.name
        print(f"Backing up {f} -> {dst}")
        shutil.copy2(f, dst)
        
        size_mb = f.stat().st_size / (1024 * 1024)
        line = f"{h}  {size_mb:.2f} MB  {f.as_posix()}"
        manifest_lines.append(line)
        print(f"  SHA-256: {h} ({size_mb:.2f} MB)")
        
    manifest_path = BACKUP_DIR / "manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"\nManifest written to {manifest_path}")

if __name__ == "__main__":
    main()
