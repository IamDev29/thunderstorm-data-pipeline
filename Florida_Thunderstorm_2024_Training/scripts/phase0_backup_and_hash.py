"""Phase 0: Backup all existing datasets and compute SHA-256 hashes."""
import hashlib
import shutil
import os
from pathlib import Path
from datetime import datetime

PROJECT = Path(r"c:\Users\KIIT0001\Desktop\Data Collection\Florida_Thunderstorm_2024_Training")
BACKUP_DIR = PROJECT / "data" / "backups" / f"audit_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

files_to_backup = [
    PROJECT / "data" / "processed" / "florida_thunderstorm_2024_master.csv",
    PROJECT / "data" / "processed" / "florida_thunderstorm_2024_forecasting_6h.csv",
    PROJECT / "data" / "processed" / "florida_thunderstorm_2024_training.csv",
    PROJECT / "data" / "processed" / "florida_thunderstorm_2024_training_noaa_centered.csv",
    PROJECT / "data" / "processed" / "florida_thunderstorm_2024_training_diagnostic.csv",
    PROJECT / "data" / "processed" / "florida_thunderstorm_2024_training_surface_full_year.csv",
    PROJECT / "data" / "processed" / "florida_thunderstorm_2024_label_audit.csv",
    PROJECT / "data" / "processed" / "noaa_florida_2024.csv",
    PROJECT / "data" / "processed" / "glm_florida_2024.csv",
    PROJECT / "data" / "processed" / "era5_florida_2024.csv",
]

reports_to_backup = list((PROJECT / "reports").glob("*.md")) + list((PROJECT / "reports").glob("*.json"))

manifest_lines = []
manifest_lines.append(f"Backup created: {datetime.now().isoformat()}")
manifest_lines.append(f"Backup directory: {BACKUP_DIR}")
manifest_lines.append("")
manifest_lines.append("FILE | SIZE_BYTES | SHA256")
manifest_lines.append("-" * 120)

for f in files_to_backup + reports_to_backup:
    if f.exists():
        dest = BACKUP_DIR / f.relative_to(PROJECT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        h = sha256(f)
        manifest_lines.append(f"{f.relative_to(PROJECT)} | {f.stat().st_size} | {h}")
        print(f"  Backed up: {f.name} ({f.stat().st_size:,} bytes)")
    else:
        manifest_lines.append(f"{f.relative_to(PROJECT)} | MISSING | N/A")
        print(f"  MISSING: {f.name}")

manifest_path = BACKUP_DIR / "manifest.txt"
manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")
print(f"\nManifest saved to: {manifest_path}")
print(f"Total files backed up: {sum(1 for l in manifest_lines if '|' in l and 'MISSING' not in l)}")
