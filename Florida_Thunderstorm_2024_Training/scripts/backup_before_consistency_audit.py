"""
Phase 1: Backup before consistency and reproducibility audit.
Creates versioned backup of all processed datasets, audit_v2 files, and reports.
Computes SHA-256 hashes, file sizes, and timestamps.
"""
import shutil
import hashlib
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(r"c:\Users\KIIT0001\Desktop\Data Collection")
PROJECT_DIR = BASE_DIR / "Florida_Thunderstorm_2024_Training"
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP_DIR = BASE_DIR / "data" / "backups" / f"consistency_audit_backup_{TS}"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

files_to_backup = [
    BASE_DIR / "data" / "processed" / "florida_thunderstorm_2024_master.csv",
    PROJECT_DIR / "data" / "processed" / "florida_thunderstorm_2024_master.csv",
    BASE_DIR / "data" / "processed" / "florida_thunderstorm_2024_forecasting_6h.csv",
    PROJECT_DIR / "data" / "processed" / "noaa_florida_2024.csv",
    PROJECT_DIR / "data" / "processed" / "glm_florida_2024.csv",
]
files_to_backup.extend(list((BASE_DIR / "data" / "processed" / "audit_v2").glob("*.*")))
files_to_backup.extend(list((BASE_DIR / "reports").glob("*.md")))
files_to_backup.extend(list((BASE_DIR / "reports").glob("*.json")))

manifest = []
manifest.append(f"Backup Timestamp: {datetime.now().isoformat()}")
manifest.append(f"Backup Directory: {BACKUP_DIR}")
manifest.append("=" * 140)
header = "{:<75} | {:>14} | {:<64}".format("File", "Size (Bytes)", "SHA-256")
manifest.append(header)
manifest.append("-" * 160)

count = 0
for p in sorted(set(files_to_backup)):
    if p.exists() and p.is_file():
        try:
            rel = p.relative_to(BASE_DIR)
        except ValueError:
            rel = p
        dest = BACKUP_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        h = sha256(p)
        entry = "{:<75} | {:>14,} | {:<64}".format(str(rel), p.stat().st_size, h)
        manifest.append(entry)
        count += 1
        print(f"Backed up: {rel}")

manifest_path = BACKUP_DIR / "manifest.txt"
manifest_path.write_text("\n".join(manifest), encoding="utf-8")
print(f"\nManifest saved to: {manifest_path}")
print(f"Total files backed up: {count}")
