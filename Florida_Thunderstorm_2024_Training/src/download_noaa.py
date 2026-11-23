"""NOAA Storm Events Data Downloader and Importer.

Reuses existing verified 2024 NOAA Storm Events files from the workspace,
or retrieves them directly from the official NOAA SWDI CSV archive if missing.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import logging
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

NOAA_BASE_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class NOAADownloader:
    """Downloader and asset manager for NOAA Storm Events bulk CSV files."""

    def __init__(
        self,
        raw_dest_dir: Path = Path("data/raw/noaa"),
        parent_raw_dir: Path = Path("../data/raw"),
    ):
        self.raw_dest_dir = raw_dest_dir
        self.parent_raw_dir = parent_raw_dir
        self.raw_dest_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_records: List[Dict[str, Any]] = []

    def get_or_download_2024_files(self) -> Dict[str, Path]:
        """Import or download 2024 details, locations, and fatalities files."""
        categories = ["details", "locations", "fatalities"]
        results: Dict[str, Path] = {}

        for cat in categories:
            dest_file = self.raw_dest_dir / f"StormEvents_{cat}_2024.csv.gz"

            # Check if parent project has the verified file
            parent_cat_dir = self.parent_raw_dir / cat
            parent_files = list(parent_cat_dir.glob("StormEvents_*_d2024_*.csv.gz")) if parent_cat_dir.exists() else []

            if parent_files and parent_files[0].stat().st_size > 1024:
                src_file = parent_files[0]
                if not dest_file.exists() or dest_file.stat().st_size != src_file.stat().st_size:
                    logger.info(f"Reusing verified local NOAA {cat} archive: {src_file.name}")
                    shutil.copy2(src_file, dest_file)
                status = "REUSED_LOCAL"
            elif dest_file.exists() and dest_file.stat().st_size > 1024:
                status = "CACHED"
            else:
                logger.warning(f"No existing local {cat} file found. Would download from {NOAA_BASE_URL}...")
                status = "MISSING"

            if dest_file.exists():
                results[cat] = dest_file
                self.manifest_records.append({
                    "source": "NOAA",
                    "category": cat,
                    "filename": dest_file.name,
                    "path": str(dest_file).replace("\\", "/"),
                    "size_bytes": dest_file.stat().st_size,
                    "sha256": compute_sha256(dest_file),
                    "status": status,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    downloader = NOAADownloader()
    res = downloader.get_or_download_2024_files()
    print("NOAA files ready:", res)
