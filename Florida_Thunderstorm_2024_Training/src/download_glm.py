"""NOAA GOES-16 Geostationary Lightning Mapper (GLM) Downloader.

Retrieves official Level-2 Lightning Cluster Filter Algorithm (LCFA)
NetCDF granules from NOAA's public AWS Open Data bucket:
s3://noaa-goes16/GLM-L2-LCFA/

Directory convention:
GLM-L2-LCFA/<YYYY>/<JJJ>/<HH>/OR_GLM-L2-LCFA_G16_s<start>_e<end>_c<create>.nc

Where:
YYYY = Year (e.g. 2024)
JJJ  = Day of year (001 to 366 for 2024 leap year)
HH   = UTC hour (00 to 23)
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

AWS_S3_BASE_URL = "https://noaa-goes16.s3.amazonaws.com"
S3_NAMESPACE = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class GLMDownloader:
    """Downloads NOAA GOES-16 GLM L2 LCFA granules."""

    def __init__(self, raw_dir: Path = Path("data/raw/glm")):
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_records: List[Dict[str, Any]] = []

    def list_granules_for_hour(self, year: int, day_of_year: int, hour: int, max_keys: int = 180) -> List[Dict[str, Any]]:
        """Query public AWS S3 XML endpoint for available 20-second granules."""
        prefix = f"GLM-L2-LCFA/{year}/{day_of_year:03d}/{hour:02d}/"
        url = f"{AWS_S3_BASE_URL}/?prefix={prefix}&max-keys={max_keys}"
        req = urllib.request.Request(url, headers={"User-Agent": "FloridaThunderstormPipeline/1.0"})

        granules = []
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read()
                root = ET.fromstring(content)
                for elem in root.findall("s3:Contents", S3_NAMESPACE):
                    key = elem.find("s3:Key", S3_NAMESPACE).text
                    size = int(elem.find("s3:Size", S3_NAMESPACE).text)
                    granules.append({"key": key, "size_bytes": size, "filename": key.split("/")[-1]})
        except Exception as exc:
            logger.error(f"Failed to query GLM granules at prefix {prefix}: {exc}")

        return granules

    def download_granule(self, key: str, filename: str) -> Optional[Path]:
        """Download a single GLM NetCDF granule from AWS Open Data."""
        target_path = self.raw_dir / filename
        if target_path.exists() and target_path.stat().st_size > 0:
            logger.debug(f"Granule {filename} already exists, skipping download.")
            return target_path

        url = f"{AWS_S3_BASE_URL}/{key}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "FloridaThunderstormPipeline/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(target_path, "wb") as f_out:
                f_out.write(resp.read())

            self.manifest_records.append({
                "source": "NOAA_GOES16_GLM",
                "category": "lightning",
                "filename": filename,
                "path": str(target_path).replace("\\", "/"),
                "size_bytes": target_path.stat().st_size,
                "sha256": compute_sha256(target_path),
                "status": "DOWNLOADED",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "notes": f"GOES-16 GLM-L2-LCFA granule: {key}",
            })
            return target_path
        except Exception as exc:
            logger.error(f"Failed downloading GLM granule {key}: {exc}")
            if target_path.exists():
                target_path.unlink()
            return None

    def retrieve_sample_granules(
        self,
        year: int = 2024,
        day_of_year: int = 136,  # May 15
        hours: List[int] = [0, 6, 12, 18],
        granules_per_hour: int = 3,
    ) -> List[Path]:
        """Retrieve representative granules for given hours to validate pipeline."""
        downloaded = []
        for h in hours:
            logger.info(f"Checking GLM granules for day {day_of_year}, hour {h:02d}:00 UTC...")
            available = self.list_granules_for_hour(year, day_of_year, h, max_keys=granules_per_hour * 2)
            for item in available[:granules_per_hour]:
                p = self.download_granule(item["key"], item["filename"])
                if p:
                    downloaded.append(p)
        logger.info(f"Total GLM granules retrieved: {len(downloaded)}")
        return downloaded


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dl = GLMDownloader()
    files = dl.retrieve_sample_granules()
    print("Downloaded GLM files:", len(files))
