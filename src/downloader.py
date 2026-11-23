"""NOAA Storm Events Data Downloader.

Automated, resilient downloading of official NOAA bulk CSV archive files.
Supports gzip verification, retry logic with exponential backoff, duplicate detection,
and manifest generation.
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from src.archive_parser import NOAAFileInfo

logger = logging.getLogger(__name__)


@dataclass
class DownloadRecord:
    """Record of an individual file download attempt."""

    filename: str
    dataset_type: str
    year: int
    source_url: str
    local_path: str
    file_size_bytes: int
    status: str  # 'DOWNLOADED', 'CACHED', 'FAILED'
    timestamp: str
    error_message: Optional[str] = None


class NOAADownloader:
    """Downloader for NOAA Storm Events bulk CSV files."""

    def __init__(
        self,
        output_dirs: Dict[str, Path],
        reports_dir: Path,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        chunk_size: int = 524288,
    ):
        self.output_dirs = {k: Path(v) for k, v in output_dirs.items()}
        self.reports_dir = Path(reports_dir)
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.chunk_size = chunk_size

        # Ensure directories exist
        for d in self.output_dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_records: List[DownloadRecord] = []

    def get_destination_path(self, file_info: NOAAFileInfo) -> Path:
        """Get the local target Path for a given NOAA file."""
        target_dir = self.output_dirs.get(file_info.dataset_type)
        if not target_dir:
            # Fallback to generic raw dir
            target_dir = self.output_dirs.get("raw", Path("data/raw"))
        return target_dir / file_info.filename

    def is_valid_local_file(self, file_path: Path, is_compressed: bool) -> bool:
        """Verify that a local file exists, is non-empty, and has valid gzip formatting if applicable."""
        if not file_path.exists():
            return False

        try:
            size = file_path.stat().st_size
            if size == 0:
                return False

            if is_compressed:
                # Validate gzip header and test readability of the first block
                with gzip.open(file_path, "rb") as gz:
                    chunk = gz.read(1024)
                    if not chunk:
                        return False
            return True
        except Exception as exc:
            logger.warning(f"Existing file {file_path.name} failed integrity check: {exc}")
            return False

    def download_file(self, file_info: NOAAFileInfo) -> DownloadRecord:
        """Download a single NOAA file with retries and verification.

        If a valid local copy exists, the download is skipped and recorded as CACHED.
        """
        dest_path = self.get_destination_path(file_info)
        now_iso = datetime.now(timezone.utc).isoformat()

        # Check for existing valid copy
        if self.is_valid_local_file(dest_path, file_info.is_compressed):
            logger.info(f"Skipping download (already cached): {file_info.filename}")
            record = DownloadRecord(
                filename=file_info.filename,
                dataset_type=file_info.dataset_type,
                year=file_info.year,
                source_url=file_info.url,
                local_path=str(dest_path),
                file_size_bytes=dest_path.stat().st_size,
                status="CACHED",
                timestamp=now_iso,
                error_message=None,
            )
            self.manifest_records.append(record)
            return record

        temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
        last_error: Optional[str] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                # Support resuming partial downloads via HTTP Range header
                existing_bytes = temp_path.stat().st_size if temp_path.exists() else 0
                headers = {"User-Agent": "NOAA-StormEvents-Pipeline/1.0"}
                mode = "wb"

                if existing_bytes > 0:
                    headers["Range"] = f"bytes={existing_bytes}-"
                    mode = "ab"
                    logger.info(
                        f"Resuming download of {file_info.filename} from byte {existing_bytes:,} (attempt {attempt}/{self.max_retries})"
                    )
                else:
                    logger.info(
                        f"Downloading {file_info.filename} (attempt {attempt}/{self.max_retries}) from {file_info.url}"
                    )

                with requests.get(
                    file_info.url,
                    stream=True,
                    timeout=self.timeout,
                    headers=headers,
                ) as resp:
                    if resp.status_code == 416:
                        # Range Not Satisfiable: restart download from scratch
                        logger.warning(f"Range not satisfiable for {file_info.filename}, restarting download.")
                        temp_path.unlink(missing_ok=True)
                        existing_bytes = 0
                        mode = "wb"
                        with requests.get(
                            file_info.url,
                            stream=True,
                            timeout=self.timeout,
                            headers={"User-Agent": "NOAA-StormEvents-Pipeline/1.0"},
                        ) as fresh_resp:
                            if fresh_resp.status_code != 200:
                                raise requests.HTTPError(f"HTTP {fresh_resp.status_code}: {fresh_resp.reason}")
                            with open(temp_path, "wb") as f_out:
                                for chunk in fresh_resp.iter_content(chunk_size=self.chunk_size):
                                    if chunk:
                                        f_out.write(chunk)
                    elif resp.status_code not in (200, 206):
                        raise requests.HTTPError(
                            f"HTTP {resp.status_code}: {resp.reason}"
                        )
                    else:
                        if resp.status_code == 200 and existing_bytes > 0:
                            # Server ignored range header, write from beginning
                            mode = "wb"

                        with open(temp_path, mode) as f_out:
                            for chunk in resp.iter_content(chunk_size=self.chunk_size):
                                if chunk:
                                    f_out.write(chunk)

                # Verify integrity of downloaded temp file
                if not self.is_valid_local_file(temp_path, file_info.is_compressed):
                    raise ValueError(f"Downloaded file {file_info.filename} failed integrity validation.")

                # Atomic rename
                if dest_path.exists():
                    dest_path.unlink()
                temp_path.rename(dest_path)

                size = dest_path.stat().st_size
                logger.info(f"Successfully downloaded {file_info.filename} ({size:,} bytes).")
                record = DownloadRecord(
                    filename=file_info.filename,
                    dataset_type=file_info.dataset_type,
                    year=file_info.year,
                    source_url=file_info.url,
                    local_path=str(dest_path),
                    file_size_bytes=size,
                    status="DOWNLOADED",
                    timestamp=now_iso,
                    error_message=None,
                )
                self.manifest_records.append(record)
                return record

            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    f"Attempt {attempt}/{self.max_retries} failed for {file_info.filename}: {exc}"
                )
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass

                if attempt < self.max_retries:
                    sleep_time = self.backoff_factor ** attempt
                    logger.info(f"Sleeping {sleep_time:.1f}s before retry...")
                    time.sleep(sleep_time)

        # All retries exhausted
        logger.error(f"Failed to download {file_info.filename} after {self.max_retries} attempts. Last error: {last_error}")
        record = DownloadRecord(
            filename=file_info.filename,
            dataset_type=file_info.dataset_type,
            year=file_info.year,
            source_url=file_info.url,
            local_path=str(dest_path),
            file_size_bytes=0,
            status="FAILED",
            timestamp=now_iso,
            error_message=last_error,
        )
        self.manifest_records.append(record)
        return record

    def download_batch(self, files: List[NOAAFileInfo]) -> List[DownloadRecord]:
        """Download a batch of NOAA archive files sequentially."""
        records: List[DownloadRecord] = []
        for file_info in files:
            rec = self.download_file(file_info)
            records.append(rec)
        self.save_manifest()
        return records

    def save_manifest(self) -> Tuple[Path, Path]:
        """Save the download manifest to CSV and JSON formats."""
        csv_path = self.reports_dir / "download_manifest.csv"
        json_path = self.reports_dir / "download_manifest.json"

        # Write CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
            fieldnames = [
                "filename",
                "dataset_type",
                "year",
                "source_url",
                "local_path",
                "file_size_bytes",
                "status",
                "timestamp",
                "error_message",
            ]
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.manifest_records:
                writer.writerow(asdict(r))

        # Write JSON
        with open(json_path, "w", encoding="utf-8") as f_json:
            json.dump([asdict(r) for r in self.manifest_records], f_json, indent=2)

        logger.info(f"Download manifest saved to {csv_path} and {json_path}")
        return csv_path, json_path
