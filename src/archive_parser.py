"""NOAA Storm Events Archive Parser.

Inspects the official NOAA bulk CSV archive:
https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/
Identifies available files, extracts metadata (type, year, creation date),
and resolves the most up-to-date files for configured historical years.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)

# Regular expression matching NOAA Storm Events bulk CSV file format
# Examples:
#   StormEvents_details-ftp_v1.0_d2024_c20260728.csv.gz
#   StormEvents_locations-ftp_v1.0_d2023_c20260707.csv.gz
#   StormEvents_fatalities-ftp_v1.0_d2022_c20260625.csv.gz
NOAA_FILENAME_PATTERN = re.compile(
    r"^StormEvents_(?P<type>details|locations|fatalities)-ftp_v1\.0_d(?P<year>\d{4})_c(?P<cdate>\d+)\.csv(?P<gz>\.gz)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NOAAFileInfo:
    """Metadata describing a downloadable NOAA Storm Events archive file."""

    filename: str
    dataset_type: str  # details, locations, fatalities
    year: int
    creation_date: int  # e.g. 20260728
    is_compressed: bool
    url: str


class ArchiveParserError(Exception):
    """Raised when parsing the NOAA bulk archive fails or encounters invalid data."""


class NOAAArchiveParser:
    """Parser and resolver for NOAA's Storm Events bulk CSV archive."""

    def __init__(self, base_url: str = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/", timeout: int = 30):
        self.base_url = base_url if base_url.endswith("/") else base_url + "/"
        self.timeout = timeout

    def fetch_archive_html(self) -> str:
        """Fetch the directory HTML listing from the official NOAA archive."""
        logger.info(f"Connecting to NOAA Storm Events archive: {self.base_url}")
        try:
            resp = requests.get(
                self.base_url,
                timeout=self.timeout,
                headers={"User-Agent": "NOAA-StormEvents-Pipeline/1.0"},
            )
            if resp.status_code != 200:
                raise ArchiveParserError(
                    f"Failed to fetch NOAA archive index. HTTP Status: {resp.status_code}. URL: {self.base_url}"
                )
            return resp.text
        except requests.RequestException as exc:
            raise ArchiveParserError(
                f"Network error connecting to NOAA archive at {self.base_url}: {exc}"
            ) from exc

    def parse_links_from_html(self, html_content: str) -> List[str]:
        """Extract all hyperlink targets from directory index HTML."""
        # Find href values in <a> tags
        href_matches = re.findall(r'href=[\'"]([^\'">\s]+)[\'"]', html_content, re.IGNORECASE)
        # Extract filename if href is relative or full path
        filenames = [h.split("/")[-1] for h in href_matches if h]
        return filenames

    def parse_file_info(self, filename: str) -> Optional[NOAAFileInfo]:
        """Parse filename and return structured NOAAFileInfo, or None if not matching."""
        match = NOAA_FILENAME_PATTERN.match(filename)
        if not match:
            return None

        dataset_type = match.group("type").lower()
        year = int(match.group("year"))
        creation_date = int(match.group("cdate"))
        is_compressed = bool(match.group("gz"))
        url = urljoin(self.base_url, filename)

        return NOAAFileInfo(
            filename=filename,
            dataset_type=dataset_type,
            year=year,
            creation_date=creation_date,
            is_compressed=is_compressed,
            url=url,
        )

    def discover_available_files(self, html_content: Optional[str] = None) -> List[NOAAFileInfo]:
        """Discover and parse all valid StormEvents archive files available in the archive."""
        if html_content is None:
            html_content = self.fetch_archive_html()

        filenames = self.parse_links_from_html(html_content)
        files: List[NOAAFileInfo] = []
        for fn in filenames:
            info = self.parse_file_info(fn)
            if info:
                files.append(info)

        if not files:
            raise ArchiveParserError(
                f"No valid StormEvents files discovered at {self.base_url}. "
                "The archive directory structure or naming scheme may have changed."
            )

        logger.info(f"Discovered {len(files)} StormEvents files in NOAA archive.")
        return files

    def get_available_years(self, files: Optional[List[NOAAFileInfo]] = None) -> List[int]:
        """Return a sorted list of all unique years available in the archive."""
        if files is None:
            files = self.discover_available_files()
        years = sorted({f.year for f in files})
        return years

    def get_complete_calendar_years(
        self, files: Optional[List[NOAAFileInfo]] = None, count: int = 10
    ) -> Tuple[int, int]:
        """Determine the start and end years for the most recent complete calendar years.

        Excludes the latest year if it represents the current ongoing calendar year.
        Returns (start_year, end_year).
        """
        years = self.get_available_years(files)
        if not years:
            raise ArchiveParserError("Cannot determine complete calendar years: no years available.")

        # Latest year in archive might be ongoing; exclude it if more than 1 year is present
        complete_years = years[:-1] if len(years) > 1 else years
        selected = complete_years[-count:] if len(complete_years) >= count else complete_years
        start_year = selected[0]
        end_year = selected[-1]
        logger.info(
            f"Determined latest {len(selected)} complete calendar years: {start_year} to {end_year} "
            f"(excluded ongoing year: {years[-1]})"
        )
        return start_year, end_year

    def select_files(
        self,
        start_year: int,
        end_year: int,
        dataset_types: Optional[List[str]] = None,
        files: Optional[List[NOAAFileInfo]] = None,
    ) -> List[NOAAFileInfo]:
        """Select files for the requested year range and dataset types.

        If multiple creation dates exist for the same (type, year), the one with
        the highest creation date (most recent update) is selected.
        """
        if files is None:
            files = self.discover_available_files()

        target_types = set(t.lower() for t in (dataset_types or ["details", "locations", "fatalities"]))

        # Group by (dataset_type, year) -> pick max creation_date
        grouped: Dict[Tuple[str, int], NOAAFileInfo] = {}
        for f in files:
            if f.dataset_type in target_types and start_year <= f.year <= end_year:
                key = (f.dataset_type, f.year)
                if key not in grouped or f.creation_date > grouped[key].creation_date:
                    grouped[key] = f

        selected_files = sorted(grouped.values(), key=lambda x: (x.year, x.dataset_type))
        logger.info(
            f"Selected {len(selected_files)} files for years {start_year}-{end_year} "
            f"covering types: {sorted(target_types)}"
        )
        return selected_files
