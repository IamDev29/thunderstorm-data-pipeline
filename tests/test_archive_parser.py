"""Unit tests for NOAA Archive Parser."""

import pytest
from src.archive_parser import ArchiveParserError, NOAAArchiveParser, NOAAFileInfo


SAMPLE_HTML = """
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
<html>
 <head>
  <title>Index of /pub/data/swdi/stormevents/csvfiles</title>
 </head>
 <body>
<h1>Index of /pub/data/swdi/stormevents/csvfiles</h1>
<table>
<tr><th><a href="?C=N;O=D">Name</a></th></tr>
<tr><td><a href="StormEvents_details-ftp_v1.0_d2023_c20250101.csv.gz">StormEvents_details-ftp_v1.0_d2023_c20250101.csv.gz</a></td></tr>
<tr><td><a href="StormEvents_details-ftp_v1.0_d2023_c20260323.csv.gz">StormEvents_details-ftp_v1.0_d2023_c20260323.csv.gz</a></td></tr>
<tr><td><a href="StormEvents_locations-ftp_v1.0_d2023_c20260707.csv.gz">StormEvents_locations-ftp_v1.0_d2023_c20260707.csv.gz</a></td></tr>
<tr><td><a href="StormEvents_fatalities-ftp_v1.0_d2023_c20260323.csv.gz">StormEvents_fatalities-ftp_v1.0_d2023_c20260323.csv.gz</a></td></tr>
<tr><td><a href="StormEvents_details-ftp_v1.0_d2024_c20260728.csv.gz">StormEvents_details-ftp_v1.0_d2024_c20260728.csv.gz</a></td></tr>
<tr><td><a href="StormEvents_details-ftp_v1.0_d2025_c20260819.csv.gz">StormEvents_details-ftp_v1.0_d2025_c20260819.csv.gz</a></td></tr>
<tr><td><a href="StormEvents_details-ftp_v1.0_d2026_c20260918.csv.gz">StormEvents_details-ftp_v1.0_d2026_c20260918.csv.gz</a></td></tr>
<tr><td><a href="Storm-Data-Bulk-csv-Format.pdf">Storm-Data-Bulk-csv-Format.pdf</a></td></tr>
</table>
</body>
</html>
"""


def test_parse_file_info_valid():
    parser = NOAAArchiveParser()
    fn = "StormEvents_details-ftp_v1.0_d2024_c20260728.csv.gz"
    info = parser.parse_file_info(fn)
    assert info is not None
    assert info.filename == fn
    assert info.dataset_type == "details"
    assert info.year == 2024
    assert info.creation_date == 20260728
    assert info.is_compressed is True
    assert info.url.endswith(fn)


def test_parse_file_info_invalid():
    parser = NOAAArchiveParser()
    assert parser.parse_file_info("Storm-Data-Bulk-csv-Format.pdf") is None
    assert parser.parse_file_info("random_file.csv") is None


def test_discover_available_files_from_mock_html():
    parser = NOAAArchiveParser()
    files = parser.discover_available_files(html_content=SAMPLE_HTML)
    assert len(files) == 7  # 7 valid StormEvents files in SAMPLE_HTML
    years = parser.get_available_years(files)
    assert years == [2023, 2024, 2025, 2026]


def test_select_files_chooses_latest_creation_date():
    parser = NOAAArchiveParser()
    files = parser.discover_available_files(html_content=SAMPLE_HTML)
    # Notice 2023 has two details files: c20250101 and c20260323
    selected = parser.select_files(start_year=2023, end_year=2023, dataset_types=["details"], files=files)
    assert len(selected) == 1
    assert selected[0].creation_date == 20260323
    assert selected[0].filename == "StormEvents_details-ftp_v1.0_d2023_c20260323.csv.gz"


def test_get_complete_calendar_years_excludes_current_ongoing_year():
    parser = NOAAArchiveParser()
    files = parser.discover_available_files(html_content=SAMPLE_HTML)
    start_yr, end_yr = parser.get_complete_calendar_years(files=files, count=2)
    # Available are [2023, 2024, 2025, 2026].
    # 2026 is excluded as the ongoing year. Complete years are [2023, 2024, 2025].
    # Last 2 complete years are 2024 to 2025.
    assert end_yr == 2025
    assert start_yr == 2024


def test_parser_raises_on_empty_content():
    parser = NOAAArchiveParser()
    with pytest.raises(ArchiveParserError):
        parser.discover_available_files(html_content="<html><body>No files</body></html>")
