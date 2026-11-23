
import cdsapi
import os
import getpass

# ==========================================
# 1. ENTER YOUR CDS API CREDENTIALS
# ==========================================

CDS_URL = os.environ.get("CDSAPI_URL", "https://cds.climate.copernicus.eu/api")
CDS_API_KEY = os.environ.get("CDSAPI_KEY")

# Initialize the CDS client (reads from ~/.cdsapirc or environment if key is None)
if CDS_API_KEY:
    client = cdsapi.Client(url=CDS_URL, key=CDS_API_KEY)
else:
    client = cdsapi.Client()

# ==========================================
# 2. DOWNLOAD SETTINGS
# ==========================================

OUTPUT_FOLDER = "ERA5_Florida_2024"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Florida bounding box: North, West, South, East
AREA = [31.0, -88.0, 24.0, -79.5]

DATASET = "reanalysis-era5-single-levels"

VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "mean_sea_level_pressure",
    "total_precipitation",
    "convective_precipitation",
    "convective_available_potential_energy",
    "convective_inhibition",
    "total_cloud_cover",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_column_water_vapour",
]

# Correct number of days in each month of leap year 2024
DAYS_IN_MONTH = {
    1: 31, 2: 29, 3: 31, 4: 30,
    5: 31, 6: 30, 7: 31, 8: 31,
    9: 30, 10: 31, 11: 30, 12: 31
}

# ==========================================
# 3. DOWNLOAD MONTH BY MONTH
# ==========================================

for month in range(1, 13):

    month_str = f"{month:02d}"

    output_file = os.path.join(
        OUTPUT_FOLDER,
        f"era5_florida_2024_{month_str}.nc"
    )

    # Skip an existing non-empty file
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"Skipping existing file: {output_file}")
        continue

    request = {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": ["2024"],
        "month": [month_str],
        "day": [
            f"{day:02d}"
            for day in range(1, DAYS_IN_MONTH[month] + 1)
        ],
        "time": [
            "00:00",
            "06:00",
            "12:00",
            "18:00"
        ],
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    temp_file = output_file + ".part"

    try:
        print(f"\nDownloading ERA5 data for 2024-{month_str}...")

        client.retrieve(
            DATASET,
            request,
            temp_file
        )

        # Rename only after a successful download
        os.replace(temp_file, output_file)

        print(f"Download successful: {output_file}")

    except Exception as error:
        print(f"Download failed for month {month_str}: {error}")

        if os.path.exists(temp_file):
            os.remove(temp_file)

        print("You can rerun the script to retry this month.")

print("\nDownload process finished.")
