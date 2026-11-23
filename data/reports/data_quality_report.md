# NOAA Storm Events Pipeline - Data Quality & Preprocessing Report

## 1. Executive Summary

| Metric | Value |
| :--- | :--- |
| **Total Archive Files Examined** | 3 |
| **Successfully Downloaded / Verified** | 3 |
| **Failed Downloads** | 0 |
| **Total Raw Event Records Ingested** | 69,801 |
| **Total Cleaned & Standardized Records** | 6,899 |
| **Unique Event IDs** | 6,899 |
| **Duplicate Records Detected / Handled** | 0 |

## 2. Coordinate & Temporal Data Quality

| Coordinate / Temporal Metric | Count | Description |
| :--- | :--- | :--- |
| **Missing Coordinates** | 3,684 | Events lacking latitude/longitude values |
| **Invalid Coordinates** | 0 | Out of bounds [-90,90]x[-180,180] or Null Island (0,0) |
| **Retained Missing-Coord Records** | 0 | Preserved in state datasets with `insufficient_coords=True` |
| **Missing / Unparseable Timestamps** | 0 | Events lacking valid begin timestamps |
| **Records Excluded by Bounding Box** | 0 | Excluded from coastal bounding box |
| **Records Excluded by Event Type** | 0 | Excluded from convective hazard subset |

## 3. Climatological Comparison Reference Note

- **Candidate US Regions:** Florida, Texas, Louisiana, Mississippi (Gulf Coast focus)
- **Comparative Site:** Chandipur, Odisha, India
- **Approximate Reference Coordinates:** Latitude 21.46° N, Longitude 87.02° E
- *Note:* Chandipur coordinates are regional reference coordinates, not a single verified meteorological station.

## 4. Event Distributions

### 4.1 Event Counts by Year

| year | event_count | direct_injuries | direct_deaths | total_property_damage_usd |
| :--- | :--- | :--- | :--- | :--- |
| 2,024 | 6,899 | 166 | 72 | 7,885,020,150 |

### 4.2 Event Counts by State

| state | event_count |
| :--- | :--- |
| TEXAS | 3,459 |
| FLORIDA | 1,441 |
| LOUISIANA | 1,092 |
| MISSISSIPPI | 907 |

### 4.3 Event Counts by Event Type

| event_type | event_count |
| :--- | :--- |
| Thunderstorm Wind | 1,325 |
| Heat | 1,143 |
| Flash Flood | 697 |
| Drought | 590 |
| Hail | 579 |
| Excessive Heat | 567 |
| High Wind | 334 |
| Tornado | 320 |
| Tropical Storm | 242 |
| Cold/Wind Chill | 155 |
| Winter Weather | 127 |
| Flood | 102 |
| Heavy Rain | 101 |
| Winter Storm | 84 |
| Storm Surge/Tide | 71 |
| Extreme Cold/Wind Chill | 54 |
| Strong Wind | 52 |
| Funnel Cloud | 51 |
| Hurricane (Typhoon) | 47 |
| Frost/Freeze | 47 |
| Wildfire | 43 |
| Lightning | 40 |
| Coastal Flood | 37 |
| Tropical Depression | 29 |
| Rip Current | 21 |
| Ice Storm | 11 |
| Astronomical Low Tide | 10 |
| Blizzard | 8 |
| Dust Storm | 6 |
| Dense Fog | 3 |
| Heavy Snow | 3 |

### 4.4 Event Counts by Month

| month_name | event_count |
| :--- | :--- |
| January | 1,002 |
| February | 280 |
| March | 234 |
| April | 638 |
| May | 1,261 |
| June | 736 |
| July | 646 |
| August | 881 |
| September | 478 |
| October | 286 |
| November | 184 |
| December | 273 |

## 5. Artifacts and Visualizations Generated

- `figures/event_counts_by_year_state.png`
- `figures/event_types_distribution.png`
- `figures/monthly_seasonality.png`
- `figures/spatial_distribution_map.png`

---
*Report generated automatically by the NOAA Storm Events Preprocessing Pipeline.*