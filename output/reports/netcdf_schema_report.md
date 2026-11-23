# NASA ISS-LIS NetCDF4 Science Data Schema Report

- **Data Model**: `NETCDF4`
- **Schema Consistency across Timeline**: `Consistent across all inspected granules`
- **Total Variables**: `147`

## Scientific Observation Hierarchy

NASA ISS-LIS represents optical lightning detections in a strict 4-level spatial-temporal hierarchy:

1. **Event (`lightning_event_*`)**: An optical transient exceeding background threshold in a single pixel during a single 2-ms frame.
2. **Group (`lightning_group_*`)**: One or more contiguous events in the same 2-ms frame.
3. **Flash (`lightning_flash_*`)**: A set of groups clustered within ~330 ms and ~16.5 km spatial proximity.
4. **Area (`lightning_area_*`)**: Spatial clustering of related flashes.

> [!IMPORTANT]
> For thunderstorm prediction and ML modeling with ERA5, **Flash (`lightning_flash_*`)** is the standard lightning observation unit. Each flash record includes child group counts, grandchild event counts, radiant energy, duration, geographic footprint, and calibrated coordinates. Group and event definitions are preserved via relational parent/child pointers.

## Key Flash Variables (`lightning_flash_*`)

| Variable Name | NetCDF Dimension | Dtype | Units | Description |
|---|---|---|---|---|
| `lightning_flash_TAI93_time` | `flash_dim` | float64 | seconds since 1993-01-01 | TAI93 timestamp of the initial event |
| `lightning_flash_delta_time` | `flash_dim` | float32 | seconds | Time span between first and last group in flash |
| `lightning_flash_observe_time` | `flash_dim` | int16 | seconds | Duration of observation of the flash region |
| `lightning_flash_lat` | `flash_dim` | float32 | degrees_north | Centroid latitude (-90 to +90) |
| `lightning_flash_lon` | `flash_dim` | float32 | degrees_east | Centroid longitude (-180 to +180) |
| `lightning_flash_radiance` | `flash_dim` | float32 | uJ/sr/m2/um | Calibrated radiance (sum of events) |
| `lightning_flash_footprint` | `flash_dim` | float32 | km2 | Geographic area covered by the flash |
| `lightning_flash_address` | `flash_dim` | int32 | 1 | Unique record identifier within granule |
| `lightning_flash_child_count` | `flash_dim` | int32 | count | Number of groups comprising this flash |
| `lightning_flash_grandchild_count` | `flash_dim` | int32 | count | Number of events comprising this flash |
| `lightning_flash_alert_flag` | `flash_dim` | uint8 | flag | Instrument, platform, & processing quality alert bitmask |
| `lightning_flash_noise_index` | `flash_dim` | int8 | % | Signal-to-noise ratio confidence |
| `lightning_flash_oblong_index` | `flash_dim` | float32 | 1 | Spatial eccentricity metric |

## Instrument Viewing & Coverage Geometry (`viewtime_*`)

The ISS-LIS instrument has a field of view (~600 km swath) moving along orbit (~90 min period). Chandipur is observed intermittently (~90 seconds per overpass).

| Variable Name | NetCDF Dimension | Dtype | Description |
|---|---|---|---|
| `viewtime_lat` | `viewtime_dim` | float32 | Center latitude of 0.5° × 0.5° grid cell |
| `viewtime_lon` | `viewtime_dim` | float32 | Center longitude of 0.5° × 0.5° grid cell |
| `viewtime_TAI93_start` | `viewtime_dim` | int32 | TAI93 second when cell entered instrument FOV |
| `viewtime_TAI93_end` | `viewtime_dim` | int32 | TAI93 second when cell exited instrument FOV |
| `viewtime_effective_obs` | `viewtime_dim` | float32 | Effective observation seconds within cell |

> [!NOTE]
> When evaluating thunderstorm prediction models, a zero-lightning observation can ONLY be established if `viewtime` shows Chandipur was in the FOV during an overpass. Periods where Chandipur was outside the FOV must be classified as unobserved, not non-thunderstorm.

## Time Representation and Conversion

- **Base Epoch**: 1993-01-01 00:00:00 TAI (`TAI93`)
- **Leap Second Offset**: In 2023, TAI - UTC difference is exactly 10.0 seconds.
- **Verification**: `orbit_summary_TAI93_start - (orbit_summary_UTC_start - 1993-01-01 UTC) = 10.000s` across granules.
- **Normalized Outputs**: Both UTC (`timestamp_utc`, `date_utc`, `hour_utc`) and Indian Standard Time (`timestamp_ist`, `date_ist`, `hour_ist`, UTC+05:30) are extracted with microsecond fidelity.