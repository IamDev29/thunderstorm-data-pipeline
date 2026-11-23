# Florida Thunderstorm 2024 Spatiotemporal Merge Report

## 1. Harmonization Summary

- **Master Grid Rows**: 53,092
- **Unique Timestamps**: 244
- **Unique Florida Grid Cells**: 218
- **NOAA Severe Weather Events Matched**: 227
- **NOAA Events Outside Time/Space Grid**: 814
- **GOES-16 GLM Covered Rows**: 26
- **GOES-16 GLM Lightning Positive Rows**: 26

## 2. Target Label (`storm_label`) Distribution

```json
{
  "-1": 52977,
  "1": 115
}
```

## 3. Label Source Attribution

```json
{
  "UNKNOWN": 52977,
  "NOAA_MATCH": 89,
  "GLM_SATELLITE": 26
}
```

## 4. Spatiotemporal Alignment Tolerances

- **Spatial Tolerance**: 0.25 degrees (~25 km radius)
- **Temporal Window**: ±3.0 hours
- **Sort Order**: ASCENDING (`timestamp_utc`, `latitude`, `longitude`)
