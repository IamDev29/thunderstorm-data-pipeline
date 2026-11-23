# Florida Thunderstorm Model Training Readiness Report

## 1. Machine Learning Readiness Assessment

The consolidated dataset has undergone rigorous spatiotemporal verification.

### Target Variable Definition & Distribution:
A multi-source defensible classification target (`storm_label`) was established:
- **Class `1` (Confirmed Storm)**: Verified severe thunderstorm events from NOAA Storm Events database OR verified GOES-16 GLM lightning detections.
- **Class `0` (Confirmed Non-Storm)**: Valid satellite observation confirming zero lightning AND quiescent thermodynamic conditions (CAPE < 300 J/kg, total precip < 0.1 mm).
- **Class `-1` (Unlabeled / Uncertain)**: Unmonitored observation windows (no satellite overpass) or ambiguous convection without storm impacts.

```json
{
  "-1": 52977,
  "1": 115
}
```

### Label Source Attribution:

```json
{
  "UNKNOWN": 52977,
  "NOAA_MATCH": 89,
  "GLM_SATELLITE": 26
}
```

## 2. Leakage Protection Certification

- **Zero Target Leakage**: Confirmed that storm narratives, NOAA event IDs, and concurrent lightning observations are quarantined from the predictor feature set.
- **Strict Ascending Sort Order**: Confirmed ascending chronological (`timestamp_utc`) and spatial (`latitude`, `longitude`) sorting to enable time-series splits (e.g., rolling origin or blocking) without lookahead bias.

## 3. Recommended ML Validation Strategy

1. **Temporal Split**: Use early months for training (e.g., Jan–Aug 2024) and late months for testing (e.g., Sept–Dec 2024).
2. **Spatial Split**: Evaluate generalization on held-out Florida subregions (e.g., train on Peninsula, validate on Panhandle).
3. **Semi-Supervised / PU Learning**: Utilize Positive-Unlabeled (PU) learning or label-masking for class `-1` rows.
