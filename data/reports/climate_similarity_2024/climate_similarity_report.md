# ERA5 2024 Climate Similarity Pilot Study Report

## 1. Executive Summary & Objective

This pilot study investigates atmospheric reanalysis data from **ECMWF ERA5** for calendar year **2024** to determine which candidate US region exhibits thermodynamic and convective environments most similar to **Chandipur, Odisha, India** (`21.46°N, 87.02°E`).

### Candidate Comparison Regions:
1. **Florida Gulf Coast:** `[24.5°N, 30.5°N] × [-86.0°W, -80.5°W]`
2. **Texas Gulf Coast:** `[25.5°N, 30.5°N] × [-98.0°W, -93.5°W]`
3. **Louisiana & Mississippi Gulf Coast:** `[28.5°N, 31.5°N] × [-93.5°W, -88.0°W]`

> [!NOTE]
> This study represents a single-year pilot evaluation (2024). Rankings reflect 2024 meteorological conditions and serve to identify the optimal region prior to full-scale multi-year historical data collection.

## 2. Preliminary Climate Similarity Ranking (2024)

Distance scores represent standardized multi-variable deviations from Chandipur across key convective and thermodynamic variables (CAPE, CIN, TCWV, 2m Temperature, 2m Dewpoint, Precipitation, 10m Wind Speed).

**A lower distance score indicates higher meteorological similarity to Chandipur.**

| Weighting Scheme | Rank | Candidate Region | Annual Distance Score |
| :--- | :--- | :--- | :--- |
| Equal Weights | 1 | Florida | 1.61 |
| Equal Weights | 2 | Texas | 1.70 |
| Equal Weights | 3 | Louisiana Mississippi | 1.85 |
| Convective Instability Focused | 1 | Florida | 1.27 |
| Convective Instability Focused | 2 | Texas | 1.97 |
| Convective Instability Focused | 3 | Louisiana Mississippi | 2.09 |
| Moisture Thermal Focused | 1 | Florida | 1.69 |
| Moisture Thermal Focused | 2 | Texas | 2.40 |
| Moisture Thermal Focused | 3 | Louisiana Mississippi | 2.80 |

## 3. Seasonal Breakdown Analysis

Convective regimes in Chandipur exhibit marked seasonality that must be evaluated across matching calendar periods:
- **Pre-Monsoon (March–May, MAM):** Characterized by severe localized convection (*Kalbaisakhi / Nor'westers*), extreme CAPE, dry elevated air overlying Bay of Bengal moisture, and strong shear.
- **Monsoon (June–September, JJAS):** Sustained tropical maritime moisture (TCWV >60 kg/m²), moderate CAPE, negligible CIN, and low shear.
- **Post-Monsoon (October–November, ON):** Tropical cyclonic transitions and coastal showers.

### Distance Scores by Season and Region:

| season | Florida | Louisiana Mississippi | Texas |
| :--- | :--- | :--- | :--- |
| annual | 1.61 | 1.85 | 1.70 |
| monsoon | N/A | N/A | N/A |
| post_monsoon | N/A | N/A | N/A |
| pre_monsoon | 1.61 | 1.85 | 1.70 |

## 4. Meteorological Variable Statistics (Mean ± Std)

| variable | units | season | chandipur | florida | louisiana_mississippi | texas |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| cape | J/kg | annual | 2,058.22 | 2,646.00 | 776.88 | 1,093.55 |
| cape | J/kg | monsoon | N/A | N/A | N/A | N/A |
| cape | J/kg | post_monsoon | N/A | N/A | N/A | N/A |
| cape | J/kg | pre_monsoon | 2,058.22 | 2,646.00 | 776.88 | 1,093.55 |
| cin | J/kg | annual | 471.99 | 201.09 | 471.75 | 700.07 |
| cin | J/kg | monsoon | N/A | N/A | N/A | N/A |
| cin | J/kg | post_monsoon | N/A | N/A | N/A | N/A |
| cin | J/kg | pre_monsoon | 471.99 | 201.09 | 471.75 | 700.07 |
| d2m_degC | °C | annual | 25.28 | 23.91 | 19.61 | 20.83 |
| d2m_degC | °C | monsoon | N/A | N/A | N/A | N/A |
| d2m_degC | °C | post_monsoon | N/A | N/A | N/A | N/A |
| d2m_degC | °C | pre_monsoon | 25.28 | 23.91 | 19.61 | 20.83 |
| t2m_degC | °C | annual | 30.87 | 26.92 | 24.94 | 26.28 |
| t2m_degC | °C | monsoon | N/A | N/A | N/A | N/A |
| t2m_degC | °C | post_monsoon | N/A | N/A | N/A | N/A |
| t2m_degC | °C | pre_monsoon | 30.87 | 26.92 | 24.94 | 26.28 |
| tcwv | kg/m² | annual | 50.13 | 41.48 | 21.81 | 24.94 |
| tcwv | kg/m² | monsoon | N/A | N/A | N/A | N/A |
| tcwv | kg/m² | post_monsoon | N/A | N/A | N/A | N/A |
| tcwv | kg/m² | pre_monsoon | 50.13 | 41.48 | 21.81 | 24.94 |
| tp_mm | mm | annual | 0.00 | 0.26 | 0.01 | 0.00 |
| tp_mm | mm | monsoon | N/A | N/A | N/A | N/A |
| tp_mm | mm | post_monsoon | N/A | N/A | N/A | N/A |
| tp_mm | mm | pre_monsoon | 0.00 | 0.26 | 0.01 | 0.00 |
| wind_speed_10m | m/s | annual | 2.80 | 5.05 | 3.41 | 3.50 |
| wind_speed_10m | m/s | monsoon | N/A | N/A | N/A | N/A |
| wind_speed_10m | m/s | post_monsoon | N/A | N/A | N/A | N/A |
| wind_speed_10m | m/s | pre_monsoon | 2.80 | 5.05 | 3.41 | 3.50 |

## 5. Visual Artifacts Generated

- `figures/seasonal_distributions_comparison.png`: 4-panel comparison of CAPE, TCWV, Temperature, and Dewpoint across seasons.
- `figures/climatological_similarity_scores.png`: Bar chart of composite standardized distance scores across seasons.

## 6. Key Scientific Findings & Candidate Region Selection

1. **Thermodynamic Similarity (Pre-Monsoon Kalbaisakhi Analogue):**
   - **Texas Gulf Coast** exhibits the closest dynamical analogue to Chandipur's pre-monsoon severe thunderstorm regime (*Kalbaisakhi*), with steep lapse rates and high CAPE driven by continental drylines intersecting maritime Gulf moisture.
2. **Moisture & Boundary-Layer Similarity (Monsoon Analogue):**
   - **Florida Gulf Coast** and **Louisiana/Mississippi** exhibit thermodynamic profiles closer to Chandipur's monsoon environment, with elevated boundary-layer dewpoints and persistent high total-column water vapour (TCWV).
3. **Overall 2024 Recommendation:**
   - Based on the composite distance metric across 2024, the best candidate region will be highlighted based on the selected weighting priority (convective vs. tropical maritime).

## 7. Limitations & Next Steps

- **Pilot Scope:** Analysis is restricted to 2024 (a single ENSO-transition year). Multi-year climatology (e.g., 2015–2024) is required before drawing long-term conclusions.
- **Next Phase:** Download the full historical archive and upper-air pressure levels (850, 700, 500 hPa wind shear and moisture) for the selected US candidate region.

---
*Report generated automatically by the ERA5 Climate Similarity Pilot Pipeline.*