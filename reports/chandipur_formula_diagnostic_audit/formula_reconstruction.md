# Chandipur Storm Score Formula: Exact Mathematical Reconstruction & Verification

**Audit Directory:** `reports/chandipur_formula_diagnostic_audit/`  
**Dataset Under Audit:** `MET_DATA_with_storm_score_label.csv` (23,376 synoptic observations, 2022–2025)  
**Auditor:** Independent Thunderstorm Prediction Verification Engine  
**Status:** DIAGNOSTIC ONLY — NO MODIFICATIONS MADE  

---

## 1. Mathematical Formulation

The existing Chandipur proxy thunderstorm classification system operates as a deterministic, contemporaneous composite index:

$$\text{Storm Score} = \sum_{i=1}^8 w_i \times f_i(X_i)$$

$$\text{storm\_label\_score} = \begin{cases} 1 & \text{if } \text{Storm Score} \ge 2.50 \\ 0 & \text{if } \text{Storm Score} < 2.50 \end{cases}$$

Across all 23,376 observations, there are **0 classification mismatches** between `storm_score` and `storm_label_score`:
$$\sum_{k=1}^{23376} \left| \text{storm\_label\_score}_k - \mathbb{I}(\text{storm\_score}_k \ge 2.50) \right| = \mathbf{0}$$

---

## 2. Parameter Specification, Weights, and Physical Mappings

The formula combines eight atmospheric parameters extracted from ERA5 reanalysis fields:

| Parameter ID | Atmospheric Variable | CSV Column Name | Assigned Weight $w_i$ | Vertical Pressure Level | Physical Dimension / Units | Scaling Domain $f_i(X)$ | Observed Direction in Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$X_1$** | Convective Available Potential Energy | `cape_Jkg` | **$+1.2$** | Surface / Parcel | $\text{J/kg}$ | Normalized $[0, 1]$ (sat. $\sim 3000\text{ J/kg}$) | Positive ($r = +0.255$) |
| **$X_2$** | Convective Precipitation | `cp_24h_sum` | **$+1.0$** | Surface (24h backward sum) | $\text{mm}$ | Normalized $[0, 1]$ (sat. $\sim 10\text{ mm}$) | Positive ($r = +0.564$) |
| **$X_3$** | Relative Humidity | `relative_humidity_300hPa` | **$+0.8$** | Upper Troposphere ($300\text{ hPa}$) | Percentage $[0, 100\%]$ | Direct Fraction $[0.0, 1.0]$ | Strongly Positive ($r = +0.742$) |
| **$X_4$** | Vertical Velocity Lift | `vertical_velocity_Pas_300hPa` | **$+1.0$** | Upper Troposphere ($300\text{ hPa}$) | $\text{Pa/s}$ ($\omega < 0 \implies \text{lift}$) | Normalized $[0, 1]$ (ascent magnitude) | Positive ($r = +0.470$ for $-\omega$) |
| **$X_5$** | Deep-Layer Wind Shear | `bulk_shear_850_300` | **$+0.7$** | $850\text{ hPa} \rightarrow 300\text{ hPa}$ | $\text{m/s}$ | Normalized $[0, 1]$ (sat. $\sim 40\text{ m/s}$) | Negative linear ($r = -0.302$) |
| **$X_6$** | Total Cloud Cover | `total_cloud_cover` | **$+0.4$** | Total Atmospheric Column | Fraction $[0.0, 1.0]$ | Native Fraction $[0.0, 1.0]$ | Positive ($r = +0.656$) |
| **$X_7$** | Low-Level Wind Speed | `wind_speed_10m` | **$+0.5$** | $10\text{ m}$ Surface | $\text{m/s}$ | Normalized $[0, 1]$ (sat. $\sim 12\text{ m/s}$) | Positive ($r = +0.423$) |
| **$X_8$** | Convective Inhibition | `cin_Jkg` | **$-0.8$** | Boundary Layer Parcel | $\text{J/kg}$ ($\ge 0$) | Normalized $[0, 1]$ (sat. $\sim 200\text{ J/kg}$) | Suppressive ($r = -0.113$) |

---

## 3. Boundary Conditions & Numerical Bounds

* **Theoretical Maximum Score:**  
  $$\text{Score}_{\text{max}} = 1.2(1) + 1.0(1) + 0.8(1) + 1.0(1) + 0.7(1) + 0.4(1) + 0.5(1) - 0.8(0) = \mathbf{5.60}$$
  * **Observed Dataset Maximum:** **$5.5720$** (99.5% of theoretical max).
* **Theoretical Minimum Score:**  
  $$\text{Score}_{\text{min}} = 0 + 0 + 0 + 0 + 0 + 0 + 0 - 0.8(1) = \mathbf{-0.80}$$
  * **Observed Dataset Minimum:** **$-0.2155$** (occurs in completely dry, stable winter air with moderate CIN).
* **Observed Mean and Standard Deviation:**  
  $$\mu = \mathbf{1.8799}, \quad \sigma = \mathbf{0.8540}, \quad \text{Median} = \mathbf{1.8152}$$
* **Threshold Placement:**  
  The cutoff $\tau = 2.50$ sits at:
  $$\tau = \mu + 0.726 \sigma$$
  This isolates **$5,462$ positive observations ($23.37\%$)** and **$17,914$ negative observations ($76.63\%$)**.

---

## 4. Component Contribution & Decisive Impact Analysis

Compiled in [`reports/chandipur_formula_diagnostic_audit/component_contribution_analysis.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_diagnostic_audit/component_contribution_analysis.csv):

| Component Name | Weight | Level | Mean Overall | Mean Positives | Mean Negatives | Delta ($\Delta$) | Correlation with Score ($r$) | Decisive Count (Positives) | Decisive % of Positives |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Upper-Level RH (`contrib_rh`)** | **$+0.8$** | $300\text{ hPa}$ | 0.3188 | **0.5965** | 0.2341 | **+0.3623** | **+0.7425** | **3,209** | **58.75%** |
| **Thermodynamic CAPE (`contrib_cape`)** | **$+1.2$** | Surface | 0.4496 | **0.5685** | 0.4133 | +0.1552 | +0.3148 | **2,758** | **50.49%** |
| **Total Cloud Cover (`contrib_cloud`)** | **$+0.4$** | Column | 0.1990 | **0.3430** | 0.1550 | +0.1880 | +0.6559 | **2,071** | **37.92%** |
| **Deep-Layer Shear (`contrib_shear`)** | **$+0.7$** | $850-300$ | 0.3068 | 0.2345 | 0.3289 | -0.0943 | -0.3032 | **1,622** | **29.70%** |
| **Low-Level Wind (`contrib_wind`)** | **$+0.5$** | $10\text{ m}$ | 0.1773 | 0.2299 | 0.1613 | +0.0686 | +0.4223 | **1,539** | **28.18%** |
| **Vertical Lift (`contrib_lift`)** | **$+1.0$** | $300\text{ hPa}$ | 0.1136 | 0.2975 | 0.0575 | +0.2400 | +0.5250 | **1,108** | **20.29%** |
| **24h Precip Sum (`contrib_cp`)** | **$+1.0$** | Surface | 0.0440 | 0.1344 | 0.0165 | +0.1179 | +0.5682 | **440** | **8.06%** |
| **CIN Penalty (`penalty_cin`)** | **$-0.8$** | Boundary | -0.3318 | -0.2662 | -0.3518 | +0.0856 | +0.0173 | **0** | **0.00%** |

*Decisive Count denotes the number of proxy positives where: $(\text{Storm Score} - \text{Contrib}_i) < 2.50$.*

---

## 5. Critical Vulnerabilities Revealed by Reconstruction

1. **Upper-Tropospheric Moisture Dominance:**
   * `relative_humidity_300hPa` is the **single most decisive feature in the formula**, accounting for **$58.75\%$** of all positive classifications and exhibiting a massive Pearson correlation of **$r = 0.7425$** with `storm_score`.
   * In physical meteorology, 300 hPa (~9 km altitude) sits near the tropical tropopause. Moisture at 300 hPa is an **after-effect** of previous deep convective anvil outflow or mid-latitude cirrus, not a pre-convective trigger.
2. **Linear Additive Compensation Flaw:**
   * Because terms are added linearly without non-linear gating, the non-thermodynamic components ($X_3$ through $X_7$) can sum to:
     $$\text{RH}_{300} (0.80) + \text{Cloud} (0.40) + \text{Shear} (0.55) + \text{Lift} (0.50) + \text{Wind} (0.25) = \mathbf{2.50}$$
   * As demonstrated in Case 176 (`2022-01-12 00:00 UTC`), a completely stable, non-convective atmosphere with $\text{CAPE} = 18.9\text{ J/kg}$ and $\text{CP} = 0.01\text{ mm}$ is classified as a thunderstorm proxy positive simply because upper-tropospheric humidity and cloudiness are elevated.
3. **Ineffective CIN Suppression:**
   * The CIN penalty is subtracted linearly ($-0.8 \times f(\text{CIN})$).
   * A heavy capping inversion ($\text{CIN} > 500\text{ J/kg}$) only subtracts $0.80$ points, which is easily overwhelmed by high CAPE ($+1.20$) and moisture ($+0.80$). Consequently, **$7.50\%$ of all extreme capping states ($\text{CIN} > 500\text{ J/kg}$) are incorrectly labeled as thunderstorms**.
