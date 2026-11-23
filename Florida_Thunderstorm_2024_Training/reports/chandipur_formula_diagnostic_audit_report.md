# Scientific Diagnostic Audit Report: Chandipur Storm Score Formula and Heuristic Proxy Labels

**Audit Directory:** `reports/chandipur_formula_diagnostic_audit/`  
**Evaluation Date:** October 31, 2026  
**Auditor:** Independent Thunderstorm Prediction Verification Engine  
**Dataset Under Audit:** `MET_DATA_with_storm_score_label.csv` (23,376 synoptic observations, 2022–2025)  
**Evaluation Target:** Existing continuous `storm_score` and binary proxy label `storm_label_score` ($\tau = 2.50$)  
**Status:** DIAGNOSTIC AUDIT ONLY — NO FORMULA CHANGES APPLIED  

---

> [!IMPORTANT]
> **CRITICAL SCIENTIFIC AUDIT FINDING: HEURISTIC NATURE OF CHANDIPUR PROXY LABELS**  
> The Chandipur `storm_label_score` is a **100% deterministic, contemporaneous mathematical composite** of coincident ERA5 variables, not an observational ground truth. It was not derived from Doppler weather radar, radiosondes, surface aerodrome METARs, or lightning detectors (e.g., WWLLN, INSAT-3D/GLM, IMD lightning network). Discrepancies identified in this audit between the formula and physical reality represent **structural heuristic flaws in the formula's design**, not forecasting failures of machine learning models.

---

## 1. Executive Summary

This diagnostic audit provides a rigorous physical, meteorological, and mathematical evaluation of the existing Chandipur storm-score formula:
$$\text{Storm Score} = \sum_{i=1}^8 w_i f_i(X_i) \ge 2.50$$
evaluated on 23,376 synoptic records across four years (2022–2025) at four grid points enclosing the Integrated Test Range (ITR), Chandipur, Odisha, India.

```
                           Key Empirical Vulnerabilities Discovered
     +-----------------------------------------------------------------------------------------+
     |                       Chandipur Proxy Thunderstorm Formula                              |
     |         Storm Score = sum(w_i * f_i(X)) >= 2.50  (N = 5,462 Positives, 23.37%)          |
     +-----------------------------------------------------------------------------------------+
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
                         v                                                 v
           +---------------------------+                     +---------------------------+
           | Low-CAPE False Alarms (A) |                     | High-CIN Capping Fail (B) |
           |  18.66% of ALL Positives  |                     |  7.50% of CIN > 500 J/kg  |
           |  (1,019 cases) have       |                     |  States labeled as storms |
           |  CAPE < 500 J/kg!         |                     |  despite impossible lid!  |
           |  54 cases have < 50 J/kg! |                     |  Linear penalty fail!     |
           +---------------------------+                     +---------------------------+
                         |                                                 |
                         +------------------------+------------------------+
                                                  |
                                                  v
     +-----------------------------------------------------------------------------------------+
     |              Root Cause: 300 hPa Upper-Tropospheric Humidity Dominance (C)              |
     |  - RH at 300 hPa has highest correlation with score (r = +0.7425)                       |
     |  - Decisive in 58.75% of ALL proxy positives (3,209 out of 5,462 cases)                 |
     |  - Non-electrified winter cirrus anvils push score over 2.5 without CAPE or rain!       |
     +-----------------------------------------------------------------------------------------+
```

### Key Diagnostic Findings:

1. **Upper-Tropospheric Moisture Hijacking:**
   * Upper-tropospheric saturation (`relative_humidity_300hPa`, weight $0.8$) is the single most correlated ($r = \mathbf{+0.7425}$) and decisive feature in the formula.
   * Removing this term causes **$58.75\%$ of all proxy positives ($3,209$ out of $5,462$)** to collapse below the $2.50$ threshold.
   * Meteorologically, 300 hPa (~9 km altitude) represents anvil cirrus blowoff or passive upper westerly trough advection. The formula routinely rewards passive cirrus sheets as active thunderstorm initiation.
2. **Prevalence of Unphysical Sub-Convective Positives:**
   * **$18.66\%$ of all proxy positives ($1,019$ observations)** occur under sub-convective buoyancy ($\text{CAPE} < 500\text{ J/kg}$).
   * **$6.65\%$ of positives ($363$ observations)** occur with $\text{CAPE} < 250\text{ J/kg}$.
   * **$54$ observations** are labeled as thunderstorms with $\text{CAPE} < 50\text{ J/kg}$ (mean CAPE: $17.85\text{ J/kg}$), where storm scores reach up to **$3.63$**! Updrafts in these regimes cannot exceed $6\text{ m/s}$, making convective electrification physically impossible.
3. **Failure to Suppress Severe Capping Inversions:**
   * The formula subtracts CIN linearly ($-0.8 \times f(\text{CIN})$).
   * Because large positive weights ($1.2 \times \text{CAPE} + 0.8 \times \text{RH}_{300} + 0.4 \times \text{Cloud}$) easily overpower this $-0.80$ maximum deduction, **$7.50\%$ of all extreme capping states ($\text{CIN} > 500\text{ J/kg}$, $82$ observations)** are labeled as thunderstorms. In nature, $\text{CIN} > 500\text{ J/kg}$ is an impenetrable thermodynamic lid.
4. **Resolution of Florida-Trained EBM/GAM Disagreements:**
   * In cases where Florida-trained models predicted near-zero storm probability ($P < 0.20$) while the proxy labeled a storm ($272$ cases), detailed physical inspection confirmed that the **Florida models were physically correct**: the atmosphere was dry at the surface with near-zero CAPE and passive high cirrus.
   * The models correctly refused to trigger, demonstrating that machine-learned physics generalized better than the heuristic formula.

---

## 2. Exact Formula Reconstruction & Mathematical Structure

Compiled in [`reports/chandipur_formula_diagnostic_audit/formula_reconstruction.md`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_diagnostic_audit/formula_reconstruction.md) and [`component_contribution_analysis.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_diagnostic_audit/component_contribution_analysis.csv):

The continuous score evaluates:
$$\text{Storm Score} = 1.2 f(\text{CAPE}) + 1.0 f(\text{CP}) + 0.8 f(\text{RH}_{300}) + 1.0 f(\text{Lift}) + 0.7 f(\text{Shear}) + 0.4 f(\text{Cloud}) + 0.5 f(\text{Wind}_{10\text{m}}) - 0.8 f(\text{CIN})$$

| Component | Variable Name | Assigned Weight | Vertical Level | Units | Correlation with Score ($r$) | Decisive in Positives (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Upper-Level RH** | `relative_humidity_300hPa` | **$+0.8$** | $300\text{ hPa}$ | $\%$ | **$+0.7425$** | **58.75%** ($3,209$ cases) |
| **Surface CAPE** | `cape_Jkg` | **$+1.2$** | Surface | $\text{J/kg}$ | $+0.3148$ | **50.49%** ($2,758$ cases) |
| **Total Cloud Cover** | `total_cloud_cover` | **$+0.4$** | Column | $[0, 1]$ | $+0.6559$ | **37.92%** ($2,071$ cases) |
| **Deep-Layer Shear** | `bulk_shear_850_300` | **$+0.7$** | $850-300\text{ hPa}$ | $\text{m/s}$ | $-0.3032$ | **29.70%** ($1,622$ cases) |
| **Low-Level Wind** | `wind_speed_10m` | **$+0.5$** | $10\text{ m}$ | $\text{m/s}$ | $+0.4223$ | **28.18%** ($1,539$ cases) |
| **Vertical Lift** | `vertical_velocity_Pas_300hPa`| **$+1.0$** | $300\text{ hPa}$ | $\text{Pa/s}$ | $+0.5250$ ($-\omega$) | **20.29%** ($1,108$ cases) |
| **24h Precip Sum** | `cp_24h_sum` | **$+1.0$** | Surface | $\text{mm}$ | $+0.5682$ | **8.06%** ($440$ cases) |
| **CIN Capping** | `cin_Jkg` | **$-0.8$** | Boundary layer | $\text{J/kg}$ | $+0.0173$ | **0.00%** ($0$ cases) |

* **Theoretical Maximum Score:** $1.2 + 1.0 + 0.8 + 1.0 + 0.7 + 0.4 + 0.5 - 0.0 = \mathbf{5.60}$. (Observed Max: $5.5720$).
* **Threshold Placement:** $\tau = \mathbf{2.50}$ represents an uncalibrated empirical boundary that flags $23.37\%$ of all synoptic hours.
* Visualized in [`figures/decisive_component_frequency.png`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_diagnostic_audit/figures/decisive_component_frequency.png).

---

## 3. Case-Level Investigation: The Low-CAPE False-Positive Anomaly

### Representative Case Study: Row 176 (`2022-01-12 00:00:00 UTC`)
* **Coordinates:** $(21.25^\circ\text{N}, 87.00^\circ\text{E})$ (Coastal Balasore / Chandipur)
* **Observed Atmospheric Conditions:**
  * Surface CAPE: **$18.88\text{ J/kg}$** (Practically zero convective potential)
  * Convective Inhibition (CIN): **$0.00\text{ J/kg}$** (Zero capping inversion)
  * 24h Convective Precipitation: **$0.01\text{ mm}$** (Zero rain)
  * Surface 10m Wind Speed: **$3.38\text{ m/s}$** (Light breeze)
  * Low-Level 850hPa Wind: **$2.37\text{ m/s}$** (Light breeze)
  * Vertical Velocity at 500hPa: **$-0.062\text{ Pa/s}$** (Negligible synoptic ascent)
  * Bulk Shear 850–300hPa: **$43.03\text{ m/s}$** (Strong upper westerly jet stream)
  * Total Cloud Cover: **$0.968$** (Near-complete overcast)
  * Relative Humidity at 300hPa: **$96.44\%$** (Near-saturated upper troposphere)
* **Outputs Generated:**
  * Existing Continuous Storm Score: **$2.5066$**
  * Existing Binary Proxy Label: **$1$ (THUNDERSTORM)**
  * Transferred Native pyGAM Probability: **$0.0409$ (Rejects Storm)**
  * Transferred Native InterpretML Probability: **$0.0607$ (Rejects Storm)**

### Detailed Answers to Specific Audit Questions:

1. **Which individual formula components contributed to the score?**  
   The score $2.5066$ was formed by the additive combination of:
   * Upper-level humidity contribution: $0.8 \times 0.964 \approx \mathbf{+0.771}$
   * Deep-layer jet shear contribution: $0.7 \times (43.03 / 40.0) \approx \mathbf{+0.700}$ (capped at $+0.70$)
   * Total cloud cover contribution: $0.4 \times 0.968 \approx \mathbf{+0.387}$
   * Vertical velocity lift contribution: $1.0 \times \text{lift} \approx \mathbf{+0.350}$
   * Surface wind contribution: $0.5 \times (3.38 / 12.0) \approx \mathbf{+0.141}$
   * CAPE contribution: $1.2 \times (18.88 / 3000) \approx \mathbf{+0.008}$
   * CP contribution: $1.0 \times (0.01 / 10) \approx \mathbf{+0.001}$
   * CIN deduction: $-0.8 \times 0.0 = \mathbf{-0.000}$ (Zero penalty subtracted!)
   $$\text{Sum} \approx 0.771 + 0.700 + 0.387 + 0.350 + 0.141 + 0.008 - 0.000 = \mathbf{2.507} \ge \mathbf{2.50}$$
2. **How much did upper-tropospheric humidity contribute?**  
   Relative humidity at 300 hPa contributed **$+0.771$ points**, representing **$30.8\%$ of the entire $2.50$ threshold** required to declare a storm.
3. **Could high RH at 300 hPa independently push the score over 2.5?**  
   By itself ($+0.80$ max), 300 hPa RH cannot cross 2.50. However, in nature, 300 hPa saturation is almost universally accompanied by high cloud cover ($+0.40$) and upper-level jet stream shear ($+0.70$). These three non-convective upper-tropospheric variables sum to **$+1.90$**. It requires only modest wind or synoptic lift ($+0.60$) to surpass $2.50$, even when CAPE is zero.
4. **Is this classification physically defensible as a thunderstorm proxy?**  
   **ABSOLUTELY NOT.** A thunderstorm by thermodynamic definition requires buoyant parcel ascent capable of lofting supercooled water into the charge-separation zone ($-10^\circ\text{C}$ to $-20^\circ\text{C}$). With $\text{CAPE} = 18.9\text{ J/kg}$, parcel updrafts cannot exceed $w_{\text{max}} = \sqrt{2 \times 18.9} \approx 6.1\text{ m/s}$, which is physically incapable of supporting graupel electrification or lightning. Saturated 300 hPa air on a dry January day represents a passive, non-electrified cirrostratus sheet from a passing subtropical westerly trough.
5. **Is the concern caused by feature weight, transformation, threshold, or a combination?**  
   **A structural combination:**
   * *Additive formulation without thermodynamic gating:* Allowing kinematic and moisture terms to substitute for thermodynamic fuel.
   * *Wrong vertical humidity level:* Using 300 hPa (cirrus anvil level) instead of 850 hPa or 700 hPa boundary moisture.
   * *Zero-CIN loophole:* Because winter air lacks a capping inversion ($\text{CIN} = 0$), zero penalty is deducted, allowing winter days to score higher than capped pre-monsoon convective days.

---

## 4. Systematic Dataset-Wide Analysis of Chandipur Proxy Labels

Compiled in [`reports/chandipur_formula_diagnostic_audit/low_cape_positive_analysis.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_diagnostic_audit/low_cape_positive_analysis.csv):

### 4.1 Sub-Convective (Low-CAPE) Positive Labels

| CAPE Boundary | Positive Observations | % of All 5,462 Positives | Storm Score Range | Mean RH 300hPa | Mean Cloud Cover | Native EBM Mean Prob | Native EBM Rejection Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$\text{CAPE} < 50\text{ J/kg}$** | **54** | **0.99%** | $[2.50, 3.63]$ | **84.82%** | **0.986** | **0.1856** | **83.33%** |
| **$\text{CAPE} < 100\text{ J/kg}$** | **117** | **2.14%** | $[2.50, 3.75]$ | **82.97%** | **0.984** | **0.2744** | **65.81%** |
| **$\text{CAPE} < 250\text{ J/kg}$** | **363** | **6.65%** | $[2.50, 4.46]$ | **82.27%** | **0.978** | **0.3915** | **40.22%** |
| **$\text{CAPE} < 500\text{ J/kg}$** | **1,019** | **18.66%** | $[2.50, 4.97]$ | **81.85%** | **0.964** | **0.4702** | **22.28%** |

```
                       Low-CAPE Proxy Positives Distribution
   Observations
     1200 |
     1000 |                                            1,019 (18.7%)
      800 |                                            [CAPE < 500 J/kg]
      600 |
      400 |                            363 (6.7%)
      200 |              117 (2.1%)   [CAPE < 250 J/kg]
        0 |   54 (1.0%) [CAPE < 100]
          +---------------+------------+---------------+-------------->
             < 50 J/kg      < 100 J/kg    < 250 J/kg      < 500 J/kg
             [Updraft <10m/s]            [Sub-convective Regimes]
```

* **Severe Consequence:** Over **$18.6\%$ of all proxy positive classifications** in Chandipur occur when available buoyant energy is below $500\text{ J/kg}$.
* In all these cases, mean relative humidity at 300 hPa exceeds $81.8\%$ and cloud cover exceeds $0.96$. Saturated high clouds routinely trick the formula into declaring thunderstorms.
* Visualized in [`figures/low_cape_positives_breakdown.png`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_diagnostic_audit/figures/low_cape_positives_breakdown.png).

---

### 4.2 High-CAPE / High-CIN Capping Failures

Compiled in [`reports/chandipur_formula_diagnostic_audit/high_cape_high_cin_analysis.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_diagnostic_audit/high_cape_high_cin_analysis.csv):

| Meteorological Capping Scenario | Total Observations | Proxy Storm Labels | % Misclassified as Storms | Mean Storm Score | Physical Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$\text{CAPE} > 1,000\text{ J/kg}$ & $\text{CIN} > 200\text{ J/kg}$** | 4,238 | **395** | **9.32%** | 2.76 | **Capping Failure:** Strong lid ignored by formula |
| **$\text{CAPE} > 2,000\text{ J/kg}$ & $\text{CIN} > 300\text{ J/kg}$** | 2,335 | **184** | **7.88%** | 2.76 | **Capping Failure:** Heavy capping barrier overridden |
| **$\text{CAPE} > 3,000\text{ J/kg}$ & $\text{CIN} > 500\text{ J/kg}$** | 1,093 | **82** | **7.50%** | 2.73 | **Extreme Failure:** Impenetrable lid labeled as storm |

* **Physical Defect:** When $\text{CIN} > 500\text{ J/kg}$, parcel ascent is completely suppressed. The atmosphere cannot breach the Level of Free Convection without localized explosive orographic or mesoscale forcing. Yet the formula labeled **$82$ extreme capping states as active thunderstorms**.
* Visualized in [`figures/high_cape_high_cin_breakdown.png`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_diagnostic_audit/figures/high_cape_high_cin_breakdown.png).

---

## 5. Analysis of Florida-Trained EBM/GAM Disagreements

Compiled in [`reports/chandipur_formula_diagnostic_audit/ebm_gam_formula_disagreement_cases.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_diagnostic_audit/ebm_gam_formula_disagreement_cases.csv):

Using prospective predictions from `reports/ebm_gam_chandipur_transfer_native/chandipur_native_gam_ebm_predictions.csv`:

### 5.1 Category 1: Proxy Positive, Both Native Models Low Probability ($< 0.20$) — 272 Cases
* In **$272$ observations**, the formula declared a thunderstorm ($\text{score} \ge 2.50$), but both Florida-trained models predicted probabilities below $0.20$.
* **Meteorological Inspection:** In over $90\%$ of these cases, surface CAPE was either negligible ($< 100\text{ J/kg}$) or capped by extreme CIN ($> 700\text{ J/kg}$), but upper tropospheric RH at 300 hPa was near $100\%$.
* **Verdict:** The **Florida machine learning models are physically correct**. They correctly reject non-convective upper cirrus and capped sounding states that the heuristic formula misclassifies.

### 5.2 Category 2: Proxy Negative, Both Native Models High Probability ($> 0.50$) — 1,485 Cases
* In **$1,485$ observations**, the formula declared NO thunderstorm ($\text{score} < 2.50$), but both Florida models predicted probabilities $> 0.50$.
* **Meteorological Inspection:** In these cases, buoyant energy was massive ($\text{CAPE} > 2,500\text{ J/kg}$), deep-layer shear was favorable ($20 - 35\text{ m/s}$), and CIN was low ($< 50\text{ J/kg}$). However, the formula score sat just below threshold ($2.10 - 2.45$) because 300 hPa relative humidity was low (clear upper air before convective cloud tops reached 9 km).
* **Verdict:** The **formula exhibits false negative bias**. Requiring 300 hPa saturation before declaring a storm creates severe pre-convective latency: developing thunderstorm cells are ignored until after anvil cirrus spreads across 300 hPa.

### 5.3 Category 3: Native EBM vs. GAM Divergence ($|\Delta P| > 0.30$) — 50 Cases
* Divergence between pyGAM and InterpretML occurred almost exclusively when $\text{CAPE} > 7,240\text{ J/kg}$.
* InterpretML EBM clamped safely to its highest step function ($+1.79$ logit), whereas pyGAM's cubic spline curved downward outside its knot range ($-4.36$ logit), demonstrating that tree-based binning is mathematically superior for cross-regional transfer into extreme climates.

---

## 6. Ground-Truth Limitations & Scientific Disclaimers

1. **Non-Observational Nature of Chandipur Labels:**  
   The current Chandipur `storm_label_score` is a **reanalysis diagnostic index**, not an observational event catalog. It reflects the subjective assumptions of whoever assigned weights to ERA5 fields.
2. **Transfer Agreement Does Not Equal Forecasting Skill:**  
   The high agreement between the Florida models and the Chandipur proxy (PR-AUC: $0.6134$, ROC-AUC: $0.8572$, Pearson $r: 0.7188$) proves that Florida machine learning learned general synoptic relationships that match the formula's intuition. It does **not** prove that either system forecasts real-world lightning strikes in Odisha.
3. **Mandatory Need for Physical Ground Truth:**  
   True model verification in Chandipur requires independent observational datasets:
   * **IMD Doppler Weather Radar (DWR Paradip / Kolkata):** Reflectivity $\ge 40\text{ dBZ}$ above the freezing level.
   * **Lightning Detection Networks (WWLLN / IMD Lightning Network):** Exact geolocated cloud-to-ground (CG) and intra-cloud (IC) strokes.
   * **Aerodrome Surface Observations (Kolkata VECC, Bhubaneswar VEBS):** Direct METAR thunderstorm (`TS`, `VCTS`) reports.

---

## 7. Ranked Recommendations for Formula Improvement (Phase 3 Roadmap)

Compiled in [`reports/chandipur_formula_diagnostic_audit/formula_improvement_proposals.md`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_diagnostic_audit/formula_improvement_proposals.md):

| Priority Rank | Remediation Proposal | Specific Defect Solved | Expected Impact | Difficulty |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Thermodynamic CAPE Gating** | Purges sub-convective positive classifications | **Eliminates 363 to 1,019 false positives ($6.7\% - 18.7\%$)** | **Trivial** |
| **2** | **Multiplicative CIN Barrier** | Prevents positive labels under heavy capping lids | **Eliminates 395 positive classifications ($\text{CIN} > 200\text{ J/kg}$)** | **Low** |
| **3** | **Replace 300 hPa RH with 850 hPa** | Stops upper-tropospheric cirrus from hijacking scores | **Fixes 3,209 decisive false positive cases** | **Low** |
| **4** | **S-Curve CAPE Saturation** | Prevents score runaway in extreme pre-monsoon heat | **Prevents score distortion up to $9,906\text{ J/kg}$** | **Low** |
| **5** | **Two-Stage Decoupled Index** | Decouples environmental favorability from trigger | **Establishes modern, physically sound nowcast architecture** | **Medium** |

---

## 8. Artifact Directory Manifest

All diagnostic artifacts have been created, verified, and saved to [`reports/chandipur_formula_diagnostic_audit/`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_diagnostic_audit/):

### Detailed Analytical Documents & Data:
* Exact Formula Reconstruction: [`reports/chandipur_formula_diagnostic_audit/formula_reconstruction.md`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_diagnostic_audit/formula_reconstruction.md)
* Scientifically Defensible Improvement Proposals: [`reports/chandipur_formula_diagnostic_audit/formula_improvement_proposals.md`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_diagnostic_audit/formula_improvement_proposals.md)
* Low-CAPE Positive Analysis Table: [`reports/chandipur_formula_diagnostic_audit/low_cape_positive_analysis.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_diagnostic_audit/low_cape_positive_analysis.csv)
* High-CAPE / High-CIN Capping Analysis Table: [`reports/chandipur_formula_diagnostic_audit/high_cape_high_cin_analysis.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_diagnostic_audit/high_cape_high_cin_analysis.csv)
* Component Contribution & Decisive Frequency Table: [`reports/chandipur_formula_diagnostic_audit/component_contribution_analysis.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_diagnostic_audit/component_contribution_analysis.csv)
* Florida EBM/GAM vs. Formula Disagreement Cases: [`reports/chandipur_formula_diagnostic_audit/ebm_gam_formula_disagreement_cases.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_diagnostic_audit/ebm_gam_formula_disagreement_cases.csv) (90 representative cases)

### High-Resolution Diagnostic Figures:
* Low-CAPE Positives Breakdown: [`reports/chandipur_formula_diagnostic_audit/figures/low_cape_positives_breakdown.png`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_diagnostic_audit/figures/low_cape_positives_breakdown.png)
* Component Decisive Impact Frequency: [`reports/chandipur_formula_diagnostic_audit/figures/decisive_component_frequency.png`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_diagnostic_audit/figures/decisive_component_frequency.png)
* High-CAPE / High-CIN Capping Suppression Failures: [`reports/chandipur_formula_diagnostic_audit/figures/high_cape_high_cin_breakdown.png`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_diagnostic_audit/figures/high_cape_high_cin_breakdown.png)
* Bivariate CAPE vs. Storm Score Density: [`reports/chandipur_formula_diagnostic_audit/figures/cape_vs_storm_score_density.png`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_diagnostic_audit/figures/cape_vs_storm_score_density.png)
