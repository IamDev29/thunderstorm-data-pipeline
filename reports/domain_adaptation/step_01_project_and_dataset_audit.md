# Comprehensive Project & Dataset Audit: Thunderstorm Domain Adaptation Framework (TDAF)

**Audit Phase:** Step 01 — Project Reconnaissance, Dataset Verification, and Baseline Audit  
**Date:** November 17, 2026  
**Auditor:** Thunderstorm Domain Adaptation Framework (TDAF) Independent Engine  
**Status:** COMPLETE — AUDIT ONLY (NO MODEL TRAINING OR DATA MODIFICATIONS APPLIED)  

---

## 1. Executive Summary

This report establishes the empirical, architectural, and meteorological baseline for the **Thunderstorm Domain Adaptation Framework (TDAF)**. TDAF investigates whether statistical domain adaptation, conventional transfer learning, and deep representation alignment can bridge the covariate and concept shift between our **Source Domain** (Florida, USA — ASOS-grounded thunderstorm labels) and our **Target Domain** (Chandipur, Odisha, India — ERA5 atmospheric observations and proxy classifications).

```
                               TDAF Dual-Domain Architectural Map
      ===================================================================================
      SOURCE DOMAIN (Florida, USA)                    TARGET DOMAIN (Chandipur, India)
      -----------------------------------------       -----------------------------------
      Dataset: florida_asos_modeling_ready_2024.parquet Dataset: MET_DATA_with_storm_score_label.csv
      Coverage: Jan 1 - Dec 31, 2024 (112,509 obs)    Coverage: Jan 1, 2022 - Dec 31, 2025 (23,376 obs)
      Target: ASOS Strict Criterion C (Observed TS)    Target: Heuristic Storm Score (Proxy >= 2.50)
      Models: Native pyGAM & InterpretML EBM          Zero-shot baseline: EBM PR-AUC=0.613, ROC=0.857
      ===================================================================================
                                                  |
                          +-----------------------+-----------------------+
                          |                                               |
                          v                                               v
            +---------------------------+                   +---------------------------+
            |  Thermodynamic Shift      |                   |  Kinematic / Shear Shift  |
            |  - CH Mean CAPE: 1,325    |                   |  - FL Mean Shear: 24.9 m/s|
            |  - FL Mean CAPE:   603    |                   |  - CH Mean Shear: 17.7 m/s|
            |  - 81 CH samples > FL Max |                   |  - Tropical monsoon lacks |
            |  - Wasserstein d = 716.9  |                   |    mid-latitude jet shear |
            +---------------------------+                   +---------------------------+
```

### Core Audit Findings:
1. **Source Dataset Integrity:** Florida modeling parquet contains **$112,509$ valid monitored observations** across 2024 ($66,255$ Train, $18,181$ Validation, $28,073$ Test). The target label is grounded in rigorously audited ASOS aerodrome observations (`target_asos_strict_crit_c`).
2. **Target Dataset Integrity:** Chandipur dataset comprises **$23,376$ synoptic observations** (6-hour intervals across 4 grid points enclosing the Integrated Test Range, 2022–2025). The label is a **$100\%$ deterministic, coincident mathematical heuristic** (`storm_label_score`), not observational ground truth.
3. **Severe Covariate Shift in Thermodynamic Drivers:**
   * **CAPE:** Chandipur mean ($1,324.8\text{ J/kg}$) is **$+119.6\%$ higher** than Florida ($603.2\text{ J/kg}$); median is **21 times higher** ($763.4\text{ J/kg}$ vs. $35.2\text{ J/kg}$). Furthermore, $81$ Chandipur samples exceed Florida's absolute maximum training CAPE ($7,240\text{ J/kg}$), reaching up to $9,906\text{ J/kg}$.
   * **CIN:** Chandipur mean ($155.4\text{ J/kg}$) is **$+78.0\%$ higher** than Florida ($87.3\text{ J/kg}$), reflecting intense pre-monsoon dryline and inversion lids.
4. **Opposing Kinematic Regimes:** Deep-layer shear ($850-300\text{ hPa}$) is **$-29.0\%$ lower** in Chandipur ($17.67\text{ m/s}$ vs. $24.89\text{ m/s}$ in Florida) due to the absence of persistent mid-latitude westerly jet dynamics.
5. **Readiness for Step 2:** All code, datasets, serialized models, and zero-shot baseline artifacts are fully preserved, verified, and ready for domain adaptation.

---

## 2. Project Directory Structure

The project environment is partitioned into operational and subproject directories:

* **Primary Project Root:** `C:\Users\KIIT0001\Desktop\Data Collection`
* **Thunderstorm Pipeline Directory:** [`Thunderstorm_pipeline/`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/Thunderstorm_pipeline/) (and mirrored at `C:\Users\KIIT0001\Desktop\Thunderstorm_pipeline`)
* **Experimental Directory (Formula V2):** [`Thunderstorm_pipeline/experimental_formula_v2/`](file:///c:/Users/KIIT0001/Desktop/Thunderstorm_pipeline/experimental_formula_v2/)
* **Centralized Image Collection:** [`Thunderstorm_pipeline/collected_project_images/`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/Thunderstorm_pipeline/collected_project_images/)
* **Domain Adaptation Experiment Root:** `reports/domain_adaptation/`
* **Florida Training Subproject Mirror:** `Florida_Thunderstorm_2024_Training/`

### Directory Inventory:
```
Data Collection/
├── config/                                # Training and ingestion configurations
├── data/
│   ├── processed/asos_parser_fixed/      # Canonical Florida 2024 modeling parquet
│   ├── raw/                               # Raw netCDFs and ASOS METAR text archives
│   └── reports/                           # Climate similarity & geographical maps
├── models/                                # Saved LightGBM, XGBoost, and Logistic Regression models
├── reports/
│   ├── ebm_gam_florida_native/            # Native pyGAM & InterpretML EBM models & metrics
│   ├── ebm_gam_chandipur_transfer_native/ # Zero-shot Florida-to-Chandipur transfer baseline
│   ├── chandipur_formula_diagnostic_audit/# Structural heuristic diagnostic report & figures
│   ├── chandipur_formula_experiment_v2/   # Controlled formula improvements V2 (26 variants)
│   └── domain_adaptation/                 # TDAF audit reports and figures (NEW)
├── scripts/
│   ├── train_and_evaluate_native_ebm_gam_florida.py # Native Florida model training
│   ├── transfer_native_ebm_gam_chandipur.py         # Zero-shot transfer implementation
│   ├── audit_chandipur_storm_score_formula.py      # Diagnostic audit engine
│   └── audit_generate_domain_adaptation_figures.py # Step 01 figure generator
├── Thunderstorm_pipeline/
│   ├── experimental_formula_v2/          # Modular formula definitions & test suite
│   ├── collected_project_images/         # 50 centralized project figures & inventory CSV
│   └── collect_images.py                 # Reusable image collection script
└── MET_DATA_with_storm_score_label.csv   # Primary Chandipur 2022-2025 dataset
```

---

## 3. Dataset Inventory & Verification

All datasets were loaded, verified, and audited without modification:

| Metric / Attribute | Florida Training Set | Florida Validation Set | Florida Locked Holdout Test Set | Florida Monitored Full Year | Chandipur Multi-Year Dataset |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **File Path** | `data/processed/asos_parser_fixed/...` | `data/processed/asos_parser_fixed/...` | `data/processed/asos_parser_fixed/...` | `data/processed/asos_parser_fixed/...` | `MET_DATA_with_storm_score_label.csv` |
| **File Format** | Apache Parquet (`.parquet`) | Apache Parquet (`.parquet`) | Apache Parquet (`.parquet`) | Apache Parquet (`.parquet`) | Comma-Separated Values (`.csv`) |
| **Row Count** | **66,255** | **18,181** | **28,073** | **112,509** (Valid target rows) | **23,376** |
| **Column Count** | 53 columns | 53 columns | 53 columns | 53 columns (119,663 raw total) | 25 columns |
| **Temporal Coverage** | Jan 1, 2024 – Jul 31, 2024 | Aug 1, 2024 – Sep 30, 2024 | Oct 1, 2024 – Dec 31, 2024 | Jan 1, 2024 – Dec 31, 2024 | Jan 1, 2022 – Dec 31, 2025 |
| **Temporal Frequency**| 6-hourly synoptic | 6-hourly synoptic | 6-hourly synoptic | 6-hourly synoptic | 6-hourly synoptic (00,06,12,18 UTC) |
| **Spatial Domain** | Florida ASOS stations ($N=54$) | Florida ASOS stations ($N=54$) | Florida ASOS stations ($N=54$) | Florida ASOS stations ($N=54$) | Chandipur ITR 4 grid cells |
| **Target Column** | `target_asos_strict_crit_c` | `target_asos_strict_crit_c` | `target_asos_strict_crit_c` | `target_asos_strict_crit_c` | `storm_label_score` |
| **Target Nature** | Observed ASOS METAR | Observed ASOS METAR | Observed ASOS METAR | Observed ASOS METAR | Heuristic composite ($\ge 2.50$) |
| **Positive Labels** | **13,228** | **7,331** | **1,918** | **22,477** | **5,462** |
| **Negative Labels** | **53,027** | **10,850** | **26,155** | **90,032** | **17,914** |
| **Prevalence (%)** | **19.9653%** | **40.3223%** | **6.8322%** | **19.9779%** | **23.3658%** |
| **Missing Values (6 Feats)** | **0 (0.00%)** | **0 (0.00%)** | **0 (0.00%)** | **0 (0.00%)** | **0 (0.00%)** |
| **Duplicate Observations** | **0** | **0** | **0** | **0** (on `[station, init_time]`) | **0** (on `[time, lat, lon]`) |

*Note: In the Florida raw parquet file ($119,663$ total rows), $7,154$ rows have null target values due to station reporting gaps during hurricane/outage events (`target_crit_c == 'unmonitored'`). All modeling scripts strictly filter to the $112,509$ valid monitored observations.*

---

## 4. Feature Compatibility Audit

The six transfer-compatible features were systematically audited across both repositories:

| Canonical Transfer Feature | Florida Column Identifier | Chandipur Column Identifier | Physical Units | Physical Definition & Level | Missing Value Handling | Extraction Alignment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$X_1$: Surface CAPE** | `cape_J_kg` | `cape_Jkg` | $\text{J/kg}$ | Most Unstable Parcel Convective Available Potential Energy (Surface) | Zero nulls; lower-bounded at $0.0$ | **Identical ECMWF Parameter.** Extreme tropical extrapolation in Chandipur. |
| **$X_2$: Boundary CIN** | `cin_J_kg` | `cin_Jkg` | $\text{J/kg}$ | Convective Inhibition for lifted parcel (Boundary layer) | Zero nulls; non-negative clamping ($\ge 0$) | **Identical ECMWF Parameter.** Higher mean in Chandipur pre-monsoon. |
| **$X_3$: Deep Shear** | `bulk_shear_850_300_ms` | `bulk_shear_850_300` | $\text{m/s}$ | Vector wind difference magnitude: $\sqrt{(u_{300}-u_{850})^2 + (v_{300}-v_{850})^2}$ | Zero nulls; physical range $[0.05, 78]$ | **Identical Vector Formulation.** Significantly weaker in tropical Chandipur. |
| **$X_4$: Cloud Cover** | `cloud_cover_fraction` | `total_cloud_cover` | Fraction $[0, 1]$ | Total Atmospheric Column Cloud Fraction | Zero nulls; strictly bounded $[0.0, 1.0]$ | **Identical ECMWF Parameter.** High bimodal saturation in both domains. |
| **$X_5$: 10m Wind Speed** | `wind_speed_10m_ms` | `wind_speed_10m` | $\text{m/s}$ | Surface 10-meter scalar wind velocity: $\sqrt{u_{10}^2 + v_{10}^2}$ | Zero nulls; physical range $[0.02, 24]$ | **Identical 10m Velocity Formulation.** Higher in coastal Chandipur. |
| **$X_6$: 850hPa Wind Speed**| `wind_speed_850hPa_ms` | `wind_speed_850hPa` | $\text{m/s}$ | Low-tropospheric scalar wind velocity: $\sqrt{u_{850}^2 + v_{850}^2}$ | Zero nulls; physical range $[0.02, 45]$ | **Identical 850hPa Formulation.** Higher in Florida winter jets. |

### Feature Incompatibility Caveats:
* **Excluded Moisture Dimension:** Florida utilizes boundary-layer moisture (`relative_humidity_850hPa_pct`, `dewpoint_depression_degC`, `tcwv_kg_m2`), whereas the standard Chandipur CSV (`MET_DATA_with_storm_score_label.csv`) preserved only upper-tropospheric moisture (`relative_humidity_300hPa`). In Step 1, moisture was explicitly excluded from the 6-feature manifold to prevent unphysical 300 hPa vs. 850 hPa misalignment.

---

## 5. Existing Model Audit: Native Florida pyGAM & InterpretML EBM

The serialized models were inspected from [`reports/ebm_gam_florida_native/`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/ebm_gam_florida_native/):

| Model Attribute | Native pyGAM Model (`native_gam_model.pkl`) | Native InterpretML EBM Model (`native_ebm_model.pkl`) |
| :--- | :--- | :--- |
| **Library & Implementation** | `pygam` (v0.12.0) | `interpret.glassbox` (v0.7.8) |
| **Model Class** | `pygam.LogisticGAM` | `ExplainableBoostingClassifier` |
| **Model Architecture** | Spline-based Generalized Additive Model: $\text{logit}(P) = \beta_0 + \sum_{j=1}^6 s_j(X_j)$ | Bagged shallow decision trees (additive stumps): $\text{logit}(P) = \beta_0 + \sum_{j=1}^6 f_j(X_j)$ |
| **Hyperparameters** | `n_splines = 15`, smoothing penalty $\lambda = 1.0$ per term | `learning_rate = 0.01`, `max_bins = 128`, `outer_bags = 16`, `interactions = 0` |
| **Input Features** | 6 compatible features (unscaled raw float array) | 6 compatible features (native feature binning) |
| **Training Dataset** | Florida Train: Jan 1 – Jul 31, 2024 ($N = 66,255$) | Florida Train: Jan 1 – Jul 31, 2024 ($N = 66,255$) |
| **Validation Dataset** | Florida Val: Aug 1 – Sep 30, 2024 ($N = 18,181$) | Florida Val: Aug 1 – Sep 30, 2024 ($N = 18,181$) |
| **Locked Holdout Test** | Florida Test: Oct 1 – Dec 31, 2024 ($N = 28,073$) | Florida Test: Oct 1 – Dec 31, 2024 ($N = 28,073$) |
| **Optimal Threshold** | **$\tau = 0.3300$** (tuned on Florida Validation F1) | **$\tau = 0.3400$** (tuned on Florida Validation F1) |
| **Florida Test PR-AUC** | **0.3362** (vs. baseline prevalence 0.0683) | **0.3513** (vs. baseline prevalence 0.0683) |
| **Florida Test ROC-AUC** | **0.8288** | **0.8304** |
| **Florida Test Brier** | **0.0556** | **0.0564** |
| **Florida Test F1** | **0.4042** (Precision: 0.3236, Recall: 0.5375) | **0.4040** (Precision: 0.3371, Recall: 0.5042) |
| **Saved Model Path** | [`reports/ebm_gam_florida_native/native_gam_model.pkl`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/ebm_gam_florida_native/native_gam_model.pkl) | [`reports/ebm_gam_florida_native/native_ebm_model.pkl`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/ebm_gam_florida_native/native_ebm_model.pkl) |
| **Model Size on Disk** | 71,883 bytes | 129,835 bytes |

*Both models are confirmed to be genuine, library-native implementations, not the earlier SplineTransformer/LightGBM-stump approximations.*

---

## 6. Zero-Shot Experiment Reconstruction & Baseline Reference

Reconstructed from [`scripts/transfer_native_ebm_gam_chandipur.py`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/scripts/transfer_native_ebm_gam_chandipur.py) and [`reports/ebm_gam_chandipur_transfer_native_report.md`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/ebm_gam_chandipur_transfer_native_report.md):

* **Exact Models Loaded:** `native_gam_model.pkl` and `native_ebm_model.pkl` directly from `reports/ebm_gam_florida_native/`.
* **Chandipur Input Matrix:** `df_ch[['cape_Jkg', 'cin_Jkg', 'bulk_shear_850_300', 'total_cloud_cover', 'wind_speed_10m', 'wind_speed_850hPa']].values` ($23,376 \times 6$).
* **Target Domain Adaptation Applied:** **ZERO.** No fine-tuning, no transfer recalibration, no target-domain fitting, and no domain alignment occurred.
* **Normalization / Scaling Applied:** None. Raw ERA5 physical units were fed directly into the models, relying on the native spline and tree split boundaries established in Florida.
* **Decision Thresholds Applied:** Frozen Florida thresholds ($\tau_{\text{GAM}} = 0.33, \tau_{\text{EBM}} = 0.34$).
* **Target Labels Used:** Chandipur heuristic proxy `storm_label_score` ($\text{Storm Score} \ge 2.50$).

### Established Zero-Shot Performance Baseline:
* **InterpretML EBM:**
  * **PR-AUC vs. Proxy:** **0.6134** (Prevalence baseline: 0.2337)
  * **ROC-AUC vs. Proxy:** **0.8572**
  * **Brier Score:** **0.1270**
  * **F1 Score ($\tau=0.34$):** **0.6059** (Precision: 0.4907, Recall: 0.7917, Accuracy: 75.89%)
  * **Correlation with Storm Score:** Pearson $r = \mathbf{0.7188}$, Spearman $\rho = \mathbf{0.7546}$
* **pyGAM LogisticGAM:**
  * **PR-AUC vs. Proxy:** **0.5523**
  * **ROC-AUC vs. Proxy:** **0.8494**
  * **Brier Score:** **0.1301**
  * **F1 Score ($\tau=0.33$):** **0.6147** (Precision: 0.4851, Recall: 0.8385, Accuracy: 75.49%)
  * **Correlation with Storm Score:** Pearson $r = \mathbf{0.7050}$, Spearman $\rho = \mathbf{0.7560}$

*This zero-shot benchmark serves as the strict reference point for all subsequent TDAF domain adaptation methods.*

---

## 7. Quantitative Domain Shift Analysis

### 7.1 Statistical Moments and Quantiles

Computed across Florida Training ($N = 66,255$), Florida Full 2024 ($N = 112,509$), and Chandipur Target ($N = 23,376$):

| Atmospheric Feature | Domain Dataset | Mean ($\mu$) | Std ($\sigma$) | Min | 25th % | Median | 75th % | 95th % | 99th % | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`cape_J_kg`** | **Florida Train** | 603.23 | 945.63 | 0.00 | 0.00 | 35.20 | 980.00 | 2,668.50 | 3,744.28 | 7,240.00 |
| | **Florida All** | 607.98 | 958.77 | 0.00 | 0.00 | 44.00 | 956.20 | 2,709.68 | 3,827.48 | 7,240.00 |
| | **Chandipur** | **1,324.85** | **1,561.00** | 0.00 | 14.47 | **763.38** | 2,102.84 | **4,611.56** | **6,386.81** | **9,906.00** |
| **`cin_J_kg`** | **Florida Train** | 87.29 | 166.80 | 0.00 | 0.00 | 0.00 | 108.70 | 460.63 | 793.95 | 997.40 |
| | **Florida All** | 77.49 | 149.80 | 0.00 | 0.00 | 0.00 | 97.30 | 394.10 | 740.39 | 997.40 |
| | **Chandipur** | **155.41** | **227.24** | 0.00 | 0.00 | **56.29** | 208.82 | **711.09** | **895.84** | **999.17** |
| **`bulk_shear_850_300_ms`** | **Florida Train** | 24.89 | 15.01 | 0.05 | 12.26 | 23.66 | 36.15 | 50.98 | 61.10 | 77.78 |
| | **Florida All** | 22.81 | 14.01 | 0.05 | 11.32 | 20.54 | 32.99 | 47.93 | 58.39 | 77.78 |
| | **Chandipur** | **17.67** | **10.47** | 0.11 | 9.36 | **16.02** | 24.56 | 37.47 | 45.35 | 56.45 |
| **`cloud_cover_fraction`** | **Florida Train** | 0.5626 | 0.3889 | 0.00 | 0.1587 | 0.6217 | 0.9706 | 1.0000 | 1.0000 | 1.0000 |
| | **Florida All** | 0.5394 | 0.3871 | 0.00 | 0.1413 | 0.5579 | 0.9600 | 1.0000 | 1.0000 | 1.0000 |
| | **Chandipur** | 0.4974 | 0.4105 | 0.00 | 0.0520 | 0.4456 | 0.9738 | 1.0000 | 1.0000 | 1.0000 |
| **`wind_speed_10m_ms`** | **Florida Train** | 3.05 | 1.63 | 0.02 | 1.87 | 2.76 | 3.94 | 6.11 | 8.12 | 15.07 |
| | **Florida All** | 3.11 | 1.69 | 0.02 | 1.92 | 2.80 | 3.96 | 6.23 | 8.45 | 23.70 |
| | **Chandipur** | **4.26** | **2.15** | 0.05 | 2.61 | **4.06** | 5.64 | 8.19 | 9.77 | 16.60 |
| **`wind_speed_850hPa_ms`** | **Florida Train** | 7.05 | 4.98 | 0.02 | 3.37 | 5.74 | 9.64 | 16.71 | 22.64 | 42.33 |
| | **Florida All** | 6.95 | 4.84 | 0.02 | 3.50 | 5.76 | 9.32 | 16.11 | 23.18 | 44.98 |
| | **Chandipur** | 5.67 | 3.08 | 0.05 | 3.51 | 5.25 | 7.28 | 11.24 | 15.07 | 28.99 |

---

### 7.2 Distribution Divergence & Statistical Distance

| Feature | Two-Sample Kolmogorov-Smirnov ($KS$) | $p$-value | Wasserstein Distance ($\mathcal{W}_1$) | Divergence Severity |
| :--- | :--- | :--- | :--- | :--- |
| **`cape_J_kg`** | **0.2596** | $< 10^{-300}$ | **$716.86\text{ J/kg}$** | **EXTREME COVARIATE SHIFT** |
| **`wind_speed_10m_ms`** | **0.2699** | $< 10^{-300}$ | **$1.15\text{ m/s}$** | **HIGH SHIFT (Marine vs Inland)** |
| **`cin_J_kg`** | **0.1954** | $< 10^{-300}$ | **$77.92\text{ J/kg}$** | **HIGH SHIFT (Pre-monsoon Caps)** |
| **`bulk_shear_850_300_ms`** | **0.1724** | $< 10^{-300}$ | **$5.13\text{ m/s}$** | **MODERATE SHIFT (Mid-latitude Jet)** |
| **`wind_speed_850hPa_ms`** | **0.1383** | $< 10^{-300}$ | **$1.32\text{ m/s}$** | **MODERATE SHIFT** |
| **`cloud_cover_fraction`** | **0.0818** | $4.8 \times 10^{-113}$ | **$0.0462$** | **LOW SHIFT (Shared Bimodal Form)** |

---

### 7.3 Inter-Feature Correlation Differences

Examining the correlation shift matrix ($\Delta r = r_{\text{Chandipur}} - r_{\text{Florida}}$):
* **CAPE vs. 10m Wind Speed:** $\Delta r = \mathbf{+0.470}$ ($r_{\text{FL}} = -0.063$ vs. $r_{\text{CH}} = +0.407$). In Chandipur, coastal sea breezes and pre-monsoon southerlies advect high-moisture air, coupling wind speed with convective buoyancy. In Florida, peak daytime CAPE occurs under calm, inland sea-breeze convergence.
* **CAPE vs. Bulk Shear:** $\Delta r = \mathbf{+0.257}$ ($r_{\text{FL}} = -0.366$ vs. $r_{\text{CH}} = -0.109$).
* **Shear vs. 850hPa Wind:** $\Delta r = \mathbf{-0.300}$ ($r_{\text{FL}} = +0.575$ vs. $r_{\text{CH}} = +0.275$). In Florida, strong 850hPa winds represent low-level jet advection tied to synoptic trough shear; in Chandipur, 850hPa winds are governed by the tropical monsoon circulation.

---

## 8. Diagnostic Visualizations

The four required audit figures were generated and saved in [`reports/domain_adaptation/figures/step_01/`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/domain_adaptation/figures/step_01/):

1. **Marginal Feature Distributions (KDE & Medians):**  
   [`reports/domain_adaptation/figures/step_01/source_vs_target_feature_distributions.png`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/domain_adaptation/figures/step_01/source_vs_target_feature_distributions.png)  
   *Visualizes the severe rightward shift of CAPE, the elevated median of CIN, and the compressed shear range in Chandipur.*
2. **Correlation Heatmaps & Covariate Shift ($\Delta r$):**  
   [`reports/domain_adaptation/figures/step_01/feature_correlation_heatmaps.png`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/domain_adaptation/figures/step_01/feature_correlation_heatmaps.png)  
   *Highlights the decoupling of shear and 850hPa winds alongside the strong positive coupling between surface wind and CAPE in the Bay of Bengal.*
3. **Joint CAPE-CIN Covariate Structure & Extrapolation Frontier:**  
   [`reports/domain_adaptation/figures/step_01/cape_cin_distribution_comparison.png`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/domain_adaptation/figures/step_01/cape_cin_distribution_comparison.png)  
   *Demonstrates the 81 extreme extrapolation cases beyond $7,240\text{ J/kg}$ and the cluster of heavy pre-monsoon capping lids ($> 500\text{ J/kg}$).*
4. **Seasonal Feature Divergence Boxplots:**  
   [`reports/domain_adaptation/figures/step_01/seasonal_feature_distribution_comparison.png`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/domain_adaptation/figures/step_01/seasonal_feature_distribution_comparison.png)  
   *Compares quarterly synoptic variations across Winter (DJF), Pre-monsoon (MAM), Monsoon (JJAS), and Post-monsoon (ON).*

---

## 9. Identified Data-Quality & Modeling Issues

1. **Spline Extrapolation Vulnerability in pyGAM:**  
   While `InterpretML` EBM clamps decision stump outputs above $7,240\text{ J/kg}$ (plateauing at $+1.79$ logit), `pyGAM` splines lack natural boundary constraints, turning sharply downward to $-4.36$ logit under extreme tropical CAPE ($> 7,500\text{ J/kg}$). Any domain adaptation method utilizing GAM architectures must enforce monotonicity or spline clamping.
2. **Heuristic Nature of Chandipur Target:**  
   The target `storm_label_score` is not observational truth. Domain adaptation cannot be treated as standard supervised transfer; instead, adaptation must target **unsupervised feature alignment** or **semi-supervised adaptation** with physical boundary constraints.
3. **Unmonitored Florida Rows:**  
   The $7,154$ unmonitored rows in the Florida dataset must remain strictly excluded from source domain estimation to avoid target contamination.
4. **Spatial Autocorrelation in Chandipur:**  
   The 4 Chandipur grid points are separated by only $0.25^\circ$ (~25 km). Temporal splits or grouped spatial cross-validation must be utilized during adaptation to avoid auto-correlation leakage across the quad-cell cluster.

---

## 10. Potential Leakage Risks & Governance Rules

To preserve rigorous scientific validity during subsequent TDAF steps:
1. **No Target Test-Period Leakage:** If Chandipur is split temporally (e.g., 2022–2024 for adaptation alignment and 2025 for holdout evaluation), **2025 data must remain completely unobserved** during feature whitening, covariance alignment, or unsupervised representation learning.
2. **No Proxy Label Leakage into Unsupervised Adaptation:** Unsupervised domain adaptation methods (e.g., CORAL, MMD, DANN) must receive **strictly unlabeled** Chandipur feature matrices $X_{\text{target}}$.
3. **No Retrospective Leakage:** Adaptation statistics (means, standard deviations, covariance matrices) must be computed chronologically.

---

## 11. Recommendations for Step 2 Experiments

Based on the verified data properties, Step 2 should implement and evaluate a hierarchy of domain adaptation strategies:

1. **Baseline Reference:** Preserve the unadapted zero-shot EBM (PR-AUC: 0.6134, ROC-AUC: 0.8572) as the non-adaptation benchmark.
2. **Statistical Covariate Alignment (Unsupervised):**
   * **CORAL (Correlation Alignment):** Align second-order statistics (covariance matrices) of Florida and Chandipur features without requiring target labels.
   * **Domain-Specific Whitening (ZCA/PCA):** Standardize marginal variances and decouple cross-feature correlations.
   * **Wasserstein / MMD Optimal Transport:** Re-weight source samples to minimize the distribution distance to the Chandipur manifold.
3. **Thermodynamic Bound Handling & Non-linear Saturation:**
   * Feature-level logistic clamping on CAPE above $5,000\text{ J/kg}$ to prevent spline collapse and stump extrapolation spikes.
4. **Unsupervised Domain-Adversarial Neural Networks (DANN):**
   * Train a domain discriminator against a feature extractor to learn representations invariant to the Florida-vs-Chandipur geographic domain while maintaining classification capacity on Florida ASOS labels.
5. **Threshold and Calibration Adaptation:**
   * Apply temperature scaling or Platt recalibration on unsupervised target pseudo-labels or physical convective bounds.
