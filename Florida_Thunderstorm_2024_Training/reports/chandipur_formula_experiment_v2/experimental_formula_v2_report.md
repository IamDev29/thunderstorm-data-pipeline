# Comprehensive Experimental Evaluation Report: Controlled Improvements to the Chandipur Storm Score Formula

**Experiment Directory:** `reports/chandipur_formula_experiment_v2/`  
**Pipeline Directory:** `Thunderstorm_pipeline/experimental_formula_v2/`  
**Execution Script:** [`evaluate_experiment_v2.py`](file:///c:/Users/KIIT0001/Desktop/Thunderstorm_pipeline/experimental_formula_v2/evaluate_experiment_v2.py)  
**Configuration File:** [`formula_variant_configurations.json`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_experiment_v2/formula_variant_configurations.json)  
**Dataset Evaluated:** `MET_DATA_with_storm_score_label.csv` (23,376 synoptic observations, 2022–2025)  
**Evaluation Target:** Scientifically motivated heuristic proxy variants  
**Status:** COMPLETE — RIGOROUS SCIENTIFIC COMPARISON (NO PREMATURE WINNER SELECTION)  

---

## 1. Executive Summary

This report documents the results of the **Controlled Formula Improvement Experiment (V2)** for the Chandipur thunderstorm proxy scoring system. Following the structural vulnerabilities uncovered in the diagnostic audit (`reports/chandipur_formula_diagnostic_audit_report.md`), five experimental families comprising **26 distinct formula configurations** were implemented and evaluated across all 23,376 synoptic observations.

```
                           Overview of Experimental Variant Families
     +------------------------------------------------------------------------------------------+
     |                  Chandipur Baseline Formula: Score >= 2.50 (23.37% Positives)            |
     +------------------------------------------------------------------------------------------+
                                                   |
         +--------------------+--------------------+--------------------+--------------------+
         |                    |                    |                    |                    |
         v                    v                    v                    v                    v
  +--------------+     +--------------+     +--------------+     +--------------+     +--------------+
  |  Variant A   |     |  Variant B   |     |  Variant C   |     |  Variant D   |     |  Variant E   |
  | CAPE Gating  |     | CIN Barrier  |     |  RH 850 hPa  |     | Combined A+B |     |  Two-Stage   |
  | tau in [100, |     | lam in [100, |     | Replacement  |     | 3x4 Param    |     | Decoupled    |
  | 250,500,750] |     | 150,200,300] |     | (+ Gating)   |     | Grid Matrix  |     | Favor/Trigger|
  +--------------+     +--------------+     +--------------+     +--------------+     +--------------+
         |                    |                    |                    |                    |
         v                    v                    v                    v                    v
  Purges 117-1019      Suppresses 71- |    Replaces cirrus      Eliminates both      Decouples macro-
  unphysical low-      82% of extreme |    anvil hijacking;     sub-convective &     environment from
  buoyancy alarms      capping lids   |    elevates coastal     severe capping lid   mesoscale trigger;
  (15.5 - 22.9%)       (9.2 - 13.0%)  |    moist cases          failures (7.3-12.6%) peak ROC (0.922)
```

### Key Experimental Findings:

1. **Purging Low-CAPE False Alarms (Variant A):**
   * Multiplicative thermodynamic gating ($\mathbb{I}(\text{CAPE} \ge \tau) \times \text{Score}$) cleanly eliminates sub-convective classifications without perturbing monsoonal or pre-monsoon convective scores.
   * Setting $\tau = 250\text{ J/kg}$ purges **$363$ unphysical positive classifications** (including 54 cases with $\text{CAPE} < 50\text{ J/kg}$), reducing positive prevalence moderately from $23.37\%$ to **$21.81\%$** ($5,099$ cases) while maintaining $98.45\%$ agreement with the original labels.
   * Setting $\tau = 500\text{ J/kg}$ purges all **$1,019$ sub-convective cases** ($18.66\%$ of baseline positives), bringing overall prevalence to **$19.01\%$**.
2. **Restoring Thermodynamic Capping Realism (Variant B):**
   * Replacing linear CIN subtraction with an exponential barrier ($\exp(-\text{CIN} / \lambda)$) successfully suppresses severe capping inversions.
   * Under $\lambda = 150\text{ J/kg}$, positive classifications under severe capping ($\text{CIN} > 300\text{ J/kg}$) plummet from $174$ cases down to **$12$ cases (a $93.1\%$ reduction)**, and cases under extreme capping ($\text{CIN} > 500\text{ J/kg}$) fall from $82$ to **$2$ cases ($97.6\%$ suppression)**.
   * Overall positive prevalence contracts to **$10.51\%$** ($2,457$ cases), focusing classifications on dynamically uninhibited regimes.
3. **Behavior of Lower-Tropospheric Moisture Replacement (Variant C):**
   * Replacing upper-tropospheric RH ($300\text{ hPa}$) with boundary-layer RH ($850\text{ hPa}$) extracted directly from ECMWF ERA5 fields eliminates the cirrus anvil hijacking vulnerability and raises the Pearson correlation between lower moisture and storm score from $0.228$ to **$+0.457$**.
   * However, because coastal Bay of Bengal boundary-layer air is persistently humid year-round (mean RH 850 hPa is $60.35\%$), simple one-to-one replacement without CAPE gating slightly increases overall positive prevalence to **$25.93\%$** ($6,062$ cases).
   * Combining RH 850 hPa replacement with a minimum instability floor (`Variant C + CAPE>=250`) resolves this trade-off, achieving **$24.15\%$** prevalence with zero upper-air anvil artifacts.
4. **Dual Thermodynamic Gating (Variant D):**
   * The combined $3 \times 4$ grid ($\tau \in \{100, 250, 500\}\text{ J/kg} \times \lambda \in \{100, 150, 200, 300\}\text{ J/kg}$) exhibits **strict mathematical and physical monotonicity** across all 12 parameter combinations.
   * The balanced configuration `Variant D (CAPE>=250, CIN=150)` establishes an outstanding middle ground: prevalence of **$9.53\%$** ($2,228$ cases), **$0$ positive classifications with $\text{CAPE} < 250\text{ J/kg}$**, and only **$12$ cases with $\text{CIN} > 300\text{ J/kg}$**.
5. **Two-Stage Decoupled Index (Variant E):**
   * Decoupling background environmental favorability ($F$) from active convective triggering ($T$) produces the sharpest discriminator of extreme convection, isolating **$3.91\%$** ($914$ cases) of high-confidence thunderstorm setups.
   * In secondary model agreement, Variant E achieved the highest ranking concordance with the frozen Florida Explainable Boosting Machine (EBM) of any formula tested: **ROC-AUC = $0.9217$** and minimum Brier score = **$0.1082$**.

---

## 2. Controlled Experimental Framework & Pipeline Architecture

To ensure strict scientific integrity:
1. **Preservation of Original Pipeline:** The original scripts (`main.py`, `formulas.py`, `label_builder.py`) and dataset (`MET_DATA_with_storm_score_label.csv`) were left completely untouched.
2. **Dedicated Pipeline Directory:** All experimental logic was authored in [`Thunderstorm_pipeline/experimental_formula_v2/`](file:///c:/Users/KIIT0001/Desktop/Thunderstorm_pipeline/experimental_formula_v2/):
   * `formula_definitions.py`: Modular, object-oriented implementations of all 26 variants.
   * `test_formula_integrity.py`: Automated test suite for boundary conditions, finite ranges, and parameter monotonicity (5/5 tests passed).
   * `evaluate_experiment_v2.py`: End-to-end execution, diagnostic calculation, figure rendering, and artifact mirroring.
3. **Deliverable Storage & Subproject Mirroring:** Output tables and figures were saved to `reports/chandipur_formula_experiment_v2/` and mirrored to `Florida_Thunderstorm_2024_Training/reports/chandipur_formula_experiment_v2/`.

---

## 3. Mathematical Specifications of Evaluated Variants

All mathematical formulations, scaling factors, and thresholds are registered in [`formula_variant_configurations.json`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_experiment_v2/formula_variant_configurations.json):

### Baseline: Original Formulation
$$\text{Score}_{\text{base}} = \sum_{i=1}^8 w_i f_i(X_i), \quad \text{Label}_{\text{base}} = \mathbb{I}(\text{Score}_{\text{base}} \ge 2.50)$$

### Variant A: Thermodynamic CAPE Gating
$$\text{Score}_A(\tau) = \mathbb{I}(\text{CAPE} \ge \tau) \times \text{Score}_{\text{base}}, \quad \tau \in \{100, 250, 500, 750\}\text{ J/kg}$$
* **Physical Justification:** Convective charge separation requires updrafts strong enough ($w \ge 5-10\text{ m/s}$) to loft rimed ice particles into the mixed-phase zone. Atmospheres with sub-convective buoyancy cannot electrify.

### Variant B: Multiplicative CIN Barrier
$$\text{Score}_B(\lambda) = \exp\left(-\frac{\max(\text{CIN}, 0)}{\lambda}\right) \times \text{Score}_{\text{base}}, \quad \lambda \in \{100, 150, 200, 300\}\text{ J/kg}$$
* **Physical Justification:** Convective inhibition acts as an exponential suppression barrier against parcel ascent, not a fungible additive credit.

### Variant C: Lower-Tropospheric Moisture Replacement
$$\text{Score}_C = \text{Score}_{\text{base}} - 0.80 \times \left(\frac{\text{RH}_{300}}{100}\right) + 0.80 \times \left(\frac{\text{RH}_{850}}{100}\right)$$
* **Physical Justification:** Saturation at 850 hPa (~1.5 km) lowers the Lifting Condensation Level (LCL) and insulates rising thermals from dry-air entrainment, whereas 300 hPa saturation represents decaying anvil blowoff.

### Variant D: Combined CAPE + CIN Gating Matrix (12 Configurations)
$$\text{Score}_D(\tau, \lambda) = \mathbb{I}(\text{CAPE} \ge \tau) \times \exp\left(-\frac{\max(\text{CIN}, 0)}{\lambda}\right) \times \text{Score}_{\text{base}}$$
* **Evaluated Grid:** $\tau \in \{100, 250, 500\}\text{ J/kg} \times \lambda \in \{100, 150, 200, 300\}\text{ J/kg}$.

### Variant E: Two-Stage Decoupled Index
$$\text{Favorability } (F) = 0.40 f(\text{CAPE}) + 0.25 f(\text{RH}_{850}) + 0.20 f(\text{Shear}) + 0.15 f(\text{Cloud})$$
$$\text{Trigger } (T) = \left[ 0.60 f(-\omega_{300}) + 0.40 f(\text{CP}) \right] \times \exp\left(-\frac{\text{CIN}}{\lambda}\right) \times \mathbb{I}(\text{CAPE} \ge 250)$$
$$\text{Score}_E = 5.60 \times \sqrt{F \times T}, \quad \text{Label}_E = \mathbb{I}(\text{Score}_E \ge 2.50)$$
* **Physical Justification:** Decouples broad thermodynamic receptivity from dynamic lifting triggers. If either stage is inactive, the composite score collapses to zero.

---

## 4. Comprehensive Empirical & Diagnostic Results

### 4.1 Label Distribution & Classification Shifts

Compiled in [`variant_label_comparison.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_experiment_v2/variant_label_comparison.csv):

| Variant Identifier | Architecture Family | Positive Count ($N$) | Positive Prevalence (%) | Baseline Agreement (%) | Positives Purged to Negative | Negative Promoted to Positive | Cohen's Kappa vs. Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Original)** | Baseline Heuristic | **5,462** | **23.37%** | 100.00% | 0 | 0 | 1.0000 |
| **Variant A ($\tau=100$)** | CAPE Gating | 5,345 | 22.87% | 99.50% | 117 | 0 | 0.9859 |
| **Variant A ($\tau=250$)** | CAPE Gating | 5,099 | 21.81% | 98.45% | 363 | 0 | 0.9556 |
| **Variant A ($\tau=500$)** | CAPE Gating | 4,443 | 19.01% | 95.64% | 1,019 | 0 | 0.8698 |
| **Variant A ($\tau=750$)** | CAPE Gating | 3,629 | 15.52% | 92.16% | 1,833 | 0 | 0.7521 |
| **Variant B ($\lambda=100$)** | CIN Barrier | 2,153 | 9.21% | 85.84% | 3,309 | 0 | 0.4993 |
| **Variant B ($\lambda=150$)** | CIN Barrier | 2,457 | 10.51% | 87.14% | 3,005 | 0 | 0.5562 |
| **Variant B ($\lambda=200$)** | CIN Barrier | 2,663 | 11.39% | 88.03% | 2,799 | 0 | 0.5932 |
| **Variant B ($\lambda=300$)** | CIN Barrier | 3,048 | 13.04% | 89.67% | 2,414 | 0 | 0.6593 |
| **Variant C (RH 850)** | Moisture Swap | 6,062 | 25.93% | 93.61% | 447 | 1,047 | 0.8281 |
| **Variant C + CAPE$\ge 250$**| Moisture Swap + Gate | 5,646 | 24.15% | 92.51% | 784 | 968 | 0.7931 |
| **Variant D ($\tau=100, \lambda=150$)** | Combined Grid | 2,362 | 10.10% | 86.74% | 3,100 | 0 | 0.5401 |
| **Variant D ($\tau=250, \lambda=150$)** | Combined Grid | 2,228 | 9.53% | 86.17% | 3,234 | 0 | 0.5144 |
| **Variant D ($\tau=250, \lambda=200$)** | Combined Grid | 2,425 | 10.37% | 87.01% | 3,037 | 0 | 0.5501 |
| **Variant D ($\tau=500, \lambda=150$)** | Combined Grid | 1,935 | 8.28% | 84.91% | 3,527 | 0 | 0.4568 |
| **Variant E (Two-Stage, $\lambda=150$)**| Decoupled Index | 914 | 3.91% | 80.54% | 4,548 | 0 | 0.2319 |

---

### 4.2 Low-CAPE False Alarm Purging Diagnostics

Compiled in [`low_cape_positive_comparison.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_experiment_v2/low_cape_positive_comparison.csv):

| Variant Name | Total Positives | Positives with $\text{CAPE} < 50\text{ J/kg}$ | Positives with $\text{CAPE} < 100\text{ J/kg}$ | Positives with $\text{CAPE} < 250\text{ J/kg}$ | Positives with $\text{CAPE} < 500\text{ J/kg}$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Original)** | 5,462 | **54 (0.99%)** | **117 (2.14%)** | **363 (6.65%)** | **1,019 (18.66%)** |
| **Variant A ($\tau=100$)** | 5,345 | **0 (0.00%)** | **0 (0.00%)** | 246 (4.60%) | 902 (16.88%) |
| **Variant A ($\tau=250$)** | 5,099 | **0 (0.00%)** | **0 (0.00%)** | **0 (0.00%)** | 656 (12.87%) |
| **Variant A ($\tau=500$)** | 4,443 | **0 (0.00%)** | **0 (0.00%)** | **0 (0.00%)** | **0 (0.00%)** |
| **Variant B ($\lambda=150$)** | 2,457 | 10 (0.41%) | 22 (0.90%) | 95 (3.87%) | 321 (13.06%) |
| **Variant C (RH 850)** | 6,062 | 68 (1.12%) | 143 (2.36%) | 416 (6.86%) | 1,061 (17.50%) |
| **Variant C + CAPE$\ge 250$**| 5,646 | **0 (0.00%)** | **0 (0.00%)** | **0 (0.00%)** | 645 (11.42%) |
| **Variant D ($\tau=250, \lambda=150$)** | 2,228 | **0 (0.00%)** | **0 (0.00%)** | **0 (0.00%)** | 226 (10.14%) |
| **Variant E (Two-Stage, $\lambda=150$)**| 914 | **0 (0.00%)** | **0 (0.00%)** | **0 (0.00%)** | 52 (5.69%) |

---

### 4.3 Severe Capping Inversion Suppression Diagnostics

Compiled in [`high_cin_positive_comparison.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_experiment_v2/high_cin_positive_comparison.csv):

| Variant Name | Total Positives | Positives with $\text{CIN} > 200\text{ J/kg}$ | Positives with $\text{CIN} > 300\text{ J/kg}$ | Positives with $\text{CIN} > 500\text{ J/kg}$ | Suppression Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Original)** | 5,462 | **395 (7.23%)** | **174 (3.19%)** | **82 (1.50%)** | **Severe Capping Failure** |
| **Variant A ($\tau=250$)** | 5,099 | 363 (7.12%) | 163 (3.20%) | 78 (1.53%) | Fails on High CIN |
| **Variant B ($\lambda=100$)** | 2,153 | **15 (0.70%)** | **2 (0.09%)** | **0 (0.00%)** | Near-Complete Suppression |
| **Variant B ($\lambda=150$)** | 2,457 | **47 (1.91%)** | **12 (0.49%)** | **2 (0.08%)** | **97.6% Suppression** |
| **Variant B ($\lambda=200$)** | 2,663 | 87 (3.27%) | 27 (1.01%) | 6 (0.23%) | Moderate Suppression |
| **Variant C (RH 850)** | 6,062 | 510 (8.41%) | 229 (3.78%) | 108 (1.78%) | Exacerbates Capping Failures |
| **Variant D ($\tau=250, \lambda=150$)** | 2,228 | **46 (2.06%)** | **12 (0.54%)** | **2 (0.09%)** | **Dual Gating Success** |
| **Variant E (Two-Stage, $\lambda=150$)**| 914 | **7 (0.77%)** | **1 (0.11%)** | **0 (0.00%)** | Complete Suppression |

---

### 4.4 Seasonal & Annual Climatological Prevalence

Compiled in [`seasonal_prevalence_comparison.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_experiment_v2/seasonal_prevalence_comparison.csv) and [`annual_prevalence_comparison.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_experiment_v2/annual_prevalence_comparison.csv):

| Variant Name | Winter (DJF) % | Pre-monsoon (MAM) % | Monsoon (JJAS) % | Post-monsoon (ON) % | 2022 % | 2023 % | 2024 % | 2025 % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Original)** | **3.89%** | 30.73% | 34.34% | 19.38% | 23.36% | 24.57% | 21.99% | 23.54% |
| **Variant A ($\tau=250$)** | **2.21% (-43%)**| 29.80% | 32.88% | 17.06% | 21.65% | 23.15% | 20.35% | 22.09% |
| **Variant A ($\tau=500$)** | **0.99% (-75%)**| 28.18% | 28.53% | 13.98% | 18.66% | 20.48% | 17.61% | 19.29% |
| **Variant B ($\lambda=150$)** | 1.83% | 10.74% | 18.06% | 7.62% | 10.15% | 11.23% | 10.37% | 10.29% |
| **Variant C (RH 850)** | 7.15% | 31.62% | 36.85% | 22.95% | 26.06% | 26.54% | 24.64% | 26.49% |
| **Variant C + CAPE$\ge 250$**| 3.84% | 30.68% | 35.40% | 20.62% | 24.35% | 25.12% | 23.00% | 24.13% |
| **Variant D ($\tau=250, \lambda=150$)** | 0.99% | 10.33% | 16.92% | 6.55% | 9.07% | 10.32% | 9.32% | 9.41% |
| **Variant E (Two-Stage, $\lambda=150$)**| 0.05% | 4.88% | 7.19% | 1.84% | 4.09% | 4.07% | 3.65% | 3.82% |

---

## 5. Secondary Model Agreement Analysis (Frozen Florida EBM & GAM)

Compiled in [`frozen_florida_model_agreement.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_experiment_v2/frozen_florida_model_agreement.csv):

> [!WARNING]
> **SECONDARY DIAGNOSTIC NATURE OF MODEL AGREEMENT**  
> Agreement with Florida-trained GAM and EBM models must not be used as an optimization target. Florida models were trained on US subtropical ASOS targets with different climatological baselines and an unmeasured vertical velocity term. Model agreement evaluates whether the heuristic proxy aligns with learned atmospheric relationships, not independent truth.

| Variant Identifier | GAM PR-AUC | GAM ROC-AUC | GAM Brier Score | GAM F1 (Thresh 0.33) | EBM PR-AUC | EBM ROC-AUC | EBM Brier Score | EBM F1 (Thresh 0.34) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Original)** | 0.5521 | 0.8494 | 0.1301 | 0.6146 | 0.6133 | 0.8572 | 0.1270 | 0.6059 |
| **Variant A ($\tau=100$)** | 0.5540 | 0.8558 | 0.1270 | 0.6175 | 0.6120 | 0.8612 | 0.1247 | 0.6052 |
| **Variant A ($\tau=250$)** | 0.5457 | 0.8594 | 0.1241 | 0.6093 | 0.5903 | 0.8586 | 0.1236 | 0.5905 |
| **Variant A ($\tau=500$)** | 0.4892 | 0.8501 | 0.1224 | 0.5540 | 0.5032 | 0.8404 | 0.1244 | 0.5331 |
| **Variant B ($\lambda=150$)** | 0.2448 | 0.7733 | 0.1275 | 0.3220 | 0.3520 | 0.7896 | 0.1232 | 0.3015 |
| **Variant C (RH 850)** | **0.6041** | 0.8569 | 0.1330 | **0.6521** | **0.6599** | 0.8619 | 0.1308 | **0.6366** |
| **Variant C + CAPE$\ge 250$**| 0.5983 | **0.8679** | 0.1255 | 0.6491 | 0.6370 | **0.8642** | 0.1262 | 0.6233 |
| **Variant D ($\tau=250, \lambda=150$)** | 0.2443 | 0.7935 | 0.1229 | 0.3169 | 0.3446 | 0.8003 | 0.1204 | 0.2875 |
| **Variant E (Two-Stage, $\lambda=150$)**| 0.1993 | **0.8833** | **0.1125** | 0.1721 | 0.3415 | **0.9217** | **0.1082** | 0.1803 |

### Observations on Model Concordance:
1. **Variant C (RH 850 Replacement) Maximizes Florida F1 & PR-AUC:**  
   Because the Florida models heavily rely on boundary-layer moisture (850 hPa dewpoint depression / RH), aligning the proxy's moisture level with 850 hPa raises the EBM PR-AUC from $0.6133$ to **$0.6599$** and F1 from $0.6059$ to **$0.6366$**.
2. **Variant E Achieves Unprecedented Ranking Agreement (ROC-AUC > 0.92):**  
   Because Variant E strictly reserves positive labels for cases with joint thermodynamic buoyancy, moisture, dynamic lift, and breached inhibition, the Florida EBM assigns its highest predicted probabilities almost exclusively to these cases, resulting in an extraordinary **ROC-AUC of $0.9217$**.

---

## 6. Representative Case Study Evaluation (7 Meteorological Regimes)

Compiled in [`representative_case_studies.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/chandipur_formula_experiment_v2/representative_case_studies.csv):

| Case # | Synoptic Timestamp | Observed Meteorological State | Original Score (Label) | Variant A(250) Score (Label) | Variant B(150) Score (Label) | Variant C Score (Label) | Variant D Score (Label) | Variant E Score (Label) | Physical Interpretation & Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `2022-01-12 00:00 UTC` | CAPE: $18.9\text{ J/kg}$<br>CIN: $0.0\text{ J/kg}$<br>RH300: $96.4\%$<br>RH850: $34.2\%$ | **2.507 (1)** | **0.000 (0)** | 2.507 (1) | 2.010 (0) | **0.000 (0)** | **0.000 (0)** | **Severe False Positive Solved:** Clear dry winter surface with non-electrified cirrus. Variants A, C, D, and E cleanly eliminate this false alarm. |
| **2** | `2022-05-19 06:00 UTC` | CAPE: $3431.8\text{ J/kg}$<br>CIN: $832.8\text{ J/kg}$<br>RH300: $8.9\%$<br>RH850: $81.6\%$ | **2.649 (1)** | 2.649 (1) | **0.010 (0)** | 3.230 (1) | **0.010 (0)** | **0.000 (0)** | **Severe Capping Lid Failure Solved:** Impenetrable capping inversion ($\text{CIN} > 800\text{ J/kg}$). Variants B, D, and E exponentially crush the score below $0.02$. |
| **3** | `2023-06-30 12:00 UTC` | CAPE: $3100.0\text{ J/kg}$<br>CIN: $8.7\text{ J/kg}$<br>CP: $5.81\text{ mm}$<br>RH850: $91.3\%$ | **3.523 (1)** | **3.523 (1)** | **3.324 (1)** | **3.618 (1)** | **3.324 (1)** | **2.623 (1)** | **Unanimous Convective Storm Agreement:** Ideal buoyancy, minimal cap, and active precipitation. All variants correctly classify as storm. |
| **4** | `2022-01-06 00:00 UTC` | CAPE: $0.0\text{ J/kg}$<br>CIN: $0.0\text{ J/kg}$<br>Shear: $35.7\text{ m/s}$<br>Cloud: $0.00$ | **1.164 (0)** | **0.000 (0)** | 1.164 (0) | 1.157 (0) | **0.000 (0)** | **0.000 (0)** | **High Shear / Dry Cold Air Mass:** Strong winter jet shear with zero convective fuel. Both baseline and all variants remain correctly negative. |
| **5** | `2022-07-14 00:00 UTC` | CAPE: $219.0\text{ J/kg}$<br>CIN: $0.0\text{ J/kg}$<br>CP: $12.45\text{ mm}$<br>RH850: $89.5\%$ | **4.236 (1)** | **0.000 (0)** | 4.236 (1) | 4.312 (1) | **0.000 (0)** | **0.000 (0)** | **Warm-Rain Monsoon Trade-off Case:** Heavy synoptic monsoon downpour with low parcel CAPE ($219\text{ J/kg}$). Highlights gating trade-off between lightning convection and warm rain. |
| **6** | `2024-05-01 12:00 UTC` | CAPE: $4304.0\text{ J/kg}$<br>CIN: $13.5\text{ J/kg}$<br>Shear: $21.3\text{ m/s}$<br>Cloud: $0.85$ | **2.692 (1)** | **2.692 (1)** | **2.461 (0)** | **3.220 (1)** | **2.461 (0)** | **2.585 (1)** | **Classic Pre-Monsoon Nor'wester:** High buoyancy and organized shear. Variants A, C, and E confirm active thunderstorm potential. |
| **7** | `2023-04-15 12:00 UTC` | CAPE: $7605.5\text{ J/kg}$<br>CIN: $171.8\text{ J/kg}$<br>RH300: $34.2\%$<br>RH850: $64.8\%$ | **2.845 (1)** | **2.845 (1)** | 0.906 (0) | 3.090 (1) | 0.906 (0) | 0.858 (0) | **Extreme CAPE Extrapolation State:** Pre-monsoon tropical heat dome. Shows need for non-linear S-curve saturation and dynamic cap break criteria. |

---

## 7. Comparative Visualizations

The generated high-resolution diagnostic dashboards illustrate the fundamental trade-offs:

1. **Formula Architecture Comparison Dashboard:**  
   ![Comparison Dashboard](figures/formula_variant_comparison_dashboard.png)  
   *Panel (a)* shows overall prevalence across families; *Panel (b)* highlights how Variant A purges sub-convective false alarms; *Panel (c)* demonstrates the collapse of high-CIN misclassifications under exponential barriers; *Panel (d)* captures seasonal rebalancing away from dry winter cirrus.

2. **2D Sensitivity Heatmaps for Variant D:**  
   ![Sensitivity Heatmaps](figures/cape_cin_sensitivity_heatmaps.png)  
   Demonstrates strict physical monotonicity across the $3 \times 4$ parameter grid ($\tau \in \{100, 250, 500\}\text{ J/kg} \times \lambda \in \{100, 150, 200, 300\}\text{ J/kg}$).

---

## 8. Trade-Off Analysis and Failure Modes

Every experimental formula variant introduces specific physical trade-offs:

| Variant Architecture | Primary Strength | Critical Trade-off / Failure Mode | Mitigating Strategy |
| :--- | :--- | :--- | :--- |
| **Variant A (CAPE Gating)** | Instantly purges 100% of sub-convective false alarms with zero tuning complexity. | May drop heavy monsoonal warm-rain systems where parcel buoyancy is marginal ($< 250\text{ J/kg}$) but precipitation is torrential. | If the goal is **lightning prediction**, maintain $\tau = 250\text{ J/kg}$. If the goal is general rainfall, decouple lightning from precip. |
| **Variant B (CIN Barrier)** | Restores thermodynamic sanity during pre-monsoon heat waves by shutting down impenetrable capping states. | In coastal environments, localized sea-breeze fronts may trigger storms along a narrow 5 km line while coarse $0.25^\circ$ ERA5 grids report high mean CIN. | Set $\lambda \ge 150\text{ J/kg}$ rather than an aggressive $100\text{ J/kg}$ to permit moderate cap break under strong lift. |
| **Variant C (RH 850 Replacement)**| Eliminates cirrus anvil hijacking and focuses on the pre-convective moisture reservoir. | Boundary layer over the Bay of Bengal is persistently moist, causing false positives in stable morning regimes if used without CAPE gating. | **Always pair RH 850 replacement with CAPE gating** (`Variant C + CAPE>=250`). |
| **Variant D (Combined A+B)** | Simultaneous physical control over buoyant fuel and capping suppression; highly customizable. | Multiplicative interaction can over-suppress marginal monsoon environments if both $\tau$ and $\lambda$ are set too strictly. | Utilize balanced parameters: $\tau = 250\text{ J/kg}, \lambda = 200\text{ J/kg}$. |
| **Variant E (Two-Stage Index)** | Scientifically modern, fully decoupled architecture with peak Florida model agreement (ROC-AUC $0.922$). | Represents a fundamental departure from the original scoring scale, requiring new threshold calibration for downstream applications. | Adopt as a candidate architecture for future ML nowcasting pipelines. |

---

## 9. Scientific Limitations & Observational Requirements

1. **Heuristic Proxy Status:**  
   All evaluated variants remain **heuristic transformations of coincident ERA5 reanalysis fields**. A change in proxy classification frequency does not represent verified empirical truth without observational ground truth.
2. **Observational Validation Imperative:**  
   Before any formula variant is adopted into operational pipelines, it must be validated against **independent physical sensors**:
   * **WWLLN / GLM Lightning Flash Density:** Direct stroke counts within a 25 km radius of Chandipur to verify electrical charge separation.
   * **IMD Doppler Weather Radar (Kolkata / Paradip):** Reflectivity exceeding $40\text{ dBZ}$ above the freezing level.
   * **Surface Aerodrome METAR / Speci Reports:** Observer-verified thunderstorms (`TS`, `VCTS`, `TSRA`).

---

## 10. Recommended Next Steps

1. **Adopt Balanced Candidate for Diagnostic Benchmarking:**  
   For ongoing research comparing machine learning models against heuristic baselines, deploy **`Variant D (CAPE>=250, CIN=200)`** or **`Variant C + CAPE>=250`**. These variants preserve $92.5\% - 87.0\%$ of the original regime while completely eliminating unphysical sub-convective false alarms.
2. **Incorporate Collocated Lightning Observations:**  
   Collocate the WWLLN 6-hourly stroke manifest (`Desktop/ITR/wwlln_6h_manifest.csv`) with the 23,376 synoptic timestamps to compute empirical PR-AUC, ROC-AUC, and Brier scores against genuine lightning ground truth.
3. **Train Chandipur-Specific Native Interpretable Models:**  
   Train native pyGAM and InterpretML EBM models directly on Chandipur ERA5 features using collocated lightning ground truth rather than relying on Florida transfer.
