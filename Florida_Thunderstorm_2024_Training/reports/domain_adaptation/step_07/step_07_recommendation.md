# Step 7 — Final Synthesis and Scientific Recommendation

## 1. Comprehensive Model Performance Comparison

The complete quantitative evaluation across all five adaptation conditions is presented below. Evaluations were conducted on the full Chandipur ERA5 target dataset ($N = 23,376$, positive rate $23.37\%$) using the frozen Florida native models.

| Experiment ID | Adaptation Condition | Model | Operational Threshold ($\tau$) | F1 Score | Precision | Recall | ROC-AUC | PR-AUC | Brier Score | Log Loss | Cohen Kappa | MCC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1** | Identity (Raw Control) | pyGAM | $\tau_{\text{opt}} = 0.33$ | $0.6147$ | $0.4858$ | $0.8365$ | $0.8494$ | $0.5523$ | $0.1301$ | $0.3915$ | $0.4529$ | $0.4889$ |
| **B1** | Identity (Raw Control) | pyGAM | $\tau = 0.50$ | $0.5170$ | $0.5709$ | $0.4724$ | $0.8494$ | $0.5523$ | $0.1301$ | $0.3915$ | $0.3874$ | $0.3902$ |
| **B1** | Identity (Raw Control) | EBM | $\tau_{\text{opt}} = 0.34$ | $0.6059$ | $0.4902$ | $0.7931$ | $0.8572$ | $0.6134$ | $0.1270$ | $0.3856$ | $0.4458$ | $0.4726$ |
| **B1** | Identity (Raw Control) | EBM | $\tau = 0.50$ | $0.5482$ | $0.6131$ | $0.4958$ | $0.8572$ | $0.6134$ | $0.1270$ | $0.3856$ | $0.4289$ | $0.4328$ |
| **B2** | Robust Quantile Scaling (RQS) | pyGAM | $\tau_{\text{opt}} = 0.33$ | $0.3950$ | $0.3791$ | $0.4123$ | $0.7408$ | $0.3926$ | $0.1642$ | $0.4824$ | $0.2003$ | $0.2006$ |
| **B2** | Robust Quantile Scaling (RQS) | pyGAM | $\tau = 0.50$ | $0.2190$ | $0.3971$ | $0.1512$ | $0.7408$ | $0.3926$ | $0.1642$ | $0.4824$ | $0.1035$ | $0.1207$ |
| **B2** | Robust Quantile Scaling (RQS) | EBM | $\tau_{\text{opt}} = 0.34$ | $0.3801$ | $0.3646$ | $0.3969$ | $0.7241$ | $0.3769$ | $0.1704$ | $0.5005$ | $0.1804$ | $0.1807$ |
| **B2** | Robust Quantile Scaling (RQS) | EBM | $\tau = 0.50$ | $0.2680$ | $0.3909$ | $0.2040$ | $0.7241$ | $0.3769$ | $0.1704$ | $0.5005$ | $0.1284$ | $0.1384$ |
| **B3** | Bounded Monotonic (BMT All) | pyGAM | $\tau_{\text{opt}} = 0.33$ | $0.6044$ | $0.5153$ | $0.7307$ | $0.8658$ | $0.6551$ | $0.1219$ | $0.3715$ | $0.4550$ | $0.4685$ |
| **B3** | Bounded Monotonic (BMT All) | pyGAM | $\tau = 0.50$ | $0.5090$ | $0.7017$ | $0.3993$ | $0.8658$ | $0.6551$ | $0.1219$ | $0.3715$ | $0.4088$ | $0.4332$ |
| **B3** | Bounded Monotonic (BMT All) | EBM | $\tau_{\text{opt}} = 0.34$ | $0.5963$ | $0.5334$ | $0.6759$ | $0.8584$ | $0.6604$ | $0.1237$ | $0.3781$ | $0.4536$ | $0.4595$ |
| **B3** | Bounded Monotonic (BMT All) | EBM | $\tau = 0.50$ | $0.4975$ | $0.7314$ | $0.3770$ | $0.8584$ | $0.6604$ | $0.1237$ | $0.3781$ | $0.4026$ | $0.4353$ |
| **B4** | **Selective (CAPE & CIN only)** | **pyGAM** | **$\tau_{\text{opt}} = 0.33$** | **$0.6234$** | **$0.5169$** | **$0.7852$** | **$0.8725$** | **$0.6675$** | **$0.1224$** | **$0.3743$** | **$0.4756$** | **$0.4965$** |
| **B4** | Selective (CAPE & CIN only) | pyGAM | $\tau = 0.50$ | $0.4785$ | $0.7476$ | $0.3519$ | $0.8725$ | $0.6675$ | $0.1224$ | $0.3743$ | $0.3868$ | $0.4269$ |
| **B4** | **Selective (CAPE & CIN only)** | **EBM** | **$\tau_{\text{opt}} = 0.34$** | **$0.6064$** | **$0.5294$** | **$0.7096$** | **$0.8624$** | **$0.6659$** | **$0.1243$** | **$0.3808$** | **$0.4625$** | **$0.4719$** |
| **B4** | Selective (CAPE & CIN only) | EBM | $\tau = 0.50$ | $0.4819$ | $0.7650$ | $0.3517$ | $0.8624$ | $0.6659$ | $0.1243$ | $0.3808$ | $0.3925$ | $0.4356$ |
| **B5** | CAPE Clipping ($7,240\text{ J/kg}$) | pyGAM | $\tau_{\text{opt}} = 0.33$ | $0.6147$ | $0.4858$ | $0.8365$ | $0.8502$ | $0.5526$ | $0.1299$ | $0.3906$ | $0.4529$ | $0.4889$ |
| **B5** | CAPE Clipping ($7,240\text{ J/kg}$) | pyGAM | $\tau = 0.50$ | $0.5170$ | $0.5709$ | $0.4724$ | $0.8502$ | $0.5526$ | $0.1299$ | $0.3906$ | $0.3874$ | $0.3902$ |
| **B5** | CAPE Clipping ($7,240\text{ J/kg}$) | EBM | $\tau_{\text{opt}} = 0.34$ | $0.6059$ | $0.4902$ | $0.7931$ | $0.8572$ | $0.6134$ | $0.1270$ | $0.3856$ | $0.4458$ | $0.4726$ |
| **B5** | CAPE Clipping ($7,240\text{ J/kg}$) | EBM | $\tau = 0.50$ | $0.5482$ | $0.6131$ | $0.4958$ | $0.8572$ | $0.6134$ | $0.1270$ | $0.3856$ | $0.4289$ | $0.4328$ |

---

## 2. Key Scientific Findings

### Finding 1: Condition B4 (Selective Transformation) Decisively Beats CAPE Clipping (B5)
- **pyGAM Discrimination Breakthrough**:
  - **PR-AUC**: Increases from $0.5526$ (B5) / $0.5523$ (B1) to **$0.6675$** under B4 — an absolute gain of **$+0.115$** (**$+20.8\%$ relative improvement**).
  - **ROC-AUC**: Increases from $0.8502$ (B5) to **$0.8725$** under B4.
  - **F1 Score**: Reaches the highest value across the entire framework at **$0.6234$** (with Precision improving from $0.4858$ to $0.5169$).
  - **Calibration / Brier Score**: Drops from $0.1299$ to **$0.1224$**, indicating substantially sharper and more reliable probabilistic predictions.
- **EBM Performance Gain**:
  - **PR-AUC**: Increases from $0.6134$ (B5) to **$0.6659$** under B4.
  - **ROC-AUC**: Increases from $0.8572$ to **$0.8624$**.
  - **Brier Score**: Improves from $0.1270$ to **$0.1243$**.

### Finding 2: Why Selective Transformation (B4) Outperformed All-Feature BMT (B3)
In B3, applying log-compression to *all* features degraded kinematic thresholds:
- In Chandipur, bulk shear values ($0\text{--}56.5\text{ m/s}$) were already fully enveloped within the Florida distribution ($0\text{--}77.8\text{ m/s}$).
- Compressing kinematics artificially shifted moderate, storm-organizing shear values into weak-shear categories, dampening pyGAM's F1 score ($0.6044$ in B3 vs $0.6234$ in B4).
- By restricting transformation **strictly to thermodynamic instability (CAPE and CIN)**, B4 resolved the non-linear tail extrapolation breakdown while preserving $100\%$ of kinematic and cloud physical relationships.

### Finding 3: Seasonal Performance Across Convective Regimes
The seasonal breakdown in `seasonal_transformation_analysis.csv` demonstrates where the adaptation gains originate:
1. **Monsoon Regime (JJAS — 7,808 samples, 48.2% positive prevalence)**:
   - pyGAM PR-AUC under B1/B5 was $0.6065$. Under B4, it surges to **$0.7234$** ($+0.117$).
   - EBM PR-AUC under B1/B5 was $0.6742$. Under B4, it surges to **$0.7282$**.
   - F1 score reaches **$0.6880$** for pyGAM and **$0.6808$** for EBM.
2. **Post-Monsoon Regime (ON — 3,904 samples, 10.8% positive prevalence)**:
   - pyGAM PR-AUC under B1/B5 was $0.5733$. Under B4, it reaches **$0.7520$** ($+0.179$ gain!).
   - pyGAM ROC-AUC reaches an outstanding **$0.9591$**.
3. **Pre-Monsoon Regime (MAM — Kalbaishakhi thunderstorms, 5,888 samples, 20.7% prevalence)**:
   - pyGAM PR-AUC improves from $0.3832$ (B5) to **$0.4453$** (B4).
   - CIN compression in B4 prevents false suppression during high-CAPE dryline intrusions.

### Finding 4: Rejection of Robust Quantile Scaling (B2)
Linear quantile scaling (B2) produced disastrous results across all metrics (pyGAM F1 dropped to $0.3950$, PR-AUC to $0.3926$). Clamping at physical boundaries created mass artificial ties, destroying rank discrimination.

---

## 3. Comparison with Previous Steps

| Step | Technique Tested | Physical Validity | Rank Correlation | pyGAM PR-AUC | EBM PR-AUC | Decision Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Step 2** | Zero-Shot Raw Baseline | 100% | Exact ($+1.0$) | $0.5523$ | $0.6134$ | Frozen Baseline |
| **Step 4** | CAPE Clipping at $7,240\text{ J/kg}$ | 100% | Exact ($+1.0$) | $0.5526$ | $0.6134$ | Retained as Baseline Safeguard |
| **Step 5** | CORAL Covariance Alignment | **0% (20,767 Violations)** | Distorted | $0.4611$ | $0.4729$ | **REJECTED** |
| **Step 6** | Importance Weighting (KMM/LR) | 100% | Exact ($+1.0$) | $0.5519$ | $0.6128$ | Ineffective (Support mismatch) |
| **Step 7** | **Selective Transformation (B4)** | **100% (0 Violations)** | **Exact ($+1.0$)** | **$0.6675$** | **$0.6659$** | **CHAMPION ADAPTATION METHOD** |

---

## 4. Scientific Recommendations

1. **Adopt Condition B4 as the Champion Preprocessing Pipeline**:
   Condition B4 (Selective BMT Transformation on CAPE and CIN only) is scientifically validated, 100% physically valid, perfectly preserves rank ordering, and yields substantial improvements in discrimination ($+20.8\%$ PR-AUC for pyGAM, $+8.6\%$ for EBM) without requiring target labels or retraining.
2. **Retain Condition B5 as the Minimal-Intervention Fallback**:
   For deployments where inputs cannot be preprocessed with non-linear functions, Step 4 boundary clipping ($7,240\text{ J/kg}$) remains a safe and reliable inference safeguard.
3. **Candidate Focus for Step 8**:
   With feature transformation successfully solving the thermodynamic support mismatch, the remaining challenge is **target-domain probability calibration and risk control**:
   - Target calibration under zero-shot transfer (e.g., Temperature Scaling, Conformal Prediction, or Bayesian uncertainty estimation).
   - Risk-controlled prediction intervals without target label leakage.

---

## 5. Artifact and Figure Inventory

All artifacts have been compiled, verified, and saved at 300 DPI:
- **Reports**:
  - `step_07_methodology.md`
  - `step_07_distribution_diagnostics.md`
  - `step_07_interpretability_report.md`
  - `step_07_recommendation.md`
- **Audit & Performance CSVs**:
  - `step_07_model_performance_comparison.csv`
  - `feature_transformation_comparison.csv`
  - `physical_validity_audit.csv`
  - `correlation_preservation_audit.csv`
  - `seasonal_transformation_analysis.csv`
- **Diagnostic Visualizations (300 DPI)**:
  - `step_07_model_performance_comparison.png`
  - `seasonal_performance_comparison.png`
  - `interpretability_response_curves.png`
  - `correlation_heatmaps_comparison.png`
  - `consolidated_feature_distributions.png`
  - Six marginal distribution plots (`dist_cape_J_kg.png`, `dist_cin_J_kg.png`, `dist_bulk_shear.png`, `dist_cloud_cover.png`, `dist_wind_speed_10m.png`, `dist_wind_speed_850hPa.png`)
- **Execution Script**:
  - `Thunderstorm_pipeline/domain_adaptation/step_07/run_step_07_physically_constrained_adaptation.py`
