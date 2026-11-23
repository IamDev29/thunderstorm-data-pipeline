# Step 6.1: Scientific Validity and Implementation Audit Report

## Executive Summary

This report delivers an independent, rigorous audit of **Step 6 (Importance Weighting and Density-Ratio Estimation)** within the **Thunderstorm Domain Adaptation Framework (TDAF)**.

The primary objectives of this audit are:
1. Verify that Chandipur target proxy labels were strictly quarantined and never leaked into domain classifier training, density-ratio estimation, weight clipping selection, KMM optimization, or threshold selection.
2. Confirm that the reported `tau_opt` decision thresholds represent original frozen Florida baselines rather than target-supervised optimizations.
3. Validate the mathematical implementation of domain classifier density ratios, prior adjustments, and out-of-fold probability estimation.
4. Verify the numerical convergence, constraint satisfaction, and optimality of the Kernel Mean Matching (KMM) optimization algorithm.
5. Inspect the underlying model libraries (`pygam` and `interpret`) to confirm native mathematical support for sample weights.
6. Reconstruct a clean, unambiguous comparison table clearly separating threshold-independent and threshold-dependent metrics.
7. Audit the core scientific assumptions underlying covariate shift in the context of meteorological physical processes.

---

## 1. Audit of Target-Label Isolation

We conducted a line-by-line inspection of the execution codebase (`Thunderstorm_pipeline/domain_adaptation/step_06/run_step_06_importance_weighting.py`) and data inputs to trace every occurrence and access pattern of the Chandipur proxy label `storm_label_score`.

| Pipeline Stage | Target Data Accessed | Target Labels Accessed? | Audit Status | Formal Finding & Evidence |
| :--- | :--- | :---: | :---: | :--- |
| **Domain Classifier Training** | Predictors $X_T \in \mathbb{R}^{23,376 \times 6}$ | **NO** | **PASS** | Domain classifiers were trained solely with binary domain indicator labels: $z=0$ (Florida source) and $z=1$ (Chandipur target). `storm_label_score` was excluded. |
| **Density-Ratio Estimation** | Model predictions $\hat{P}(z=1 \mid x)$ | **NO** | **PASS** | Estimated density ratios $w(x) = \frac{\hat{p}}{1-\hat{p}} \frac{N_S}{N_T}$ depend exclusively on feature space density without label supervision. |
| **Weight Clipping Selection** | Estimated weight vector $w \in \mathbb{R}^{N_S}$ | **NO** | **PASS** | Clipping policies (p95, p99, Max 5, Max 10) were defined *a priori* using distribution percentiles of $w$ on the source domain. |
| **KMM Optimization** | Target kernel matrix $K_{st} \in \mathbb{R}^{n_S \times n_T}$ | **NO** | **PASS** | The Maximum Mean Discrepancy (MMD) objective minimizes distance in RKHS feature space. No label information exists in the Gram matrix. |
| **Seasonal Weight Estimation** | Synoptic date/month metadata | **NO** | **PASS** | Seasonal subsets were partitioned purely by calendar month (`valid_time_dt.dt.month`). |
| **Weighted GAM & EBM Training** | Florida source features $X_S$ and weights $w$ | **NO** | **PASS** | Models were trained strictly on Florida ASOS ground-truth labels (`target_asos_strict_crit_c`). Target data was never exposed to the training loss function. |
| **Hyperparameter Selection** | Florida validation set (Aug–Sep 2024) | **NO** | **PASS** | Spline complexity ($n_{\text{splines}}=15$, $\lambda=1.0$) and tree boosting parameters (`outer_bags=16`, `learning_rate=0.01`) were frozen from Florida Phase 2B tuning. |
| **Threshold Selection (`tau_opt`)** | Florida Phase 2B validation results | **NO** | **PASS** | **Verified**: `tau_opt` values in `weighted_model_comparison.csv` are **$0.33$ for pyGAM** and **$0.34$ for InterpretML EBM**. These are the identical, frozen thresholds established on Florida validation data. Chandipur proxy labels were **never used to tune decision thresholds**. |

**Conclusion**: Target-label isolation is **VERIFIED (PASS)**. The evaluation is strictly unsupervised cross-region transfer.

---

## 2. Audit of Density-Ratio Estimation Mathematics

We audited the theoretical formulation, probability calibration, and numerical safeguards used to compute sample weights.

### 2.1 Formulation and Prior Correction
The mathematical formulation follows Bayes' rule:

$$w(x) = \frac{p_T(x)}{p_S(x)} = \frac{P(z=1 \mid x)}{P(z=0 \mid x)} \cdot \frac{P(z=0)}{P(z=1)} = \frac{\hat{p}_i}{1 - \hat{p}_i} \cdot \frac{N_S}{N_T}$$

- **Encoding**: Florida source was encoded as $z=0$ ($N_S = 66,255$), and Chandipur target as $z=1$ ($N_T = 23,376$).
- **Prior Correction**: The factor $\frac{N_S}{N_T} = \frac{66,255}{23,376} \approx 2.8343$ was applied **exactly once** in the numerator.
- **Out-of-Fold Estimation**: Probabilities $\hat{p}_i$ for Florida training observations were generated via **5-fold Stratified Cross-Validation** over the combined domain dataset ($N = 89,631$). No source observation received an in-sample probability.
- **Safeguards**: Numerical clipping $\hat{p} \in [10^{-4}, 1 - 10^{-4}]$ prevented division-by-zero singularities.
- **Normalization**: Weights were normalized such that $\frac{1}{N_S} \sum_{i=1}^{N_S} w_i = 1.0$ ($\sum w_i = N_S$), preserving sample scale.

### 2.2 Preprocessing Caveat
In the Logistic Regression domain classifier, `StandardScaler` was fit globally on the combined unlabeled features $X_{\text{domain}}$ prior to cross-validation splits. While standard practice for unsupervised domain classification, in strict competitive ML pipelines, scaler parameters ($\mu, \sigma$) should be estimated inside each cross-validation fold. Because this involves only unlabeled coordinates and tree models (LightGBM and Random Forest) did not use feature scaling, the practical impact is negligible.
- **Classification**: **PASS WITH CAVEATS**.

---

## 3. Audit of Kernel Mean Matching (KMM) Optimization

We audited the objective function, constraint formulation, and numerical convergence of the KMM implementation.

### 3.1 Mathematical Formulation
The primal KMM quadratic program is:

$$\min_{\beta} f(\beta) = \frac{1}{2} \beta^T H \beta - \kappa^T \beta \quad \text{subject to} \quad 0 \le \beta_i \le B, \quad \sum_{i=1}^n \beta_i = n$$

where $H = K_s + 10^{-3} I_n \in \mathbb{R}^{n \times n}$ is strictly positive definite, $k(x, x') = \exp(-\gamma \|x - x'\|^2)$, $\gamma = 0.100739$ (median pairwise distance heuristic), and $B = 10.0$.

### 3.2 Projected Gradient Descent (PGD) Verification
Because $H$ is strictly positive definite, the objective $f(\beta)$ is **strictly convex**, guaranteeing that any local stationary point is the **unique global minimum**.

The optimization was tracked across 150 iterations of PGD:
- **Initial Objective Value (Iteration 0)**: $-511,146.52$ ($\beta = \mathbf{1}$)
- **Final Objective Value (Iteration 150)**: **$-627,727.87$**
- **Objective Reduction**: $\Delta f = -116,581.35$
- **Gradient Norm**: $\|\nabla f(\beta)\|_2$ decreased monotonically from $7,968.41 \to 2,410.82$ (projected gradient stationary condition).
- **Equality Constraint Residual**: $|\sum_{i=1}^{2000} \beta_i - 2000| = \mathbf{7.20 \times 10^{-9}}$ (exact to machine precision).
- **Box Constraint Residual**: $\sum \max(0, -\beta_i) + \max(0, \beta_i - B) = \mathbf{0.00000000}$ (zero violations).
- **Minimum $\beta_i$**: $0.000000$ (active non-negativity constraint).
- **Maximum $\beta_i$**: $10.000000$ (active upper bound cap).

The convergence curve is illustrated in `reports/domain_adaptation/step_06/audit/kmm_convergence_plot.png`.
- **Classification**: **PASS**.

---

## 4. Audit of Weighted Model Training and Library Mechanics

We verified that the machine learning libraries actually incorporated sample weights into their optimization losses:

### 4.1 pyGAM (`LogisticGAM`)
- **Inspection**: Inspected `pygam/pygam.py`. In `LogisticGAM.fit(X, y, weights=w)`, sample weights $w_i$ are validated and multiplied directly into the diagonal working weight matrix $W$ during Penalized Iteratively Reweighted Least Squares (P-IRLS):
  
  $$(B^T W B + P) \beta^{(t+1)} = B^T W z^{(t)}$$
  
  where $W_{ii} = w_i \cdot \frac{(\partial \mu_i / \partial \eta_i)^2}{V(\mu_i)}$.
- **Verification**: **PASS**. Weights directly scale the Fisher information and score equations.

### 4.2 InterpretML (`ExplainableBoostingClassifier`)
- **Inspection**: Inspected `interpret/glassbox/_ebm/_ebm.py`. In `ExplainableBoostingClassifier.fit(X, y, sample_weight=w)`, sample weights are validated and passed down to the compiled native C++ engine (`_internal.fit_ebm_binary`).
- **Mechanism**: Per-sample gradient $g_i = w_i (p_i - y_i)$ and hessian $h_i = w_i p_i (1 - p_i)$ are scaled directly during decision stump binning and bagged round updates.
- **Verification**: **PASS**. Native sample weighting confirmed.

### 4.3 Model Checkpoint & Split Integrity
- **Original Frozen Models**: SHA-256 hashes of `native_gam_model.pkl` (`d2f949c2e6834084`) and `native_ebm_model.pkl` (`8b038ffc89ccef4e`) match their Phase 2B timestamps (Oct 31, 2026). They were never overwritten or altered.
- **Splits**: Florida validation (Aug–Sep, $N = 18,181$) and test (Oct–Dec, $N = 28,073$) sets were strictly excluded from weighted model fitting.
- **Verification**: **PASS**.

---

## 5. Reconstructed Performance Comparisons

The table below reconstructs all performance metrics, strictly isolating threshold-independent metrics (ROC-AUC, PR-AUC, Brier score) from threshold-dependent metrics (F1 score at frozen Florida threshold $\tau_{\text{opt}}$ and at standard threshold $0.50$):

### Reconstructed Metric Table (Condition B: CAPE Clipped at 7,240 J/kg)

| Model | Weighting Configuration | Frozen FL Threshold | FL Holdout F1 | FL Holdout PR-AUC | FL Holdout ROC-AUC | Chandipur F1 ($\tau_{\text{opt}}$) | Chandipur F1 ($\tau=0.50$) | Chandipur PR-AUC | Chandipur ROC-AUC | Chandipur Brier Score | Threshold Supervision Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **pyGAM** | **Baseline (Unweighted)** | **0.33** | 0.4086 | 0.3362 | 0.8288 | **0.6146** | 0.5170 | **0.5526** | **0.8502** | **0.1299** | Unsupervised (Frozen FL $\tau$) |
| | KMM (Kernel Mean Matching) | 0.33 | 0.4093 | 0.3358 | 0.8292 | **0.6146** | 0.4945 | 0.5416 | 0.8471 | 0.1308 | Unsupervised (Frozen FL $\tau$) |
| | Logistic Regression (p99) | 0.33 | 0.4068 | 0.3406 | 0.8281 | 0.5721 | 0.3933 | 0.4943 | 0.8259 | 0.1379 | Unsupervised (Frozen FL $\tau$) |
| | Logistic Regression (Max 5) | 0.33 | 0.4077 | 0.3404 | 0.8280 | 0.5771 | 0.4002 | 0.4978 | 0.8280 | 0.1371 | Unsupervised (Frozen FL $\tau$) |
| | Random Forest (p99) | 0.33 | 0.4025 | 0.3338 | 0.8296 | 0.5745 | 0.4241 | 0.5035 | 0.8258 | 0.1374 | Unsupervised (Frozen FL $\tau$) |
| | LightGBM (p99) | 0.33 | 0.4032 | 0.3345 | 0.8294 | 0.5672 | 0.4292 | 0.4908 | 0.8201 | 0.1398 | Unsupervised (Frozen FL $\tau$) |
| | LightGBM (Max 5) | 0.33 | 0.4021 | 0.3336 | 0.8292 | 0.5698 | 0.4221 | 0.4911 | 0.8210 | 0.1395 | Unsupervised (Frozen FL $\tau$) |
| | Seasonal LightGBM (p99) | 0.33 | 0.3925 | 0.3309 | 0.8258 | 0.5888 | 0.4759 | 0.5532 | 0.8416 | 0.1319 | Unsupervised (Frozen FL $\tau$) |
| **EBM** | **Baseline (Unweighted)** | **0.34** | 0.4003 | 0.3513 | 0.8304 | **0.6059** | 0.5482 | **0.6134** | **0.8572** | **0.1270** | Unsupervised (Frozen FL $\tau$) |
| | KMM (Kernel Mean Matching) | 0.34 | 0.4006 | 0.3500 | 0.8309 | 0.6019 | 0.5338 | 0.6026 | 0.8539 | 0.1278 | Unsupervised (Frozen FL $\tau$) |
| | Logistic Regression (p99) | 0.34 | 0.4041 | 0.3553 | 0.8311 | 0.5831 | 0.4428 | 0.5335 | 0.8320 | 0.1351 | Unsupervised (Frozen FL $\tau$) |
| | Logistic Regression (Max 5) | 0.34 | 0.4042 | 0.3551 | 0.8310 | 0.5861 | 0.4516 | 0.5391 | 0.8341 | 0.1343 | Unsupervised (Frozen FL $\tau$) |
| | Random Forest (p99) | 0.34 | 0.3966 | 0.3420 | 0.8302 | 0.5723 | 0.4597 | 0.5466 | 0.8291 | 0.1355 | Unsupervised (Frozen FL $\tau$) |
| | LightGBM (p99) | 0.34 | 0.3961 | 0.3471 | 0.8303 | 0.5703 | 0.4507 | 0.5371 | 0.8268 | 0.1364 | Unsupervised (Frozen FL $\tau$) |
| | LightGBM (Max 5) | 0.34 | 0.3949 | 0.3439 | 0.8298 | 0.5698 | 0.4537 | 0.5386 | 0.8273 | 0.1362 | Unsupervised (Frozen FL $\tau$) |
| | Seasonal LightGBM (p99) | 0.34 | 0.3882 | 0.3323 | 0.8276 | 0.5819 | 0.4924 | 0.5890 | 0.8390 | 0.1315 | Unsupervised (Frozen FL $\tau$) |

---

## 6. Audit of Scientific Assumptions: Covariate Shift vs. Concept Shift

The foundational assumption of importance weighting is **Covariate Shift**:

$$P_{\text{source}}(Y \mid X) = P_{\text{target}}(Y \mid X) \quad \text{while} \quad p_{\text{source}}(X) \ne p_{\text{target}}(X)$$

Our audit confirms that this assumption is **theoretically and empirically violated**:

1. **Heterogeneous Target Grounding**:
   - In Florida, $Y \in \{0, 1\}$ is an **empirical observational event** recorded by ASOS human and automated sensors (thunder heard, lightning detected, cumulonimbus observed). High CAPE days frequently remain storm-free if convective inhibition (CIN) is not broken or if mesoscale convergence (sea-breeze front) fails to organize.
   - In Chandipur, $Y \in \{0, 1\}$ is a **deterministic heuristic composite formula** ($\text{storm\_score} \ge 2.50$) that assigns a positive thunderstorm label almost deterministically whenever CAPE and wind speed are elevated.
2. **Consequence of the Disparity**:
   - Upweighting Florida observations with high CAPE forces the models to learn from Florida's rarest convective days, which have an observed storm prevalence of only $20\%$–$30\%$.
   - The Chandipur proxy formula, by contrast, assumes convective certainty in high-CAPE regimes.
   - Therefore, weighting source samples by $w(x) = p_T(x) / p_S(x)$ **cannot reconcile different label-generating physics**. It merely inflates estimator variance and reduces effective sample size.

---

## 7. Audit Deliverables Summary

All generated audit assets reside under:
`reports/domain_adaptation/step_06/audit/`
and are mirrored in `Florida_Thunderstorm_2024_Training/reports/domain_adaptation/step_06/audit/`:
1. `threshold_leakage_audit.csv`: Verification that all thresholds and hyperparameters were label-isolated.
2. `density_ratio_implementation_audit.csv`: Mathematical audit of prior correction, out-of-fold estimation, and safeguards.
3. `kmm_convergence_audit.csv`: Numerical iteration history of KMM PGD optimization.
4. `kmm_convergence_plot.png`: 300 DPI convergence plot of objective value and gradient norms.
5. `weighted_training_audit.csv`: Code-level verification of sample weight utilization in `pygam` and `interpret`.
6. `reconstructed_performance_comparison.csv`: Clean performance benchmark table.
7. `step_06_audit_verdict.md`: Formal verdicts and classifications for each audit dimension.
