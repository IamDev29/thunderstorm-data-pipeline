# Step 7 — Physically Constrained Feature Adaptation: Methodology

## 1. Executive Summary and Motivation

In the Thunderstorm Domain Adaptation Framework (TDAF), previous steps investigated whether domain transfer could be achieved through correlation alignment (Step 5 CORAL) or source density-ratio reweighting (Step 6 / 6.1 Importance Weighting):
- **Step 5 (CORAL)**: Linear covariance alignment applied a full affine rotation matrix $A \in \mathbb{R}^{6 \times 6}$ to Chandipur predictors ($X_T \mapsto X_T A$). Because meteorological variables have asymmetric hard physical bounds ($x \ge 0$, $[0, 1]$), linear rotation created **37.6% negative CAPE** (down to $-1,745\text{ J/kg}$) and **46.0% negative CIN**, distorting feature semantics and degrading both `pyGAM` and `InterpretML` EBM transfer performance.
- **Step 6 / 6.1 (Importance Weighting)**: Reweighting source observations failed because Florida simply lacks support in the extreme convective regime ($\text{CAPE} > 7,240\text{ J/kg}$, where Chandipur exhibits observations up to $9,906\text{ J/kg}$). The covariate shift assumption $P_T(x) \ll P_S(x)$ was violated.
- **Step 4 (CAPE Clipping Reference)**: Simply clamping CAPE at the Florida training boundary ($7,240\text{ J/kg}$) prevented spline divergence, producing stable baseline transfer.

**Step 7 Hypothesis**: Non-linear, univariate, monotonic transformations that strictly respect physical boundaries ($x \ge 0$, $[0, 1]$), preserve monotonic ordering ($\rho_S = +1.0$), and compress extreme convective tails into the source training support can resolve pyGAM spline instability without degrading kinematic predictors or violating physical laws.

---

## 2. Mathematical Formulations of Experimental Conditions

Five experimental conditions were formulated, implemented, and audited under strict zero-shot target-label isolation:

### Condition B1: Identity (Raw Control)
Direct zero-shot transfer without any modification of target features:
$$\tilde{x}_{j} = x_{j}, \quad \forall j \in \{1, \dots, 6\}$$
Serves as the unadapted reference baseline established in Step 2.

### Condition B2: Robust Quantile Scaling (RQS) with Clamping
Maps target feature values using robust interquartile scaling anchored to source statistics, with physical zero-floor clamping:
$$\tilde{x}_{j} = \max\left(0, \, q_{50,S}^{(j)} + \frac{x_{j} - q_{50,T}^{(j)}}{\text{IQR}_{T}^{(j)}} \cdot \text{IQR}_{S}^{(j)}\right)$$
For cloud cover, bounding is clamped to $[0, 1]$.
*Theoretical limitation*: Truncating at physical bounds or shifting medians alters extreme quantiles and can introduce rank ties at boundary cutoffs.

### Condition B3: Bounded Monotonic Transformation (BMT — Tail Compression)
A strictly monotonic, concavely bounded logarithmic compression for unbounded positive predictors, mapping extreme target values into the Florida support while preserving exact rank order:
For $\text{CAPE}, \text{CIN}, \text{Wind}_{10\text{m}}, \text{Wind}_{850\text{hPa}}, \text{Shear}$:
$$\tilde{x}_{j} = x_{\max, S}^{(j)} \cdot \frac{\ln(1 + \gamma_j \, x_j)}{\ln(1 + \gamma_j \, x_{\max, T}^{(j)})}$$
where $\gamma_j = 1 / q_{50, T}^{(j)}$ scales the non-linear curvature according to the target median.
- **Physical boundary preservation**: Since $x_j \ge 0 \implies \ln(1 + \gamma_j x_j) \ge 0$, non-negativity is preserved for all non-negative predictors.
- **Upper support bounding**: At $x_j = x_{\max, T}^{(j)}$, $\tilde{x}_j = x_{\max, S}^{(j)}$. Thus, no transformed value can ever exceed the maximum value observed in the Florida training set.
- **Monotonicity**: $\frac{d\tilde{x}_j}{dx_j} > 0$ for all $x_j \ge 0$. The Spearman rank correlation between raw and transformed features is identically $+1.000$.
- Cloud cover fraction (already strictly bounded in $[0, 1]$) is left unchanged: $\tilde{x}_{\text{cloud}} = x_{\text{cloud}}$.

### Condition B4: Selective Transformation (ST — Convective Thermodynamics Only)
Applies the BMT log-compression **exclusively to thermodynamic instability features (CAPE and CIN)**, leaving kinematic predictors (Bulk Shear, 10m Wind Speed, 850hPa Wind Speed) and Total Cloud Cover **100% untouched**:
$$\tilde{x}_{\text{CAPE}} = \text{BMT}(x_{\text{CAPE}}), \quad \tilde{x}_{\text{CIN}} = \text{BMT}(x_{\text{CIN}})$$
$$\tilde{x}_{\text{shear}} = x_{\text{shear}}, \quad \tilde{x}_{\text{cloud}} = x_{\text{cloud}}, \quad \tilde{x}_{\text{wind10m}} = x_{\text{wind10m}}, \quad \tilde{x}_{\text{wind850}} = x_{\text{wind850}}$$
*Physical rationale*: As established in Step 3 diagnostics, kinematic features in Chandipur ($10\text{m}$ wind, $850\text{hPa}$ wind, and bulk shear) fall completely *within* the Florida support ($x_{T} \le x_{\max, S}$). Compressing kinematics unnecessarily distorts physically meaningful shear thresholds required for thunderstorm organization.

### Condition B5: CAPE Boundary Clipping at $7,240\text{ J/kg}$ (Step 4 Reference)
The reference benchmark established in Step 4:
$$\tilde{x}_{\text{CAPE}} = \min(x_{\text{CAPE}}, 7240.0), \quad \tilde{x}_{j} = x_{j} \quad (\forall j \ne \text{CAPE})$$

---

## 3. Strict Scientific Isolation Protocol

To ensure methodological validity and eliminate potential leakage:
1. **Target Labels Quarantined**: Chandipur proxy labels (`storm_label_score`) were never used during transformation derivation, parameter estimation, model training, or threshold selection.
2. **Deterministic & Frozen Models**: Both Florida native models (`pyGAM` LogisticGAM and `InterpretML` ExplainableBoostingClassifier) were loaded from frozen disk checkpoints and never retrained, fine-tuned, or re-weighted.
3. **Threshold Protocol**: Primary metrics are reported at the source-optimal operational thresholds ($\tau = 0.33$ for pyGAM, $\tau = 0.34$ for EBM) determined solely on Florida validation data, as well as the standard threshold $\tau = 0.50$.
4. **Physical Audit Checkpoints**: Every transformed target dataset was audited for:
   - $\text{CAPE} \ge 0$
   - $\text{CIN} \ge 0$
   - $\text{Bulk Shear} \ge 0$
   - $\text{Wind Speeds} \ge 0$
   - $\text{Cloud Cover} \in [0, 1]$
   - Zero $\text{NaN}$ or $\pm\infty$ values
   - Pearson and Spearman rank correlation preservation against raw target features.
