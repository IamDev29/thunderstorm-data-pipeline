# Step 7 — Model Interpretability and Partial Dependence Analysis

## 1. Interpretability Mechanics and Spline Extrapolation

One of the foundational requirements of the Thunderstorm Domain Adaptation Framework (TDAF) is the preservation of intrinsic model interpretability. The native Florida models are:
1. **pyGAM (LogisticGAM)**: A Generalized Additive Model representing the log-odds of convective initiation as a sum of smooth cubic B-spline functions:
   $$\log\left(\frac{p}{1-p}\right) = \beta_0 + \sum_{j=1}^{6} f_j(x_j)$$
2. **InterpretML EBM (Explainable Boosting Classifier)**: A tree-based Generalized Additive Model with pairwise interaction terms (GA2M):
   $$\log\left(\frac{p}{1-p}\right) = \beta_0 + \sum_{j=1}^{6} f_j(x_j) + \sum_{j < k} f_{jk}(x_j, x_k)$$

### The Spline Breakdown Mechanism in Unadapted pyGAM
As discovered in Step 3 and Step 4, cubic B-spline bases in `pyGAM` are anchored to knots placed over the empirical support of the Florida training data ($x_{\max} = 7,240.0\text{ J/kg}$). 
- When evaluated at raw Chandipur CAPE values exceeding $7,240\text{ J/kg}$ (up to $9,906\text{ J/kg}$), the cubic polynomials in the outer interval extrapolate wildly downward due to boundary derivative constraints.
- Consequently, at extreme convective instability ($\text{CAPE} = 9,906\text{ J/kg}$), the raw pyGAM model predicts a convective probability of **$p \approx 0.0039$**, which is physically absurd and represents a catastrophic failure of extrapolation.

In contrast, **InterpretML EBM** utilizes decision-tree stumps. Points exceeding the outermost split threshold simply land in the terminal leaf, producing an asymptotic plateau rather than a polynomial collapse.

---

## 2. Response Curve Behavior Under Transformed Inputs

The diagnostic visualization `interpretability_response_curves.png` illustrates the partial dependence functions $f_j(x_j)$ across the dynamic range of CAPE, CIN, and Bulk Shear under each adaptation condition:

```
[Interpretability Response Curves Visualized at 300 DPI]
File: reports/domain_adaptation/step_07/interpretability_response_curves.png
```

### 1. CAPE Response Curves ($0$ to $10,000\text{ J/kg}$):
- **Raw pyGAM (B1)**: Rapidly climbs to peak log-odds around $\text{CAPE} \approx 4,000\text{ J/kg}$, remains elevated until $7,240\text{ J/kg}$, and then abruptly collapses toward negative infinity as the polynomial spline diverges.
- **CAPE Boundary Clipping at $7,240\text{ J/kg}$ (B5)**: Hard clamps all inputs at $7,240\text{ J/kg}$, holding the log-odds constant at the boundary value ($f(\text{CAPE}) \approx +1.8$). However, it creates a non-differentiable sharp slope discontinuity at $7,240\text{ J/kg}$.
- **Bounded Monotonic Transformation (B3 & B4)**: Maps $[0, 9906]$ smoothly and monotonically into $[0, 7240]$ via logarithmic curvature. As a result:
  - The model traverses the full monotonic ascent of the spline without ever falling off the outer knot boundary.
  - The response curve plateaus smoothly at high CAPE, reflecting physical saturation of convective potential energy without artificial step discontinuities.
  - Probability collapse is **100% eliminated** across the entire target domain.

### 2. CIN Response Curves ($0$ to $1,000\text{ J/kg}$):
- For CIN, both pyGAM and EBM exhibit an inhibitory response curve ($f(\text{CIN}) < 0$ as CIN increases).
- In B1 and B5, high CIN in Chandipur ($> 500\text{ J/kg}$) drives strong suppression of convection, which is physically consistent with capping inversions.
- In B4, selective logarithmic compression scales CIN moderately, allowing moderate convective signals to penetrate during monsoon breaks while maintaining convective suppression under extreme capping.

### 3. Kinematic Predictor Preservation (Bulk Shear):
- In B3 (all-feature BMT), compressing bulk shear artificially shifted moderate Chandipur shear values ($15\text{–}25\text{ m/s}$) to lower effective values ($10\text{–}18\text{ m/s}$), reducing the model's ability to recognize organized multicell and supercell storm structures.
- In **B4 (Selective Transformation)**, leaving bulk shear and wind speeds **100% untouched** preserved the exact Florida kinematic sensitivities, allowing the models to fully utilize deep tropospheric shear cues for thunderstorm discrimination.

---

## 3. Preservation of Interpretability Summary

| Dimension | Raw (B1) | BMT All (B3) | Selective B4 (CAPE/CIN) | Clipping B5 |
| :--- | :---: | :---: | :---: | :---: |
| **Spline Extrapolation Stability** | Broken (Collapses) | Fully Stabilized | **Fully Stabilized** | Stabilized (Discontinuous) |
| **EBM Monotonic Saturation** | Intrinsic | Preserved | **Preserved** | Preserved |
| **Kinematic Feature Semantics** | Preserved | Artificially Shifted | **100% Preserved** | 100% Preserved |
| **Physical Derivative Sign ($\frac{\partial p}{\partial \text{CAPE}} \ge 0$)** | Violated at $> 7,240$ | Maintained Everywhere | **Maintained Everywhere** | Maintained Everywhere |
| **Interpretability Fidelity** | High (within domain) | Moderate (all scaled) | **Optimal (Thermodynamic fix only)** | High |
