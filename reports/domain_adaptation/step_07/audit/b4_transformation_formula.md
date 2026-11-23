# Step 7.1 — B4 Transformation Mathematical Specification and Parameter Origin

## 1. Mathematical Formulation

Condition B4 (Selective Transformation) applies non-linear, concavely bounded monotonic logarithmic compression strictly to thermodynamic instability predictors (Surface CAPE and Boundary CIN), leaving kinematic predictors (Bulk Shear, 10m Wind Speed, 850hPa Wind Speed) and Total Cloud Cover completely unaltered.

### Feature 1: Surface CAPE ($x_{\text{CAPE}} \ge 0$)
$$\tilde{x}_{\text{CAPE}} = \alpha_{\text{CAPE}} \cdot \ln\left(1 + \frac{x_{\text{CAPE}}}{1000.0}\right)$$

where:
$$\alpha_{\text{CAPE}} = \frac{q_{95, S}(\text{CAPE})}{\ln\left(1 + \frac{q_{95, T}(\text{CAPE})}{1000.0}\right)}$$

- $q_{95, S}(\text{CAPE}) = 2668.50\text{ J/kg}$ is the 95th percentile of CAPE in the Florida training split (Jan–Jul 2024, $N = 66,255$).
- $q_{95, T}(\text{CAPE}) = 4611.56\text{ J/kg}$ is the 95th percentile of CAPE in the Chandipur target dataset (2022–2025, $N = 23,376$).
- The scaling constant $\alpha_{\text{CAPE}}$ evaluates to:
$$\alpha_{\text{CAPE}} = \frac{2668.50}{\ln(1 + 4.61156)} = \frac{2668.50}{\ln(5.61156)} \approx \frac{2668.50}{1.724819} \approx \mathbf{1547.1097}$$

### Feature 2: Convective Inhibition CIN ($x_{\text{CIN}} \ge 0$)
$$\tilde{x}_{\text{CIN}} = \alpha_{\text{CIN}} \cdot \ln\left(1 + \frac{x_{\text{CIN}}}{100.0}\right)$$

where:
$$\alpha_{\text{CIN}} = \frac{q_{95, S}(\text{CIN})}{\ln\left(1 + \frac{q_{95, T}(\text{CIN})}{100.0}\right)}$$

- $q_{95, S}(\text{CIN}) = 460.63\text{ J/kg}$ is the 95th percentile of CIN in the Florida training split.
- $q_{95, T}(\text{CIN}) = 711.09\text{ J/kg}$ is the 95th percentile of CIN in the Chandipur target dataset.
- The scaling constant $\alpha_{\text{CIN}}$ evaluates to:
$$\alpha_{\text{CIN}} = \frac{460.63}{\ln(1 + 7.1109)} = \frac{460.63}{\ln(8.1109)} \approx \frac{460.63}{2.093208} \approx \mathbf{220.0598}$$

### Features 3 through 6: Kinematics and Clouds (Identity)
$$\tilde{x}_{\text{shear}} = x_{\text{shear}}, \quad \tilde{x}_{\text{cloud}} = x_{\text{cloud}}, \quad \tilde{x}_{\text{wind10m}} = x_{\text{wind10m}}, \quad \tilde{x}_{\text{wind850}} = x_{\text{wind850}}$$

---

## 2. Parameter Origin and Label Quarantine Verification

| Parameter | Value | Domain Origin | Label Access | Quarantine Status |
| :--- | :---: | :--- | :--- | :--- |
| $q_{95, S}(\text{CAPE})$ | $2668.50\text{ J/kg}$ | Florida Source Training Set ($N = 66,255$) | Unsupervised feature only | **VERIFIED CLEAN** |
| $q_{95, T}(\text{CAPE})$ | $4611.56\text{ J/kg}$ | Chandipur Target Dataset ($N = 23,376$) | Unsupervised feature only | **VERIFIED CLEAN** |
| $q_{95, S}(\text{CIN})$ | $460.63\text{ J/kg}$ | Florida Source Training Set ($N = 66,255$) | Unsupervised feature only | **VERIFIED CLEAN** |
| $q_{95, T}(\text{CIN})$ | $711.09\text{ J/kg}$ | Chandipur Target Dataset ($N = 23,376$) | Unsupervised feature only | **VERIFIED CLEAN** |
| Denominators ($1000, 100$) | Constant | Fixed physical scale constants | Fixed hyperparameter | **VERIFIED CLEAN** |

**Zero Target Label Access**: Chandipur proxy labels (`storm_label_score`) were neither accessed nor referenced during quantile computation, parameter calculation, or feature mapping.

---

## 3. Mathematical Properties

### 1. Strict Monotonicity
For any $x \ge 0$:
$$\frac{d\tilde{x}_{\text{CAPE}}}{dx_{\text{CAPE}}} = \frac{\alpha_{\text{CAPE}}}{1000.0 \cdot \left(1 + \frac{x_{\text{CAPE}}}{1000.0}\right)} = \frac{1.5471}{1 + \frac{x_{\text{CAPE}}}{1000.0}} > 0$$
$$\frac{d\tilde{x}_{\text{CIN}}}{dx_{\text{CIN}}} = \frac{\alpha_{\text{CIN}}}{100.0 \cdot \left(1 + \frac{x_{\text{CIN}}}{100.0}\right)} = \frac{2.2006}{1 + \frac{x_{\text{CIN}}}{100.0}} > 0$$
Because the first derivatives are strictly positive across the entire positive domain, the transformation is **strictly monotonically increasing**. It preserves the exact ordinal ranking of all observations.

### 2. Analytical Invertibility
The transformation is analytically invertible:
$$x_{\text{CAPE}} = 1000.0 \cdot \left(\exp\left(\frac{\tilde{x}_{\text{CAPE}}}{\alpha_{\text{CAPE}}}\right) - 1\right)$$
$$x_{\text{CIN}} = 100.0 \cdot \left(\exp\left(\frac{\tilde{x}_{\text{CIN}}}{\alpha_{\text{CIN}}}\right) - 1\right)$$
In Experiment C, applying the forward transformation and analytical inverse across all $23,376$ Chandipur samples reconstructed the original data with maximum numerical precision error of **$2.73 \times 10^{-12}\text{ J/kg}$** for CAPE and **$7.96 \times 10^{-13}\text{ J/kg}$** for CIN, establishing **exact machine-precision invertibility**.

### 3. Physical Units and Dimensionality
- The input features are physical quantities with units of Joules per kilogram ($\text{J/kg}$).
- Inside the logarithm, $x / 1000$ and $x / 100$ are dimensionless ratios.
- The multiplier $\alpha$ carries nominal units of $\text{J/kg}$ because it is scaled by the source 95th percentile.
- However, because the mapping is non-linear ($\frac{d^2\tilde{x}}{dx^2} < 0$), increments of $\Delta\tilde{x}$ do not correspond to constant increments of physical energy $\Delta x$. The transformed feature is mathematically a **non-linear thermodynamic index**, not a physical energy density.
