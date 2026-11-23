# Step 7.1 — Independent Transformation Compatibility and Scientific Validity Audit

## 1. Executive Summary

This independent audit investigates whether Condition B4 (Selective Transformation of CAPE and CIN), introduced in Step 7, is mathematically and scientifically compatible with the frozen Florida-trained Generalized Additive Model (`pyGAM`) and Explainable Boosting Machine (`InterpretML` EBM).

Step 7 reported substantial improvements on the Chandipur proxy benchmark under B4 (pyGAM PR-AUC rose from $0.5523$ to $0.6675$). This audit evaluated whether this gain represents:
1. A genuine domain adaptation breakthrough that preserves physical semantics, OR
2. An asymmetric feature-semantic mismatch where transformed inputs are evaluated against untransformed model splines/bins, OR
3. A mathematically rigorous transformation that holds when applied symmetrically.

### Key Audit Findings:
1. **Target-Only B4 is Asymmetric**: The original frozen models were trained on **raw physical units ($\text{J/kg}$)** with cubic spline knots and tree splits anchored to raw Florida distributions. Applying B4 *only* to target inference data creates a feature-semantic distortion: an extreme Chandipur sounding ($9,906\text{ J/kg}$) is presented to the frozen model as $3,696.5$, which the model interprets as a moderate-to-high Florida sea-breeze event. This stabilizes pyGAM's cubic spline extrapolation, but alters the physical rate of change $\frac{\partial p}{\partial \text{CAPE}}$.
2. **Symmetric Retraining Resolves the Semantic Mismatch**: When the exact B4 transformation is applied **symmetrically** to both Florida training data and Chandipur target data:
   - **`pyGAM`**: Achieves **$\text{PR-AUC} = 0.5999$** and **$\text{ROC-AUC} = 0.8540$** on Chandipur (a solid **$+0.0476$ PR-AUC gain** over raw baseline $0.5523$ and clipping $0.5526$) while **identically preserving Florida holdout performance** ($\text{PR-AUC} = 0.3337$ vs $0.3362$, $\text{ROC-AUC} = 0.8291$ vs $0.8288$).
   - **`InterpretML` EBM**: Because decision tree splits are purely ordinal and invariant under strictly monotonic univariate transformations, **retrained Symmetric EBM is mathematically and numerically isomorphic to the Raw EBM** ($\text{PR-AUC} = 0.6134$, $\text{ROC-AUC} = 0.8572$ on Chandipur; $\text{PR-AUC} = 0.3513$, $\text{ROC-AUC} = 0.8304$ on Florida test). The higher PR-AUC of $0.6659$ observed in Step 7 was entirely an artifact of *asymmetric evaluation* (evaluating compressed inputs against fixed raw tree bin cuts).
3. **Correlation Invariance Verified**: Spearman rank correlation and Kendall's Tau are strictly preserved ($\Delta R = 0.000000$), confirming rank ordering invariance.

---

## 2. Reconstructed B4 Transformation

The implementation in `run_step_07_physically_constrained_adaptation.py` was inspected and verified:
- **Surface CAPE**: $\tilde{x}_{\text{CAPE}} = \alpha_{\text{CAPE}} \cdot \ln(1 + x_{\text{CAPE}} / 1000.0)$ with $\alpha_{\text{CAPE}} = 1547.1097$.
- **Boundary CIN**: $\tilde{x}_{\text{CIN}} = \alpha_{\text{CIN}} \cdot \ln(1 + x_{\text{CIN}} / 100.0)$ with $\alpha_{\text{CIN}} = 220.0598$.
- **Kinematics & Clouds**: Identity mapping ($\tilde{x}_j = x_j$).
- **Parameters**: Estimated strictly from unsupervised 95th percentiles ($q_{95, S}$ from Florida train; $q_{95, T}$ from Chandipur predictors).
- **Target Label Quarantine**: **100% CLEAN**. Chandipur proxy labels were never used in any parameter estimation.
- **Invertibility**: Analytically invertible with machine precision error ($< 2.73 \times 10^{-12}\text{ J/kg}$).

Detailed mathematical derivations and properties are documented in [b4_transformation_formula.md](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/domain_adaptation/step_07/audit/b4_transformation_formula.md).

---

## 3. Critical Audit: Frozen Model Input Compatibility

The table below contrasts the representation expected by the frozen models versus the representation delivered by B4:

| Feature | Florida Training Representation | Chandipur B4 Inference Representation | Compatibility Status | Semantic Distortion Note |
| :--- | :--- | :--- | :--- | :--- |
| **Surface CAPE** | Raw Surface CAPE ($\text{J/kg}$) in $[0.0, 7240.0]$ | Log-compressed index in $[0.0, 3696.5]$ | **INCOMPATIBLE (Semantic Mismatch)** | Frozen model's spline knots were trained on raw $\text{J/kg}$. B4 compresses $9906\text{ J/kg}$ to $3696.5$, altering the physical derivative $\frac{\partial p}{\partial \text{CAPE}}$. |
| **Boundary CIN** | Raw Boundary CIN ($\text{J/kg}$) in $[0.0, 997.4]$ | Log-compressed index in $[0.0, 527.5]$ | **INCOMPATIBLE (Semantic Mismatch)** | Model trained on physical $\text{J/kg}$. B4 applies logarithmic curvature, scaling moderate inhibition non-linearly. |
| **Bulk Shear** | Raw Bulk Shear ($\text{m/s}$) in $[0.05, 77.78]$ | Raw Bulk Shear ($\text{m/s}$) in $[0.11, 56.45]$ | **COMPATIBLE (Identity)** | Untouched; target values fall entirely within Florida training support. |
| **Cloud Cover** | Total Cloud Fraction in $[0.00, 1.00]$ | Total Cloud Fraction in $[0.00, 1.00]$ | **COMPATIBLE (Identity)** | Untouched; identical $[0, 1]$ bounding. |
| **10m Wind Speed**| 10m Wind Speed ($\text{m/s}$) in $[0.02, 15.07]$ | 10m Wind Speed ($\text{m/s}$) in $[0.05, 16.60]$ | **COMPATIBLE (Identity)** | Untouched; minor upper-tail coastal difference ($0.02\%$ exceedance). |
| **850hPa Wind** | 850hPa Wind Speed ($\text{m/s}$) in $[0.02, 42.33]$ | 850hPa Wind Speed ($\text{m/s}$) in $[0.05, 28.99]$ | **COMPATIBLE (Identity)** | Untouched; target values fully enveloped by Florida distribution. |

### Spline Knots and EBM Bins Inspection:
1. **`pyGAM` Spline Basis**:
   - `native_gam_model.pkl` possesses $15$ B-spline basis functions for CAPE with boundary edge knots `edge_knots_ = [0.0, 7240.0]`.
   - In raw zero-shot transfer, $82$ target observations ($0.35\%$) exceed $7,240\text{ J/kg}$ (up to $9,906\text{ J/kg}$). Because cubic polynomials extrapolate erratically beyond boundary knots, raw pyGAM collapsed to $p \approx 0.0039$ at $9,906\text{ J/kg}$.
   - Under Target-Only B4, the maximum transformed value is $3,696.5$, which places all target observations strictly inside the spline support, safely below the 8th interior knot ($3,620.0\text{ J/kg}$).
   - **Conclusion**: Target-Only B4 successfully eliminates the spline extrapolation failure, but it does so by re-mapping the physical input scale into a completely different range than the model was trained on.
2. **`InterpretML` EBM Bins**:
   - `native_ebm_model.pkl` possesses $125$ bin cuts for CAPE, with the maximum bin cut at $4,113.25\text{ J/kg}$.
   - In raw zero-shot transfer, $7.13\%$ of Chandipur observations exceed the maximum bin cut. Decision trees treat all values above the maximum cut identically (landing in the outermost leaf score), producing a constant plateau.
   - When Target-Only B4 is applied, the inputs are compressed, shifting many observations from high-CAPE leaves into lower-CAPE leaves.

```
[Diagnostic Visualizations Generated at 300 DPI]
- reports/domain_adaptation/step_07/audit/spline_knot_and_ebm_bin_compatibility.png
- reports/domain_adaptation/step_07/audit/model_input_compatibility_overview.png
- reports/domain_adaptation/step_07/audit/symmetric_vs_asymmetric_response_curves.png
```

---

## 4. Experiment A — Frozen Models with B4 (Reproduction)

Reproduction of the Step 7 target-only B4 condition using the frozen original models confirmed the reported metrics to $6$ decimal places:
- **Frozen `pyGAM` (Target-Only B4, $\tau = 0.33$)**:
  - Accuracy: $0.7783$, Precision: $0.5169$, Recall: $0.7852$, **F1: $0.6234$**
  - **PR-AUC: $0.6675$**, **ROC-AUC: $0.8725$**, Brier Score: $0.1224$
- **Frozen `InterpretML` EBM (Target-Only B4, $\tau = 0.34$)**:
  - Accuracy: $0.7847$, Precision: $0.5294$, Recall: $0.7096$, **F1: $0.6064$**
  - **PR-AUC: $0.6659$**, **ROC-AUC: $0.8624$**, Brier Score: $0.1243$

Prediction generation, threshold application, and target-label isolation are fully verified and reproducible.

---

## 5. Experiment B — Symmetric Transformation and Retraining

To resolve the semantic mismatch, independent model variants were trained where the **exact same B4 transformation was applied symmetrically to the Florida training data**:
- **Dataset**: Florida ASOS 2024 modeling dataset ($N_{\text{train}} = 66,255$, $N_{\text{val}} = 18,181$, $N_{\text{test}} = 28,073$).
- **Symmetric Models**:
  - `symmetric_gam_model.pkl`: LogisticGAM ($15$ splines, $\lambda = 1.0$) trained on B4-transformed Florida training features. Optimal validation threshold tuned on B4-transformed validation set: $\tau_{\text{opt}} = 0.33$.
  - `symmetric_ebm_model.pkl`: ExplainableBoostingClassifier ($128$ bins, $16$ outer bags, interactions $= 0$, learning rate $= 0.01$) trained on B4-transformed Florida training features. Optimal validation threshold: $\tau_{\text{opt}} = 0.34$.
- Checkpoints saved under `reports/domain_adaptation/step_07/audit/` without overwriting original native models.

### Comparative Performance Matrix:

| Model Configuration | Model Type | Evaluation Domain | Threshold ($\tau$) | PR-AUC | ROC-AUC | F1 Score | Precision | Recall | Brier Score | Compatibility Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Original Raw Baseline** | pyGAM | Florida Test (Holdout) | $0.33$ | $0.3362$ | $0.8288$ | $0.4086$ | $0.3798$ | $0.4421$ | $0.0556$ | Baseline Reference |
| **4. Symmetrically Retrained B4**| pyGAM | Florida Test (Holdout) | $0.33$ | $0.3337$ | $0.8291$ | $0.4013$ | $0.3572$ | $0.4578$ | $0.0566$ | **Fully Preserved ($< 0.003$ diff)** |
| **1. Original Raw Baseline** | EBM | Florida Test (Holdout) | $0.34$ | $0.3513$ | $0.8304$ | $0.4003$ | $0.3578$ | $0.4541$ | $0.0564$ | Baseline Reference |
| **4. Symmetrically Retrained B4**| EBM | Florida Test (Holdout) | $0.34$ | **$0.3513$** | **$0.8304$** | **$0.4003$** | **$0.3578$** | **$0.4541$** | **$0.0564$** | **Identically Isomorphic** |
| | | | | | | | | | | |
| **1. Original Raw Baseline** | pyGAM | Chandipur Target | $0.33$ | $0.5523$ | $0.8494$ | $0.6147$ | $0.4858$ | $0.8365$ | $0.1301$ | Severe Spline Extrapolation |
| **2. CAPE Clipping (B5)** | pyGAM | Chandipur Target | $0.33$ | $0.5526$ | $0.8502$ | $0.6147$ | $0.4858$ | $0.8365$ | $0.1299$ | Minimal-Intervention Reference |
| **3. Target-Only B4 (Step 7)** | pyGAM | Chandipur Target | $0.33$ | $0.6675$ | $0.8725$ | $0.6234$ | $0.5169$ | $0.7852$ | $0.1224$ | Asymmetric / Semantic Distortion |
| **4. Symmetrically Retrained B4**| pyGAM | Chandipur Target | $0.33$ | **$0.5999$** | **$0.8540$** | **$0.6086$** | **$0.4837$** | **$0.8204$** | **$0.1279$** | **Mathematically & Scientifically Valid** |
| | | | | | | | | | | |
| **1. Original Raw Baseline** | EBM | Chandipur Target | $0.34$ | $0.6134$ | $0.8572$ | $0.6059$ | $0.4902$ | $0.7931$ | $0.1270$ | Plateauing Tree Leaves |
| **2. CAPE Clipping (B5)** | EBM | Chandipur Target | $0.34$ | $0.6134$ | $0.8572$ | $0.6059$ | $0.4902$ | $0.7931$ | $0.1270$ | Identical to Raw |
| **3. Target-Only B4 (Step 7)** | EBM | Chandipur Target | $0.34$ | $0.6659$ | $0.8624$ | $0.6064$ | $0.5294$ | $0.7096$ | $0.1243$ | Asymmetric / Bin-Shift Artifact |
| **4. Symmetrically Retrained B4**| EBM | Chandipur Target | $0.34$ | **$0.6134$** | **$0.8572$** | **$0.6059$** | **$0.4902$** | **$0.7931$** | **$0.1270$** | **Tree Monotonic Isomorphism** |

### Fundamental Theoretical Insights from Experiment B:
1. **`pyGAM` Gains from Log-Compression Are Real (+0.0476 PR-AUC)**:
   In pyGAM, cubic B-splines place knots uniformly across the feature span. Under raw CAPE ($0\text{--}7240\text{ J/kg}$), knots are heavily spaced across sparse high-CAPE values, leaving fewer knots in the dense lower-CAPE regime ($0\text{--}1000\text{ J/kg}$). Symmetrically log-compressing CAPE spreads training observations much more evenly across the spline basis. As a result, the retrained symmetric pyGAM achieves **$\text{PR-AUC} = 0.5999$** on Chandipur, outperforming both the raw baseline ($0.5523$) and CAPE clipping ($0.5526$) by **$+0.0476$** without any semantic mismatch!
2. **`InterpretML` EBM is Strictly Invariant**:
   For Explainable Boosting Machines (and any decision tree ensemble), tree splits are determined purely by rank order. A strictly monotonic univariate transformation changes the numerical values of the cut points, but preserves the exact partition of training samples in every bin. Symmetrically retrained EBM produces the **exact same predictions and metrics** as raw EBM down to machine precision.
   This proves that the $+0.0525$ PR-AUC boost seen in Target-Only B4 on EBM was entirely caused by evaluating compressed inputs against raw bins (shifting high-CAPE points into lower leaves).

---

## 6. Experiment C — Invertibility and Precision Audit

Forward transformation followed by analytical inversion was executed on all $23,376$ Chandipur samples:
- **Maximum CAPE Reconstruction Error**: $2.73 \times 10^{-12}\text{ J/kg}$
- **Mean CAPE Reconstruction Error**: $1.50 \times 10^{-13}\text{ J/kg}$
- **Maximum CIN Reconstruction Error**: $7.96 \times 10^{-13}\text{ J/kg}$
- **Mean CIN Reconstruction Error**: $1.84 \times 10^{-14}\text{ J/kg}$
- **Verdict**: **PASS**. Invertibility is mathematically exact and numerically verified to machine precision.

---

## 7. Audit of Feature Correlation Claims

Step 7 reported that B4 preserves Spearman correlation exactly. This claim was verified computationally and mathematically:

| Metric | Frobenius Norm Difference | Max Absolute Difference | Mathematical Invariance Status | Audit Verdict |
| :--- | :---: | :---: | :--- | :--- |
| **Pearson Correlation ($r$)** | $0.2674$ | $0.1081$ | **NOT Invariant** (Altered by non-linear concave logarithmic curvature) | **EXPECTED CHANGE** (Log compresses extreme Euclidean distances) |
| **Spearman Rank Correlation ($\rho$)** | **$0.000000$** | **$0.000000$** | **STRICTLY INVARIANT** (Monotonic univariate transformation preserves rank order) | **VERIFIED EXACT** ($\Delta R_{\text{Spearman}} \equiv 0$) |
| **Kendall's Tau ($\tau$)** | **$0.000000$** | **$0.000000$** | **STRICTLY INVARIANT** (Preserves concordant and discordant pairs) | **VERIFIED EXACT** ($\Delta R_{\text{Kendall}} \equiv 0$) |

### Mathematical Invariance Proof:
Let $x_i$ and $x_j$ be two predictors, and let $g_i: \mathbb{R} \to \mathbb{R}$ be any strictly monotonically increasing function ($\frac{dg_i}{dx_i} > 0$).
For any two observations $a$ and $b$:
$$x_{ia} < x_{ib} \iff g_i(x_{ia}) < g_i(x_{ib})$$
Consequently, the rank assignment function $\text{rank}(g_i(x_{ik})) = \text{rank}(x_{ik})$ is an identity map. Because Spearman's $\rho$ and Kendall's $\tau$ are computed exclusively from ranks and concordance signs, they are **strictly invariant under any monotonic univariate transformation**.

**Important Physical Caveat**: While exact rank preservation ensures that relative convective risks are never inverted, **rank invariance alone does NOT guarantee atmospheric physical validity or model compatibility**. Model response functions depend on numerical values and derivatives, not merely ordinal ranks.

---

## 8. Seasonal Performance Revalidation

Seasonal evaluation confirmed that Chandipur exhibits extreme seasonal prevalence differences:
- **Monsoon (JJAS)**: $7,808$ samples, $3,761$ positives (**$48.17\%$ prevalence**)
- **Post-Monsoon (ON)**: $3,904$ samples, $422$ positives (**$10.81\%$ prevalence**)
- **Pre-Monsoon (MAM)**: $5,888$ samples, $1,216$ positives (**$20.65\%$ prevalence**)
- **Winter (DJF)**: $5,776$ samples, $63$ positives (**$1.09\%$ prevalence**)

### Seasonal PR-AUC Revalidation:

| Model Configuration | Season | Sample Count | Positive Prevalence | pyGAM PR-AUC | EBM PR-AUC | pyGAM F1 | EBM F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Original Raw Baseline** | JJAS (Monsoon) | 7,808 | 48.17% | $0.6065$ | $0.6742$ | $0.6819$ | $0.6803$ |
| **4. Symmetrically Retrained B4**| JJAS (Monsoon) | 7,808 | 48.17% | **$0.6594$** | **$0.6742$** | $0.6788$ | $0.6803$ |
| **3. Target-Only B4 (Step 7)** | JJAS (Monsoon) | 7,808 | 48.17% | $0.7234$ | $0.7282$ | $0.6880$ | $0.6808$ |
| | | | | | | | |
| **1. Original Raw Baseline** | ON (Post-monsoon) | 3,904 | 10.81% | $0.5733$ | $0.6977$ | $0.5403$ | $0.5513$ |
| **4. Symmetrically Retrained B4**| ON (Post-monsoon) | 3,904 | 10.81% | **$0.6579$** | **$0.6977$** | $0.5461$ | $0.5513$ |
| **3. Target-Only B4 (Step 7)** | ON (Post-monsoon) | 3,904 | 10.81% | $0.7520$ | $0.7903$ | $0.5782$ | $0.5885$ |
| | | | | | | | |
| **1. Original Raw Baseline** | MAM (Pre-monsoon) | 5,888 | 20.65% | $0.3820$ | $0.3616$ | $0.4612$ | $0.3832$ |
| **4. Symmetrically Retrained B4**| MAM (Pre-monsoon) | 5,888 | 20.65% | **$0.3764$** | **$0.3616$** | $0.4289$ | $0.3832$ |
| **3. Target-Only B4 (Step 7)** | MAM (Pre-monsoon) | 5,888 | 20.65% | $0.4453$ | $0.3828$ | $0.4288$ | $0.3136$ |
| | | | | | | | |
| **1. Original Raw Baseline** | DJF (Winter) | 5,776 | 1.09% | $0.1546$ | $0.2955$ | $0.1429$ | $0.3717$ |
| **4. Symmetrically Retrained B4**| DJF (Winter) | 5,776 | 1.09% | **$0.2072$** | **$0.2955$** | $0.2115$ | $0.3717$ |
| **3. Target-Only B4 (Step 7)** | DJF (Winter) | 5,776 | 1.09% | $0.2773$ | $0.3820$ | $0.2500$ | $0.4000$ |

**Seasonal Insight**: In the mathematically valid Symmetric pyGAM model, performance gains over baseline are concentrated during the convective active regimes:
- In the **Monsoon (JJAS)**, PR-AUC increases from $0.6065$ to **$0.6594$** ($+0.0529$).
- In the **Post-monsoon (ON)**, PR-AUC increases from $0.5733$ to **$0.6579$** ($+0.0846$).
- In the **Winter (DJF)**, PR-AUC increases from $0.1546$ to **$0.2072$** ($+0.0526$).
