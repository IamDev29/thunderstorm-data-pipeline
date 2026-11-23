# TDAF Step 2: Baseline Reproduction & Integrity Verification

**Project**: Thunderstorm Domain Adaptation Framework (TDAF)  
**Step**: 02 — Reconstruct and Freeze the Baseline Reference  
**Timestamp**: 2026-11-17T20:05:00+05:30  
**Status**: **VERIFIED — ZERO DRIFT (0.000000 Error)**

---

## 1. Objective of the Reproduction Audit

Before any domain adaptation method (such as CORAL, optimal transport, kernel mean matching, or quantile alignment) is developed or tested in Step 3+, this audit verifies that the **unadapted baseline reference models and predictions** can be reconstructed with absolute mathematical fidelity from the frozen source models and frozen datasets.

This check ensures:
1. The pre-trained model files (`native_gam_model.pkl` and `native_ebm_model.pkl`) have remained uncorrupted and unaltered.
2. The Florida test partition (Oct–Dec 2024, $N=28,073$) and Chandipur target dataset (2022–2025, $N=23,376$) have remained identical.
3. Feature alignment, transformations, and probability calibrations replicate previous experiments without data leakage.
4. All evaluation metrics reproduce previous benchmark values with zero numerical drift.

---

## 2. Side-by-Side Reproduction Verification Tables

### 2.1 Florida Test Set (Oct–Dec 2024, $N=28,073$, Positives = 1,918, Prev = 6.8322%)
**Ground Truth**: Rigorously audited NOAA ASOS station observations (`target_asos_strict_crit_c`).

| Model | Metric | Previously Reported (Phase 2B) | Step 2 Reconstructed | Absolute Difference | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Native pyGAM** ($\tau=0.33$) | True Positives (TP) | 848 | 848 | 0 | **Exact Match** |
| | False Positives (FP) | 1,385 | 1,385 | 0 | **Exact Match** |
| | True Negatives (TN) | 24,770 | 24,770 | 0 | **Exact Match** |
| | False Negatives (FN) | 1,070 | 1,070 | 0 | **Exact Match** |
| | Accuracy | 0.912549 | 0.912549 | 0.000000 | **Exact Match** |
| | Precision | 0.379758 | 0.379758 | 0.000000 | **Exact Match** |
| | Recall | 0.442127 | 0.442127 | 0.000000 | **Exact Match** |
| | F1-Score | 0.408576 | 0.408576 | 0.000000 | **Exact Match** |
| | PR-AUC (Average Precision) | 0.336215 | 0.336215 | 0.000000 | **Exact Match** |
| | ROC-AUC | 0.828762 | 0.828762 | 0.000000 | **Exact Match** |
| | Brier Score | 0.055595 | 0.055595 | 0.000000 | **Exact Match** |
| | Log Loss | 0.205998 | 0.205998 | 0.000000 | **Exact Match** |
| **Native InterpretML EBM** ($\tau=0.34$) | True Positives (TP) | 871 | 871 | 0 | **Exact Match** |
| | False Positives (FP) | 1,563 | 1,563 | 0 | **Exact Match** |
| | True Negatives (TN) | 24,592 | 24,592 | 0 | **Exact Match** |
| | False Negatives (FN) | 1,047 | 1,047 | 0 | **Exact Match** |
| | Accuracy | 0.907028 | 0.907028 | 0.000000 | **Exact Match** |
| | Precision | 0.357847 | 0.357847 | 0.000000 | **Exact Match** |
| | Recall | 0.454119 | 0.454119 | 0.000000 | **Exact Match** |
| | F1-Score | 0.400276 | 0.400276 | 0.000000 | **Exact Match** |
| | PR-AUC (Average Precision) | 0.351314 | 0.351314 | 0.000000 | **Exact Match** |
| | ROC-AUC | 0.830433 | 0.830433 | 0.000000 | **Exact Match** |
| | Brier Score | 0.056448 | 0.056448 | 0.000000 | **Exact Match** |
| | Log Loss | 0.206581 | 0.206581 | 0.000000 | **Exact Match** |

---

### 2.2 Chandipur Zero-Shot Transfer ($N=23,376$, Positives = 5,462, Prev = 23.3658%)
**Reference Target**: Deterministic ERA5-derived heuristic storm score label (`storm_label_score` $\ge 2.50$).

| Model | Threshold / Setting | Metric | Previously Reported (Phase 3 Native) | Step 2 Reconstructed | Absolute Difference | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Native pyGAM** | $\tau=0.33$ (FL Optimal) | TP / FP / TN / FN | 4569 / 4836 / 13078 / 893 | 4569 / 4836 / 13078 / 893 | 0 / 0 / 0 / 0 | **Exact Match** |
| | | Accuracy | 0.754920 | 0.754920 | 0.000000 | **Exact Match** |
| | | Precision | 0.485805 | 0.485805 | 0.000000 | **Exact Match** |
| | | Recall | 0.836507 | 0.836507 | 0.000000 | **Exact Match** |
| | | F1-Score | 0.614650 | 0.614650 | 0.000000 | **Exact Match** |
| | | Cohen's Kappa | 0.452916 | 0.452916 | 0.000000 | **Exact Match** |
| | | MCC | 0.488897 | 0.488897 | 0.000000 | **Exact Match** |
| | | PR-AUC | 0.552264 | 0.552264 | 0.000000 | **Exact Match** |
| | | ROC-AUC | 0.849445 | 0.849445 | 0.000000 | **Exact Match** |
| | | Brier Score | 0.130084 | 0.130084 | 0.000000 | **Exact Match** |
| | | Log Loss | 0.391486 | 0.391486 | 0.000000 | **Exact Match** |
| **Native pyGAM** | $\tau=0.50$ (Standard) | F1-Score | 0.516982 | 0.516982 | 0.000000 | **Exact Match** |
| | | Accuracy | 0.793763 | 0.793763 | 0.000000 | **Exact Match** |
| **Native InterpretML EBM** | $\tau=0.34$ (FL Optimal) | TP / FP / TN / FN | 4332 / 4506 / 13408 / 1130 | 4332 / 4506 / 13408 / 1130 | 0 / 0 / 0 / 0 | **Exact Match** |
| | | Accuracy | 0.758898 | 0.758898 | 0.000000 | **Exact Match** |
| | | Precision | 0.490156 | 0.490156 | 0.000000 | **Exact Match** |
| | | Recall | 0.793116 | 0.793116 | 0.000000 | **Exact Match** |
| | | F1-Score | 0.605874 | 0.605874 | 0.000000 | **Exact Match** |
| | | Cohen's Kappa | 0.445813 | 0.445813 | 0.000000 | **Exact Match** |
| | | MCC | 0.472614 | 0.472614 | 0.000000 | **Exact Match** |
| | | PR-AUC | 0.613421 | 0.613421 | 0.000000 | **Exact Match** |
| | | ROC-AUC | 0.857167 | 0.857167 | 0.000000 | **Exact Match** |
| | | Brier Score | 0.126992 | 0.126992 | 0.000000 | **Exact Match** |
| | | Log Loss | 0.385558 | 0.385558 | 0.000000 | **Exact Match** |
| **Native InterpretML EBM** | $\tau=0.50$ (Standard) | F1-Score | 0.548234 | 0.548234 | 0.000000 | **Exact Match** |
| | | Accuracy | 0.809078 | 0.809078 | 0.000000 | **Exact Match** |

---

## 3. Data Integrity & Leakage Prevention Confirmation

1. **Chronological Splitting Preserved**:
   - The Florida test set remains strictly isolated to months 10, 11, and 12 (October through December 2024).
   - No features or statistics from August–September (Validation) or October–December (Test) were exposed to the models.
2. **Chandipur Unsupervised Status Verified**:
   - The models did not train on any Chandipur data (neither features nor proxy labels).
   - Predictions were generated via zero-shot feed-forward evaluation.
3. **Artifact Immutability**:
   - Original model files in `reports/ebm_gam_florida_native/` were opened in read-only binary mode (`rb`).
   - Original datasets in `data/processed/asos_parser_fixed/` and `MET_DATA_with_storm_score_label.csv` were untouched.
   - All newly generated files are cleanly stored in `reports/domain_adaptation/step_02/`.

---

## 4. Conclusion

The baseline reconstruction is complete, verified, and frozen. Every single evaluation metric across all 28,073 Florida test observations and all 23,376 Chandipur observations matches previously published values with zero numerical discrepancy. All downstream domain adaptation experiments in Step 3+ can measure improvement against this benchmark.
