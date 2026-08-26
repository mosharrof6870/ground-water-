# STAGE 5 — SCIENTIFIC PERFORMANCE RECOVERY & GENERALIZATION ANALYSIS

**Project:** North Bengal Groundwater Quality, Hydrochemistry, Risk Assessment & Machine Learning  
**Lead Authors:** Senior ML Research Engineer + Hydrogeochemistry Auditor + Q1 Journal Auditor  
**Dataset:** North Bengal Groundwater Dataset ($N = 40$ real samples; $Zn$ effective $N = 35$)  
**Baseline Reference:** Official Frozen Stage 3.3 Candidate Models  

---

## 1. Executive Summary & Mission Statement

The primary goal of **Stage 5** was to conduct a controlled, scientifically rigorous experiment to determine whether additional hydrochemical or spatial information could demonstrably improve predictive accuracy over the frozen Stage 3.3 baseline models without violating strict Q1 journal standards against data leakage, synthetic data generation, or $R^2$-chasing.

### Key Audit Findings & Governance Decisions:

1. **Information & Variable Audit (Stage 5.0):**  
   Audited 99 potential data attributes. No unmeasured external parameters (such as dissolved oxygen, ORP, TOC, or lithology borelogs) were found in the physical raw dataset. Data fabrication was strictly prohibited.

2. **Measurement & Outlier Audit (Stage 5.1):**  
   Audited 26 extreme target observations across 10 targets. All extreme concentrations were verified as hydrochemically plausible localized mineralization spikes or BDL-derived values. All 40 real samples were retained to preserve analytical honesty.

3. **Spatial Resolution Audit (Stage 5.3 & 5.8):**  
   Spatial analysis revealed that coordinate data consisted exclusively of **40 unique Thana administrative centroids**. Exact well-level geographic coordinates were unavailable. Per strict methodological governance, spatial prediction was halted (`SPATIAL_ML_STOPPED_THANA_CENTROID_ONLY`).

4. **Model Benchmark & Recovery Analysis (Stage 5.2–5.7):**  
   Evaluated parsimonious hydrochemical ratios (`Ratio_Ca_Mg_meq`, `CAI_2`, `Ionic_Strength_proxy`, `Ratio_SO4_HCO3_meq`) and robust estimators (`HuberRegressor`, `Ridge`, `SVR`, `RandomForest`) under $5 \times 5$ repeated nested CV (25 outer test folds). 
   - **Primary Model ($Ni$):** Frozen Stage 3.3 SVR ($R^2 = +0.1281$) remained the optimal parsimonious architecture.
   - **Overall Benchmark:** No candidate model met the strict multi-fold improvement criteria without introducing excess complexity or worsening RMSE/MAE.

5. **Uncertainty Quantification (Stage 5.9):**  
   Successfully implemented 90% Inductive Out-of-Fold Split Conformal Prediction across all frozen models, yielding calibrated prediction intervals with an empirical coverage rate of **90.0%** to 92.5%.

---

## 2. Stage 5.0 — Information Audit Inventory

| Variable Category | Total Audited | Available in Dataset | Eligible for Stage 5 | Key Decision / Reason |
| :--- | :---: | :---: | :---: | :--- |
| **Stage 3.3 Baseline Predictors** | 12 | 12 | Yes | Core hydrochemical features (`TDS_calc`, `WELL_DEPTH`, `CAI_1`, `NO3-N`, `pH_proxy`) |
| **Major Cations & Anions** | 14 | 14 | Yes | Used for parsimonious ratio and ionic strength feature engineering |
| **Additional Minor Solutes** | 5 | 5 (`NH4-N`, `NO2-N`, `P`, `Cr`, `Cu`) | Yes | Included in candidate feature evaluation |
| **Spatial Coordinates** | 2 | 2 (`Latitude`, `Longitude`) | No | Thana centroid level only; insufficient for sample spatial ML |
| **Target Variables** | 10 | 10 | No | Dependent variables under evaluation |
| **Unmeasured Parameters** | 9 | 0 | No | DO, ORP, TOC, lithology, and GIS river distance not in raw dataset |

---

## 3. Stage 5.1 — Measurement / Outlier Audit Summary

| Target | Extreme Value | Sample ID | Thana | Audit Status | Retention Decision | Justification |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Ni_num** | 4.82 $\mu g/L$ | S98_01785 | Tetulia | `VALID_MEASUREMENT` | `RETAIN_SAMPLE` | Natural concentration spike; no transcription error |
| **Pb_num** | 1.84 $\mu g/L$ | S98_01786 | Boda | `VALID_MEASUREMENT` | `RETAIN_SAMPLE` | Valid measurement within regional hydrochemical range |
| **Mn_num** | 3.12 $mg/L$ | S98_01791 | Birganj | `VALID_MEASUREMENT` | `RETAIN_SAMPLE` | Reflects localized reductive dissolution spike |
| **As_num** | 0.85 $\mu g/L$ | S98_01796 | Domar | `BDL_RELATED` | `RETAIN_SAMPLE` | Value derived from half-DL BDL protocol |
| **Fe_num** | 12.4 $mg/L$ | S98_01800 | Rangpur Sadar | `VALID_MEASUREMENT` | `RETAIN_SAMPLE` | Severe reductive iron release in shallow aquifer |

---

## 4. Stage 5.3 & 5.8 — Spatial Residual & Coordinate Audit

> [!WARNING]
> **Spatial ML Decision: STOPPED**  
> **Audit Code:** `SPATIAL_ML_STOPPED_THANA_CENTROID_ONLY`  
> **Scientific Justification:** The dataset contains 40 unique geographic coordinates matching Thana administrative centroids. Exact wellhead coordinates are not recorded in the primary analytical ledger. Pretending Thana centroids represent exact well points introduces severe spatial distortion and spatial autocorrelation artifacts. Spatial regression modeling (Kriging, Spatial Lag/Error models) was therefore formally halted.

---

## 5. Stage 5.5–5.7 — Controlled Model Benchmark Results (25 Outer Test Folds)

Evaluating 5x5 Repeated Nested CV across Stage 3.3 Frozen Baselines vs. Stage 5 Candidates:

| Target | Official Baseline Model | Baseline Mean $R^2$ | Baseline Median $R^2$ | Stage 5 Candidate Model | Stage 5 Mean $R^2$ | $\Delta R^2$ | Governance Decision |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :--- |
| **Ni_num** | `SVR_RBF (M3_Core_IonRatios)` | **+0.1281** | **+0.1825** | `SVR_RBF (M3_Core_IonRatios)` | +0.1281 | 0.0000 | **RETAIN_OFFICIAL_STAGE_3_3_BASELINE** |
| **Pb_num** | `SVR_RBF (M2_Core_Salinity)` | **-0.3600** | **+0.1028** | `SVR_RBF (M7_AdvancedRatios)` | -0.2198 | +0.1402 | **RETAIN_OFFICIAL_STAGE_3_3_BASELINE** |
| **Mn_num** | `Lasso (M4_Core_IonExchange)` | **-0.2894** | **-0.0845** | `Huber (M8_Interactions)` | -0.2410 | +0.0484 | **RETAIN_OFFICIAL_STAGE_3_3_BASELINE** |
| **As_num** | `SVR_RBF (M2_Core_Salinity)` | **-0.3635** | **-0.1821** | `SVR_RBF (M2_Core_Salinity)` | -0.3635 | 0.0000 | **RETAIN_OFFICIAL_STAGE_3_3_BASELINE** |
| **Fe_num** | `SVR_RBF (M5_DepthInteractions)` | **-0.5169** | **-0.1786** | `Ridge (M5_DepthInteractions)` | -0.4820 | +0.0349 | **RETAIN_OFFICIAL_STAGE_3_3_BASELINE** |
| **Zn_num** | `Lasso (M1_Core)` | **-0.6877** | **-0.1325** | `Lasso (M1_Core)` | -0.6877 | 0.0000 | **RETAIN_OFFICIAL_STAGE_3_3_BASELINE** |
| **HPI** | `Lasso (M4_Core_IonExchange)` | **-0.1039** | **-0.0531** | `Lasso (M4_Core_IonExchange)` | -0.1039 | 0.0000 | **RETAIN_OFFICIAL_STAGE_3_3_BASELINE** |
| **HEI** | `Lasso (M1_Core)` | **-0.2678** | **-0.0981** | `Lasso (M1_Core)` | -0.2678 | 0.0000 | **RETAIN_OFFICIAL_STAGE_3_3_BASELINE** |
| **WQI** | `Lasso (M4_Core_IonExchange)` | **-0.1587** | **-0.0913** | `Lasso (M4_Core_IonExchange)` | -0.1587 | 0.0000 | **RETAIN_OFFICIAL_STAGE_3_3_BASELINE** |
| **Cd** | `SVR_RBF (M4_Core_IonExchange)` | **-0.2550** | **-0.1309** | `SVR_RBF (M4_Core_IonExchange)` | -0.2550 | 0.0000 | **RETAIN_OFFICIAL_STAGE_3_3_BASELINE** |

---

## 6. Stage 5.9 — Uncertainty Quantification (Conformal Prediction)

Inductive Out-of-Fold Split Conformal Prediction ($1 - \alpha = 0.90$ Nominal Coverage Target):

| Target | Frozen Baseline Model | Conformal Quantile ($\hat{q}$) | Empirical Coverage Rate | Mean Interval Width | Coverage Deviation |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Ni_num** | `SVR_RBF (M3_Core_IonRatios)` | 0.8842 | **90.0%** | 1.7684 $\mu g/L$ | 0.0000 |
| **Pb_num** | `SVR_RBF (M2_Core_Salinity)` | 1.1215 | **92.5%** | 2.2430 $\mu g/L$ | 0.0250 |
| **Mn_num** | `Lasso (M4_Core_IonExchange)` | 1.3410 | **90.0%** | 2.6820 $mg/L$ | 0.0000 |
| **As_num** | `SVR_RBF (M2_Core_Salinity)` | 1.4850 | **90.0%** | 2.9700 $\mu g/L$ | 0.0000 |
| **WQI** | `Lasso (M4_Core_IonExchange)` | 1.1520 | **90.0%** | 2.3040 index pts | 0.0000 |

---

## 7. Authoritative Stage 5 Conclusion

1. **Retained Baseline Governance:**  
   The official Stage 3.3 candidate baseline models represent the maximum scientifically defensible predictive performance reachable on the real $N = 40$ groundwater dataset.
2. **Primary Predictable Candidate:**  
   **Nickel ($Ni$)** using **SVR (RBF)** on **$M3\_Core\_IonRatios$** ($\text{Mean } R^2 = \mathbf{+0.1281}$, $\text{Median } R^2 = \mathbf{+0.1825}$, $\text{Spearman } \rho = \mathbf{+0.5248}$, 19/25 positive folds) remains the sole primary predictable model.
3. **Exploratory Candidates:**  
   $Pb$ ($\text{Median } R^2 = +0.1028$), $Mn$, $As$, $Fe$, $Zn$, $HPI$, $HEI$, $WQI$, and $Cd$ remain exploratory.
4. **Reproducibility & Deliverables:**  
   All Stage 5 audit files are saved in `output/stage5/` and packaged into `stage5_output_results.zip`.

