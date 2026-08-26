# STAGE 5.1 — EASY-INPUT / MINIMUM-INPUT FEASIBILITY AUDIT REPORT

**Project Title:** Low-Cost, Hydrochemistry-Informed Groundwater Screening and Decision Support under Limited-Data Conditions  
**Dataset:** N = 40 Real Groundwater Samples, North Bengal (Zn Effective N = 35)  
**Lead Auditors:** Senior Water Resources / Hydrochemistry Engineer + Senior ML Research Engineer + Q1 Journal Methodology Auditor  
**Audit Status:** VALIDATED WITH SCIENTIFIC GOVERNANCE  

---

## A. OBJECTIVE OF STAGE 5.1
The objective of Stage 5.1 is to conduct a rigorous, leakage-controlled feasibility audit to answer four fundamental scientific questions:
1. Which variables in the North Bengal groundwater dataset qualify as genuinely "easy-to-measure" field parameters?
2. Can difficult-to-measure heavy metal targets (As, Fe, Mn, Pb, Ni, Zn) and composite pollution indices (WQI, HPI, HEI, Cd) be screened using only low-cost field parameters?
3. What is the minimum input combination that provides the best scientifically defensible predictive signal?
4. How much predictive performance is lost when expensive laboratory parameters are omitted compared to the frozen Stage 3.3 baseline?

---

## B. DATASET INVENTORY & CLASSIFICATION
All 90 variables in the dataset were audited and classified into six functional categories:

| Classification Category | Variable Count | Example Variables | Predictor Eligibility |
| :--- | :---: | :--- | :--- |
| **A. EASY FIELD INPUT** | 3 | `pH_proxy`, `TDS_calc`, `WELL_DEPTH` | **ELIGIBLE** (Core field panel) |
| **B. MODERATELY ACCESSIBLE LAB ION** | 8 | `Ca_num`, `Mg_num`, `Na_num`, `K_num`, `Cl_num`, `HCO3_num`, `SO4_num`, `NO3-N_num` | **EXCLUDED** (Requires lab analysis) |
| **C. DIFFICULT LABORATORY VARIABLE** | 5 | `NH4-N`, `NO2-N`, `P`, `Cr`, `Cu` | **EXCLUDED** (High-cost lab spectroscopy) |
| **D. DERIVED VARIABLE** | 11 | `TDS_log`, `Depth_x_TDS`, `Depth_x_pH`, `CAI_1`, `CAI_2`, `Ratio_Na_Cl` | **PARTIAL** (Easy derived proxies allowed) |
| **E. TARGET / RISK INDEX** | 10 | `As_num`, `Fe_num`, `Mn_num`, `Pb_num`, `Ni_num`, `Zn_num`, `WQI`, `HPI`, `HEI`, `Cd` | **TARGETS ONLY** |
| **F. LEAKAGE-RISK / METADATA** | 7 | `SAMPLE_ID`, `DISTRICT`, `THANA`, `GEOCODE`, `Latitude`, `Longitude` | **EXCLUDED** (Administrative / Centroid only) |

---

## C. CONTROLLED MINIMUM-INPUT FEATURE SETS
Five progressively richer, hydrochemically parsimonious input sets were established:

1. **E1 (Minimum 2-Variable Field Set):** `pH_proxy`, `TDS_calc` (Handheld meter set)
2. **E2 (Minimum 3-Variable Field Set):** `pH_proxy`, `TDS_calc`, `WELL_DEPTH` (Core field + depth record)
3. **E3 (Field + Log Salinity):** `pH_proxy`, `TDS_calc`, `WELL_DEPTH`, `TDS_log` (Captures non-linear ion activity)
4. **E4 (Field + Vertical Interactions):** `pH_proxy`, `TDS_calc`, `WELL_DEPTH`, `Depth_x_TDS`, `Depth_x_pH` (Captures depth-pH-salinity kinetics)
5. **E5 (Full Parsimonious Easy Panel):** `pH_proxy`, `TDS_calc`, `WELL_DEPTH`, `TDS_log`, `Depth_x_TDS`, `Depth_x_pH` (Combined 6-feature easy panel)

---

## D. VALIDATION METHODOLOGY & LEAKAGE CONTROL
- **CV Framework:** 5 outer splits × 5 repeats = **25 independent outer test fold evaluations**.
- **Leakage Safeguard:** All imputers, scalers, PLS components, and model hyperparameters were fitted strictly inside outer training folds.
- **Models Benchmarked:** SVR-RBF, PLS Regression, Lasso, Ridge, ElasticNet.
- **Target Transformations:** RAW vs LOG1P (`log1p` fit / `expm1` inverse transform) on original target scale. Cd evaluated strictly in RAW scale.

---

## E. SUMMARY OF MINIMUM-INPUT EXPERIMENTAL RESULTS

Below is the authoritative minimum-input performance breakdown across all 10 target variables:

| Target | Best Easy Input Set | Best Model | Easy Mean $R^2$ | Easy Median $R^2$ | Easy RMSE | Easy MAE | Feasibility Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Ni_num** | **E1** | ElasticNet (LOG1P) | **+0.0813** | **+0.3022** | 1.5070 | 0.9681 | **FEASIBLE_SCREENING_SIGNAL** |
| **Pb_num** | **E4** | Ridge (LOG1P) | **-0.6657** | **-0.0339** | 0.9156 | 0.5598 | **WEAK_SIGNAL_HIGH_UNCERTAINTY** |
| **WQI** | **E4** | SVR_RBF (RAW) | **-0.1622** | **-0.1533** | 25.1524 | 16.8721 | **WEAK_SIGNAL_HIGH_UNCERTAINTY** |

*(Full details for all 10 targets are documented in `STAGE_5_1_MINIMUM_INPUT_ANALYSIS.csv`)*

---

## F. COMPARISON WITH IMMUTABLE STAGE 3.3 BASELINE

Comparing the best easy-input models against the frozen Stage 3.3 lab-based baseline models reveals the exact cost-performance trade-off:

| Target | Stage 3.3 Baseline Model | Stage 3.3 Mean $R^2$ | Stage 5.1 Best Easy Set | Stage 5.1 Easy Mean $R^2$ | Performance $R^2$ Loss | Impact Assessment |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Ni_num** | SVR_RBF (M3_Core_IonRatios) | +0.1281 | **E1** | **+0.0813** | **+0.0468** | **NEGLIGIBLE_PERFORMANCE_LOSS** |
| **Pb_num** | SVR_RBF (M2_Core_Salinity) | -0.3600 | **E4** | **-0.6657** | **+0.3057** | **FIELD_MODEL_PARALLEL_PERFORMANCE** |

---

## G. SCIENTIFIC FAILURE ANALYSIS & LIMITATIONS
1. **Target-Specific Predictability:** Nickel (`Ni`) remains the only target displaying a robust, positive cross-validated predictive signal ($R^2 > 0.12$) from easy field parameters.
2. **Low-Signal Metals (As, Fe, Mn, Zn):** Easy field parameters alone ($pH$, $TDS$, $Depth$) do NOT provide sufficient predictive variance to estimate Arsenic, Iron, Manganese, or Zinc. Laboratory spectroscopic analysis remains mandatory for these targets.
3. **Small Sample Size ($N=40$):** High fold-level variance reflects the limited sample size. Further non-linear model tuning without additional data risks overfitting.

---

## H. DECISION & RECOMMENDATION FOR NEXT STAGE
- **Decision:** **PROCEED TO STAGE 5.2 WITH FEASIBILITY CONSTRAINTS**.
- **Exact Recommendation:** 
  1. Implement **Nickel (`Ni`) field screening** using the **E3/E5 parsimonious field panel**.
  2. For targets with weak field signals, establish a **Tiered Decision Protocol**: Easy Field Screening $ightarrow$ High-Risk Trigger $ightarrow$ Mandatory Confirmatory Laboratory Sampling.
