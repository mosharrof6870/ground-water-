# FINAL TARGETED SCIENTIFIC AUDIT & VALIDATION REPORT
## Groundwater AI Heavy-Metal Screening System (North Bengal Aquifer Framework)

---

### FINAL VALIDATION STATUS

- **Statistical Validation:** `PASS WITH LIMITATIONS` (Small sample size $N=40$, limited quantitative predictive power for Ni)
- **Model Consistency:** `PASS` (Preserved exact binary `HuberRegressor` & `LinearRegression` pipelines)
- **Preprocessing:** `PASS` (Integrated `Pipeline` with `SimpleImputer` & `StandardScaler`)
- **Domain Validation:** `PASS` (2-Layer physical limits + range-based model domain check)
- **Conformal Implementation:** `PASS WITH LIMITATIONS` (90% nominal out-of-fold conformal interval; observed coverage in evaluation artifact = 90.0%)
- **Threshold Logic:** `PASS` (Conservative 5-state logic; zero "SAFE" terminology)
- **Report Consistency:** `PASS` (10-section HTML report with unified recommendation block)
- **Reproducibility:** `PASS` (Deterministic inference across 12 automated test cases)

#### **OVERALL SCIENTIFIC AUDIT:** `PASS WITH LIMITATIONS`

---

### 1. TARGETED ISSUE AUDIT SUMMARY TABLE (ISSUE 8 REQUIRED FORMAT)

| Issue | Current State | Actual Verified State | Severity | Action |
|---|---|---|---|---|
| **Training N** | 39 | **40** (Model estimators fit on $N=40$ via `SimpleImputer(strategy='median')`; complete-case domain profiling evaluated $N=39$ due to 1 missing `WELL_DEPTH` in sample `S98_01798`) | LOW | Preserved models unchanged ($N=40$). Explicitly documented complete-case vs imputed sample trace in manifest & reports. |
| **Conformal wording** | 90% empirical coverage | **90% nominal out-of-fold conformal prediction interval; observed coverage in available evaluation artifact = 90.0% (36/40 samples covered)** | MEDIUM | Updated all UI cards, HTML reports, and audit logs to distinguish nominal interval designation from observed evaluation coverage. |
| **OOF R² reporting** | Difference 0.000 | **Primary ($5\times5$ Nested CV): Ni=0.2240, Cd=0.7061**<br>**Secondary (Single-pass OOF): Ni=0.2271, Cd=0.7158**<br>**Actual Diff: Ni = +0.0031, Cd = +0.0097** | MEDIUM | Reported Primary and Secondary metrics separately with actual non-zero differences. Stopped rounding differences to zero. |
| **Scientific validity** | PASS | **PASS WITH LIMITATIONS** (due to small $N=40$, low Ni $R^2$, empirical association nature, and preliminary screening scope) | HIGH | Replaced blanket "PASS" with granular component scores and explicit scientific limitation statements. |

---

### 2. DETAILED BREAKDOWN OF THE FOUR CORE AUDIT ISSUES

#### ISSUE 1 — TRAINING N TRACE & ROW-LEVEL DATA FLOW

```
[Original Excel Dataset: Groundwater quality data_Northbengal.xlsx]
       ↓ (N = 40 samples: S98_01785 to S98_01834)
[Feature Processing: data/processed/phase4_features_track1.csv]
       ↓ (N = 40 rows: pH_proxy=40, TDS_calc=40, NO3-N_num=40, WELL_DEPTH=39 valid + 1 NaN)
       ↓
 ┌───────────────────────────────────────────────┴───────────────────────────────────────────────┐
 │                                                                                               │
 [Branch A: Complete-Case Statistics Profiling]                              [Branch B: ML Estimator Training & Conformal Calibration]
  - Applied df.dropna(subset=['WELL_DEPTH'])                                  - Passed 40 rows to sklearn Pipeline
  - Dropped Sample S98_01798 (depth missing)                                  - SimpleImputer(strategy='median') imputed S98_01798 depth = 22.0m
  - Computed domain_bounds.json: n_samples = 39                               - Fit HuberRegressor (Ni) & LinearRegression (Cd) on N = 40
  - Mean depth = 22.615 m                                                     - Calibrated CONFORMAL_NI.csv & CONFORMAL_CD.csv on N = 40
```

- **Target | Original N | Final Training N | Excluded N | Excluded Sample(s) | Reason | Justified?**:
  - **Nickel (Ni)**: Original N = 40 | Final Training N = **40** | Excluded N = 0 in model fitting (1 in complete-case profiling) | Sample `S98_01798` | Missing `WELL_DEPTH` (NaN) imputed via median in sklearn `Pipeline` | **YES**. Complete-case profiling evaluated $N=39$ complete rows, while the ML model estimator utilized all $N=40$ samples.
  - **Cadmium (Cd)**: Original N = 40 | Final Training N = **40** | Excluded N = 0 in model fitting (1 in complete-case profiling) | Sample `S98_01798` | Missing `WELL_DEPTH` (NaN) imputed via median in sklearn `Pipeline` | **YES**. Identical dataset handling applied to Cd.

---

#### ISSUE 2 — CONFORMAL COVERAGE & TERMINOLOGY AUDIT

Inspection of `CONFORMAL_NI.csv` and `CONFORMAL_CD.csv`:
- **Total evaluation observations**: $40$ samples ($S98\_01785$ through $S98\_01834$).
- **Covered count (`is_covered == True`)**: $36$ out of $40$ samples for both Ni and Cd.
- **Observed empirical coverage rate**: $\frac{36}{40} = 90.0\%$.

##### Terminology Standards Enforced:
1. **Nominal Conformal Level:** $90\%$ (the mathematical quantile target $q_{0.90}$ selected for interval width calculation).
2. **Observed Empirical Coverage:** $90.0\%$ ($36/40$ samples covered in the evaluation artifact).
3. **Standard Wording:** *"90% nominal CV+ conformal prediction interval; observed coverage in the available evaluation artifact = 90.0% (36/40)."*
4. **Strictly Prohibited Claims:** Neither UI nor report claims *"90% probability that true value lies inside interval"* nor *"guaranteed future coverage"*.

---

#### ISSUE 3 — OOF R² REPORTING & ACTUAL NON-ZERO DIFFERENCES

| Target Metal | Primary Research Metric<br>($5\times5$ Repeated Nested CV Mean) | Secondary Diagnostic Metric<br>(Single-Pass OOF Concatenation) | Actual Difference<br>(Secondary $-$ Primary) | Primary vs Secondary Definition |
| :--- | :---: | :---: | :---: | :--- |
| **Nickel (Ni)** | $R^2_{\text{Repeated-OOF}} = 0.2240$<br>($\text{RMSE} = 1.7707\ \mu\text{g/L}$) | $R^2_{\text{Single-OOF}} = 0.2271$<br>($\text{RMSE} = 1.7675\ \mu\text{g/L}$) | **$+0.0031$** | **Primary:** Mean score across 25 fold evaluations in $5\times5$ nested CV (`TABLE_4`).<br>**Secondary:** Global $R^2$ on concatenated 5-fold OOF vector (`MODEL_PERFORMANCE_SUMMARY`). |
| **Cadmium (Cd)** | $R^2_{\text{Repeated-OOF}} = 0.7061$<br>($\text{RMSE} = 0.0658\ \mu\text{g/L}$) | $R^2_{\text{Single-OOF}} = 0.7158$<br>($\text{RMSE} = 0.0648\ \mu\text{g/L}$) | **$+0.0097$** | **Primary:** Mean score across 25 fold evaluations in $5\times5$ nested CV (`TABLE_4`).<br>**Secondary:** Global $R^2$ on concatenated 5-fold OOF vector (`MODEL_PERFORMANCE_SUMMARY`). |

*Note: Previous audit displayed "Difference = 0.000" due to over-rounding. The true non-zero differences (+0.0031 for Ni and +0.0097 for Cd) are now explicitly reported.*

---

#### ISSUE 4 — AUDIT GRANULARITY & SCIENTIFIC LIMITATIONS

The blanket label "Scientific Validity: PASS" has been permanently replaced with **PASS WITH LIMITATIONS**.

##### Explicit System Limitations Documented in All Reports:
1. **Small Sample Size:** Trained on $N=40$ regional groundwater wells in North Bengal.
2. **Limited Ni Predictive Power:** Nickel model (`HuberRegressor`) exhibits limited quantitative predictive power ($R^2_{\text{Repeated-OOF}} = 0.2240$).
3. **Observational Data Scope:** Predictors represent empirical statistical associations, not established hydrochemical causation.
4. **Preliminary Screening Scope:** Outputs represent model-based preliminary screening tools, not regulatory compliance certificates or laboratory analytical confirmations.
5. **Domain Restrictiveness:** Validated applicability is strictly restricted to the empirical training bounds (pH $6.87–7.32$, TDS $44.5–593.4\text{ mg/L}$, $\text{NO}_3\text{-N } 0.10–12.50\text{ mg/L}$, Depth $9.0–61.0\text{ m}$).

---

### 3. 12-CASE AUTOMATED TEST SUITE MATRIX (100% PASSED)

| Test ID | Test Scenario | Input Vector (pH, TDS, NO3, Depth) | Domain State | Ni Status Code | Cd Status Code | Test Result |
| :---: | :--- | :--- | :---: | :--- | :--- | :---: |
| **1** | Normal In-Domain Sample | (7.1, 220, 1.2, 25) | `IN_DOMAIN` | `BELOW_THRESHOLD` | `BELOW_THRESHOLD` | **PASS** |
| **2** | Low pH (Out-of-Domain) | (6.5, 220, 1.2, 25) | `OUT_OF_DOMAIN` | `OUT_OF_DOMAIN` | `OUT_OF_DOMAIN` | **PASS** |
| **3** | High TDS (Out-of-Domain) | (7.1, 750, 1.2, 25) | `OUT_OF_DOMAIN` | `OUT_OF_DOMAIN` | `OUT_OF_DOMAIN` | **PASS** |
| **4** | High NO3-N (Out-of-Domain) | (7.1, 220, 15.0, 25) | `OUT_OF_DOMAIN` | `OUT_OF_DOMAIN` | `OUT_OF_DOMAIN` | **PASS** |
| **5** | Deep Well (Out-of-Domain) | (7.1, 220, 1.2, 80) | `OUT_OF_DOMAIN` | `OUT_OF_DOMAIN` | `OUT_OF_DOMAIN` | **PASS** |
| **6** | Negative TDS (Invalid) | (7.1, -50, 1.2, 25) | `INVALID` | Rejected | Rejected | **PASS** |
| **7** | Out-of-Bounds pH (Invalid) | (20.0, 220, 1.2, 25) | `INVALID` | Rejected | Rejected | **PASS** |
| **8** | Non-Numeric Input | ('abc', 220, 1.2, 25) | `INVALID` | Rejected | Rejected | **PASS** |
| **9** | Conformal Data Missing | Simulated missing artifact | `IN_DOMAIN` | `CONFIDENCE_UNAVAILABLE` | `CONFIDENCE_UNAVAILABLE` | **PASS** |
| **10** | Interval Overlaps Threshold | High prediction near threshold | `IN_DOMAIN` | `UNCERTAIN` | `UNCERTAIN` | **PASS** |
| **11** | Interval Below Threshold | Normal sample | `IN_DOMAIN` | `BELOW_THRESHOLD` | `BELOW_THRESHOLD` | **PASS** |
| **12** | Potential Exceedance | High prediction > threshold | `IN_DOMAIN` | `POTENTIAL_EXCEEDANCE` | `POTENTIAL_EXCEEDANCE` | **PASS** |

---

### 4. COMPUTATIONAL MODEL MANIFEST

```json
{
  "application": "Groundwater AI Heavy-Metal Screening Web Application",
  "version": "1.0.0-research-hardened",
  "framework": "North Bengal Aquifer Hydrochemistry Framework",
  "models": {
    "ni": {
      "target": "Nickel (Ni_num)",
      "unit": "µg/L",
      "model_type": "HuberRegressor (scikit-learn Pipeline)",
      "file_path": "app/models/ni_model/FINAL_MODEL_NI.joblib",
      "conformal_artifact": "app/models/ni_model/CONFORMAL_NI.csv",
      "feature_list": ["pH_proxy", "TDS_calc", "NO3-N_num", "WELL_DEPTH"],
      "training_n_imputed": 40,
      "training_n_complete_case": 39,
      "sample_exclusion_note": "Sample S98_01798 has 1 missing WELL_DEPTH; complete-case domain profiling used N=39, while model estimator trained on N=40 via median imputer.",
      "oof_r2_primary_5x5_nested_cv": 0.2240,
      "oof_r2_secondary_single_pass": 0.2271,
      "oof_r2_actual_difference": 0.0031,
      "oof_rmse": 1.7707,
      "conformal_nominal_coverage": "90%",
      "conformal_observed_coverage": "90.0% (36/40 samples covered)",
      "screening_threshold": 20.0
    },
    "cd": {
      "target": "Cadmium (Cd_num)",
      "unit": "µg/L",
      "model_type": "LinearRegression (scikit-learn Pipeline)",
      "file_path": "app/models/cd_model/FINAL_MODEL_CD.joblib",
      "conformal_artifact": "app/models/cd_model/CONFORMAL_CD.csv",
      "feature_list": ["pH_proxy", "TDS_calc", "NO3-N_num", "WELL_DEPTH"],
      "training_n_imputed": 40,
      "training_n_complete_case": 39,
      "sample_exclusion_note": "Sample S98_01798 has 1 missing WELL_DEPTH; complete-case domain profiling used N=39, while model estimator trained on N=40 via median imputer.",
      "oof_r2_primary_5x5_nested_cv": 0.7061,
      "oof_r2_secondary_single_pass": 0.7158,
      "oof_r2_actual_difference": 0.0097,
      "oof_rmse": 0.0658,
      "conformal_nominal_coverage": "90%",
      "conformal_observed_coverage": "90.0% (36/40 samples covered)",
      "screening_threshold": 3.0
    }
  }
}
```

---

### 5. RESEARCH-SAFE TERMINOLOGY & MODEL INTEGRITY CONFIRMATION

- **Zero ML Models Altered:** Confirmed that `FINAL_MODEL_NI.joblib` and `FINAL_MODEL_CD.joblib` were not retrained, refitted, or modified in any manner.
- **Forbidden Terms Verification:** Searching the entire application codebase (`grep`) confirms 0 occurrences of non-defensible terms ("safe", "guaranteed safe", "proves", "causes", "100% accurate", "regulatory compliance confirmed").
