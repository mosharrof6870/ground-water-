# EXECUTIVE RESEARCH & DATA PIPELINE REPORT
## North Bengal Groundwater Hydrochemistry & Smart Heavy-Metal Screening

---

### EXECUTIVE SUMMARY
This report provides a clear, step-by-step overview of the groundwater quality analysis and smart screening model developed for the North Bengal aquifer (*N* = 40 regional wells). 

The primary objective is to enable **rapid, low-cost field screening** for dangerous heavy metals (**Cadmium** and **Nickel**) using 4 routine water parameters (**pH, TDS, Nitrate, and Well Depth**), significantly reducing the need for immediate, expensive laboratory analytical testing.

---

### 1. STEP-BY-STEP DATA PREPARATION PIPELINE

```
[ STEP 1: RAW DATA COLLECTION ]
  • Excel File: Groundwater_quality_data_Northbengal.xlsx (N = 40 tube wells)
  • Parameters: Major cations (Ca, Mg, Na, K), anions (Cl, HCO3, SO4, NO3), and trace metals (As, Pb, Ni, Zn, Cd, Cr, Cu, Fe, Mn)
         ↓
[ STEP 2: QA/QC & MISSING DATA IMPUTATION ]
  • Ionic Balance Check: Confirmed cation-anion balance error (CBE) < 5% across wells
  • Sample S98_01798 had 1 missing Well Depth value
  • Treatment: Imputed via median depth (22.0 m) in Scikit-Learn Pipeline
         ↓
[ STEP 3: HYDROCHEMISTRY & RISK INDEXING ]
  • Evaluated water usability: Water Quality Index (WQI)
  • Evaluated heavy metal pollution: Heavy Metal Pollution Index (HPI) & Heavy Metal Evaluation Index (HEI)
  • Classified hydrochemical facies: Gibbs Mechanisms & Ion Exchange Processes
         ↓
[ STEP 4: FINAL ML FEATURE DATASET CREATION ]
  • File Created: data/processed/phase4_features_track1.csv
  • Standardized feature matrix ready for machine learning model training (N = 40 rows)
```

---

### 2. INPUT FEATURES VS. OUTPUT TARGETS

| Category | Parameter Name | Column Code in Dataset | Unit | Role / Purpose |
| :--- | :--- | :--- | :---: | :--- |
| **Input Feature 1** | Water pH | `pH_proxy` | pH unit | Controls metal solubility & mobility |
| **Input Feature 2** | Total Dissolved Solids | `TDS_calc` | mg/L | Indicates overall mineral concentration |
| **Input Feature 3** | Nitrate-Nitrogen | `NO3-N_num` | mg/L | Proxy for agricultural/anthropogenic contamination |
| **Input Feature 4** | Well Depth | `WELL_DEPTH` | meters | Captures aquifer depth layer |
| **Output Target 1** | Nickel Concentration | `Ni_num` | µg/L | Target metal screening (Threshold: 20 µg/L) |
| **Output Target 2** | Cadmium Concentration | `Cd_num` | µg/L | Target metal screening (Threshold: 3 µg/L) |

---

### 3. MODEL PREPARATION & ACCURACY PERFORMANCE

Two machine learning pipelines were trained using **Repeated 5×5 Nested Cross-Validation** to ensure zero data leakage and reliable generalization:

| Target Metal | Model Type | Accuracy Score (*R*²) | Prediction Error (RMSE) | Water Management Interpretation |
| :--- | :--- | :---: | :---: | :--- |
| **Cadmium (Cd)** | Linear Regression | ***R*² = 0.71** (71% variance explained) | 0.0658 µg/L | **High Predictive Accuracy:** Reliable for quantitative estimation and field risk prioritization. |
| **Nickel (Ni)** | Huber Regressor | ***R*² = 0.22** (22% variance explained) | 1.7707 µg/L | **Baseline Screening Tool:** Suitable for preliminary risk categorization, not exact concentration prediction. |

---

### 4. DECISION SUPPORT & UNCERTAINTY BOUNDS (CONFORMAL PREDICTION)

To guarantee water management safety and prevent false negatives, every model prediction is wrapped with a **90% Conformal Uncertainty Interval**:

* **Observed Reliability:** **90.0% coverage** (36/40 evaluation samples fell within predicted confidence bands).
* **5-State Screening Decision Categories:**
  1. `BELOW_THRESHOLD`: Metal concentration is reliably below regulatory limits.
  2. `UNCERTAIN`: Prediction interval overlaps safety threshold; laboratory re-testing recommended.
  3. `POTENTIAL_EXCEEDANCE`: High likelihood of exceeding safety limits; priority site action required.
  4. `OUT_OF_DOMAIN`: Water parameters fall outside validated regional limits (pH < 6.87 or > 7.32, TDS > 593.4 mg/L).
  5. `INVALID`: Inputs physically impossible (e.g., negative depth or TDS).

---

### 5. COMPLETE FILE & DATASET INVENTORY MAP

| Phase | Purpose | File Name & Path | Key Contents |
| :--- | :--- | :--- | :--- |
| **Raw Data** | Original Laboratory Readout | `data/raw/Groundwater_quality_data_Northbengal.xlsx` | 40 sample records with 26 chemical parameters |
| **Hydrochemistry** | Water Quality & Risk Indices | `data/processed/phase2_risk_indices_results.csv` | Calculated WQI, HPI, HEI, and Gibbs classifications |
| **ML Training Set** | Prepared ML Input Dataset | `data/processed/phase4_features_track1.csv` | Clean 40-row feature matrix (`pH_proxy`, `TDS_calc`, `NO3-N`, `WELL_DEPTH`) |
| **Predictions** | Final Screening Decisions | `water_final_results/predictions/FINAL_SCREENING_DECISIONS.csv` | Per-well model predictions, uncertainty intervals, & screening decisions |
| **Audit Report** | Technical Validation Report | `scientific_audit_report.pdf` | Complete 5-page forensic audit for peer review |

---

### CONCLUSION FOR WATER MANAGEMENT POLICY
This pipeline demonstrates that routine water quality parameters can serve as an **effective, low-cost first line of defense** for groundwater monitoring in North Bengal, enabling authorities to target expensive heavy-metal laboratory analysis only where screening uncertainty or risk exceedance is flagged.
