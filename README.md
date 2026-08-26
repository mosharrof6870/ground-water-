# North Bengal Groundwater Heavy-Metal AI Screening & Audit System

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Validation](https://img.shields.io/badge/Audit%20Status-PASS%20WITH%20LIMITATIONS-orange.svg)](scientific_audit_report.md)
[![Tests](https://img.shields.io/badge/Test%20Suite-12%2F12%20PASSED-brightgreen.svg)](app/test_app_suite.py)

---

## 1. Project Overview

This repository contains the complete research pipeline, forensic audit documentation, and deployment web application for AI-driven heavy-metal preliminary screening ($\text{Ni}$ and $\text{Cd}$) in the North Bengal groundwater aquifer system.

### Key Research Models & Metrics (Frozen Pre-trained Estimators)
- **Target 1: Nickel ($\text{Ni}$)**
  - Model: `HuberRegressor` (with `SimpleImputer` & `StandardScaler`)
  - Primary $5\times5$ Nested CV OOF $R^2$: $\approx 0.2240$
  - Single-Pass OOF $R^2$: $0.2271$ (Difference: $+0.0031$)
  - Interpretation: *Limited quantitative predictive performance; preliminary screening only.*
- **Target 2: Cadmium ($\text{Cd}$)**
  - Model: `LinearRegression` (with `SimpleImputer` & `StandardScaler`)
  - Primary $5\times5$ Nested CV OOF $R^2$: $\approx 0.7061$
  - Single-Pass OOF $R^2$: $0.7158$ (Difference: $+0.0097$)
  - Interpretation: *Stronger quantitative predictive performance.*

---

## 2. Research Architecture

```
RAW GROUNDWATER DATA (N=40)
        ↓
DATA VALIDATION / PREPROCESSING (SimpleImputer + StandardScaler)
        ↓
HYDROCHEMICAL CHARACTERIZATION (Facies, Gibbs, Ion-Exchange)
        ↓
CONTAMINATION / RISK INDICES (WQI, HPI, HEI)
        ↓
PCA / HYDROCHEMICAL REGIME ANALYSIS
        ↓
TARGET-SPECIFIC ML SCREENING (HuberRegressor for Ni, LinearRegression for Cd)
        ↓
LEAKAGE-FREE VALIDATION (5x5 Repeated Nested CV)
        ↓
SHAP / MODEL INTERPRETATION (TDS primary driver for Ni; NO3-N for Cd)
        ↓
OUT-OF-FOLD RESIDUAL QUANTILE UNCERTAINTY (90% Nominal Coverage)
        ↓
VALIDATED INPUT-DOMAIN CHECK (2-Layer Physical + Range Check)
        ↓
SCREENING DECISION ENGINE (5 Conservative Decision States)
        ↓
RESPONSIVE WEB APPLICATION & EXPORTABLE REPORT (Streamlit + HTML)
```

---

## 3. Uncertainty Quantification Method

- **Method Name:** **Out-of-Fold Residual Quantile Conformal Prediction**
- **Quantile Thresholds ($q_{\text{hat}}$):**
  - $\text{Ni}: q_{\text{hat}} \approx 1.88145\ \mu\text{g/L}$
  - $\text{Cd}: q_{\text{hat}} \approx 0.09335\ \mu\text{g/L}$
- **Nominal Level:** $90\%$ ($\alpha = 0.10$)
- **Observed Empirical Evaluation Coverage:** $90.0\%$ ($36/40$ samples covered for both $\text{Ni}$ and $\text{Cd}$)
- **Disclaimer:** Intervals represent out-of-fold empirical calibration metrics. They do not guarantee miscoverage rates on unobserved future samples under spatial or temporal shifts.

---

## 4. Validated Model Domain Ranges

Inputs outside these ranges trigger an **OUT-OF-DOMAIN** status requiring mandatory laboratory confirmation:

| Feature | Validated Training Range | Unit |
|---|:---:|:---:|
| **pH** | $6.87 – 7.32$ | $-\log[\text{H}^+]$ |
| **TDS** | $44.5 – 593.4$ | $\text{mg/L}$ |
| **$\text{NO}_3\text{-N}$** | $0.10 – 12.50$ | $\text{mg/L}$ |
| **Well Depth** | $9.0 – 61.0$ | $\text{m}$ |

---

## 5. Project Directory Structure

```
groundwater_research/
├── README.md                          # Project documentation
├── CHANGELOG.md                       # Version history log
├── PROJECT_FILE_INVENTORY.csv         # Full 418-file classification inventory
├── PROJECT_MANIFEST.csv               # 147 core active files manifest
├── scientific_audit_report.md         # Formal scientific audit report
├── requirements.txt                   # Python environment dependencies
├── environment.yml                    # Conda environment definition
│
├── app/                               # Streamlit deployment application
│   ├── app.py                         # Main Streamlit web application
│   ├── config.py                      # System configuration & thresholds
│   ├── test_app_suite.py              # 12-case validation test suite
│   ├── models/                        # Pre-trained joblib models & artifacts
│   ├── services/                      # Inference, uncertainty & decision logic
│   └── components/                    # UI cards and HTML report generator
│
├── data/                              # Groundwater datasets
│   ├── raw/                           # Original Excel field dataset (N=40)
│   └── processed/                     # Feature dataset (phase4_features_track1.csv)
│
├── scripts/                           # Reproducibility pipeline scripts
│   ├── master_groundwater_screening_architecture.py
│   └── reorganize_project.py
│
├── water_final_results/               # Publication research results
│   ├── tables/                        # Tables (TABLE_4_REPEATED_OOF_PERFORMANCE.csv)
│   ├── figures/                       # Research figures
│   ├── predictions/                   # Out-of-fold prediction matrices
│   ├── uncertainty/                   # Conformal evaluation CSVs
│   └── interpretation/                # SHAP matrix artifacts
│
└── archive/                           # Safely preserved historical stage outputs
    ├── intermediate/                  # Stage 3/4/5 CSVs & reports
    ├── obsolete/                      # Archived zip files & redundant outputs
    └── debug/                         # Runtime logs & lock files
```

---

## 6. Execution & Reproducibility Instructions

### Running the Web Application Locally
```bash
streamlit run app/app.py
```
*Access via browser at `http://localhost:8501`*

### Running the 12-Case Validation Test Suite
```bash
python3 app/test_app_suite.py
```

### Reviewing the Audit Report
View [scientific_audit_report.md](scientific_audit_report.md) for full forensic analysis, sample traces, and scientific limitations.
