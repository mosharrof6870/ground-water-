# Groundwater AI Heavy-Metal Screening Web Application

## Overview
This production-style Streamlit web application provides **AI-assisted preliminary groundwater heavy-metal screening** for Nickel (Ni) and Cadmium (Cd) based on low-cost hydrochemical field parameters (`pH`, `TDS`, `NO3-N`, `Well Depth`).

The application directly loads pre-trained research ML pipelines (`FINAL_MODEL_NI.joblib` and `FINAL_MODEL_CD.joblib`) and cross-conformal prediction calibration artifacts (`CONFORMAL_NI.csv` and `CONFORMAL_CD.csv`).

---

## Key Features & Scientific Principles

### 1. 2-Layer Input Validation & 3-State Domain Logic
- **Layer A (Basic Input Validity)**: Rejects physically invalid values immediately (pH $\le 0$ or $> 14$, TDS $< 0$, NO3-N $< 0$, Well Depth $\le 0$). Prediction is cancelled.
- **Layer B (Range-Based Model Applicability Check)**: Compares inputs against empirical training dataset bounds (`pH`: 6.87–7.32, `TDS`: 44.5–593.4 mg/L, `NO3-N`: 0.10–12.50 mg/L, `Well Depth`: 9.0–61.0 m).
- **Domain States**:
  - `IN_DOMAIN`: Input falls within training data distribution.
  - `OUT_OF_DOMAIN`: Input passes basic validity but lies outside training bounds. Prediction is tagged with prominent warnings requiring laboratory confirmation.
  - `INVALID`: Input parameters fail physical checks. Prediction is rejected.

### 2. Validated Model Input Range vs. Environmental Reference Range
- **Validated Model Input Range**: The actual min-max parameter bounds of the empirical training dataset used for model fitting.
- **Environmental Reference Range**: Water quality standard benchmarks (e.g. WHO/BIS drinking water guidelines). These are kept strictly distinct in UI tables and report generation.

### 3. Conformal Uncertainty & Conservative Decision Engine
- **Nickel (Ni)**: `HuberRegressor` ($R^2_{\text{OOF}} = 0.224$, $\text{RMSE} = 1.771\ \mu\text{g/L}$, Spearman $\rho = 0.633$). Screening threshold: $20.0\ \mu\text{g/L}$.
- **Cadmium (Cd)**: `LinearRegression` ($R^2_{\text{OOF}} = 0.706$, $\text{RMSE} = 0.066\ \mu\text{g/L}$, Spearman $\rho = 0.685$). Screening threshold: $3.0\ \mu\text{g/L}$.
- Decisions are based on 90% conformal prediction bounds. Never uses misleading labels like "SAFE".

### 4. Non-Causal Scientific Interpretation
- Features describe empirical statistical predictive associations in trained models without asserting physical causation (e.g. "TDS is the primary predictive variable in the trained Ni HuberRegressor model").

---

## Directory Structure
```
app/
├── app.py                     # Main Streamlit Dashboard Application
├── config.py                  # Thresholds, Model Paths, Validation Ranges
│
├── models/
│   ├── ni_model/              # Nickel Model Artifacts (joblib & conformal CSV)
│   ├── cd_model/              # Cadmium Model Artifacts (joblib & conformal CSV)
│   └── domain_bounds.json     # Empirical Training Dataset Bounds & Statistics
│
├── services/
│   ├── model_loader.py        # Singleton Loader for Joblib & Conformal Objects
│   ├── prediction_service.py  # Orchestrator for Validation -> Model -> Uncertainty
│   ├── uncertainty_service.py # CV+ Conformal Prediction Bounds Calculator
│   └── decision_engine.py     # Regulatory Threshold Screening Logic
│
├── components/
│   ├── input_form.py          # Input Parameter Card with Presets
│   ├── result_card.py         # Glassmorphic Result Card & Badges
│   ├── threshold_chart.py     # Horizontal Plotly Range Chart
│   ├── applicability_panel.py # Range-Based Model Applicability Table
│   ├── interpretation_panel.py# Non-Causal Interpretation & Model Specs Expander
│   └── report_generator.py    # Downloadable HTML Report Generator (10 Sections)
│
├── utils/
│   ├── validation.py          # Strict 2-Layer Numeric & Range Input Validator
│   └── formatting.py          # Status Badge Color Styling & Formatting
│
├── test_app_suite.py          # 12-Case Automated Test Suite
├── requirements.txt           # Python Dependencies
└── README.md                  # Setup & Deployment Instructions
```

---

## How to Run Locally

### 1. Install Dependencies
```bash
pip install -r app/requirements.txt
```

### 2. Launch Streamlit Application
```bash
streamlit run app/app.py
```

### 3. Run Automated Test Suite
```bash
python app/test_app_suite.py
```

---

## Research Disclaimer
> **IMPORTANT:** Research prototype — model-based preliminary groundwater screening. Results do not replace laboratory measurement (AAS / ICP-MS) and should not be interpreted as definitive regulatory compliance.
