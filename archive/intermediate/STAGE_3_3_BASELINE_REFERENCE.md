# STAGE 3.3 BASELINE REFERENCE DOCUMENT

## 1. Executive Summary & Freeze Policy
This document establishes immutable baseline references from prior stages (Stage 3, 3.1, and 3.2) to evaluate Stage 3.3 feature engineering and refinement. All outer test folds remain sacred; zero leakage is strictly enforced.

## 2. Immutable References Across Stages

### Stage 3: Raw-Target 5x5 Repeated Nested CV Baseline
- **N = 40** (Zn effective N = 35 due to missing values).
- **Core Baseline Models:** Ridge, Lasso, ElasticNet, SVR, HuberRegressor, RandomForest, ExtraTrees, GradientBoosting.
- **Key Findings:** Raw target distributions exhibited severe right-skewness (As skew = 3.08, Fe skew = 2.14, Pb skew = 2.45), leading to negative mean R² for raw linear estimators due to extreme prediction errors on skewed out-of-fold samples.

### Stage 3.1: Raw vs Log1p Target Scale Audit
- **Log1p Sensitivity Audit:** Applying `log1p` target transformation inside `TransformedTargetRegressor` substantially improved rank stability and out-of-fold generalization for skewed heavy metals (As, Fe, Mn, Pb, Ni, Zn) and risk indices (HPI, HEI, WQI).
- **Domain Safety Check for Cd:** Cadmium (`Cd`) contains negative values (minimum -5.55 mg/L equivalent index), rendering `log1p` mathematically invalid. `Cd` was strictly kept on the RAW/Yeo-Johnson scale.

### Stage 3.2: Performance Optimization & Modeling Discipline
- **Hyperparameter & Convergence Optimization:** Solvers set to `max_iter=20000`, `tol=1e-2` to eliminate convergence warnings across 640 configurations.
- **Top Performing Candidates:**
  - **Ni (Nickel):** `HuberRegressor` + `LOG1P` + `CORE 4` feature set achieved **Mean R² = +0.131**, **Median R² = +0.257**, and **Spearman $\rho = +0.654$ ($p < 0.001$)**.
  - **Pb (Lead):** `Ridge` / `ExtraTrees` + `LOG1P` achieved **Median R² = +0.121** and **Spearman $\rho = +0.673$ ($p < 0.001$)**.
  - **Mn (Manganese):** `HuberRegressor` achieved **Spearman $\rho = +0.463$ ($p < 0.005$)**.
  - **WQI (Water Quality Index):** `RandomForest` achieved **Spearman $\rho = +0.421$ ($p < 0.01$)** (Deterministic reconstruction benchmark).

## 3. Major Limitations Identified
1. **Sample Size ($N=40$):** High sample-to-feature ratio risk if feature dimension is artificially inflated.
2. **Missing Environmental Controls:** Absence of measured redox potential (Eh), dissolved oxygen (DO), oxidation-reduction potential (ORP), organic carbon (TOC), and detailed aquifer mineralogy.
3. **Multicollinearity:** High inter-ion redundancy ($|r| > 0.80$ between TDS, Ca, Mg, Na, HCO3).
