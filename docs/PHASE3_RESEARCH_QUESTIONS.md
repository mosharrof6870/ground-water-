# North Bengal Groundwater Research Project — Phase 3: Research Questions & Target Definition

**Target Journal Standards:** *Journal of Hydrology / Water Research / Environmental Pollution (Q1)*  
**Core Dataset:** $N = 40$ primary groundwater samples, 16 districts, North Bengal, Bangladesh  
**Output Document:** `docs/PHASE3_RESEARCH_QUESTIONS.md`  

---

## 1. Context & Literature Grounding

Based on our reference literature collection (`ref pep/` repository including XAI groundwater quality studies in Saudi Arabia, India, and global alluvial basins), machine learning applications in groundwater studies frequently suffer from **predictor-target leakage** (e.g., using component metals to predict composite pollution indices) or **lack of hydrochemical grounding**.

Phase 3 addresses these vulnerabilities by establishing **three locked research questions**, **15 domain-valid predictors**, and **8 explicit prediction targets** divided into isolated operational tracks.

---

## 2. Formal Research Questions (Q1 Journal Standard)

### Research Question 1 (Track 1: Heavy Metal Concentration Modeling)
> **"To what extent can low-cost field parameters (pH, EC, Depth) combined with major ion hydrochemistry predict toxic heavy metal concentrations (As, Fe, Mn, Pb, Ni, Zn) in alluvial groundwater aquifers?"**

- **Scientific Rationale:** High-resolution laboratory measurement of heavy metals ($\text{As}, \text{Fe}, \text{Mn}$) requires expensive ICP-MS/AAS equipment, which is often unavailable in low-resource alluvial regions. Predicting concentrations from routine hydrochemical monitoring provides a low-cost, scalable early-warning surrogate.
- **Target Variables (6 separate continuous models):**
  1. $\text{As}$ Concentration ($\mu\text{g/L}$) — Arsenic contamination prediction
  2. $\text{Fe}$ Concentration ($\text{mg/L}$) — Iron enrichment prediction
  3. $\text{Mn}$ Concentration ($\text{mg/L}$) — Manganese anomaly prediction
  4. $\text{Pb}$ Concentration ($\mu\text{g/L}$) — Lead trace contamination
  5. $\text{Ni}$ Concentration ($\mu\text{g/L}$) — Nickel trace contamination
  6. $\text{Zn}$ Concentration ($\mu\text{g/L}$) — Zinc concentration (KNN-imputed target)

---

### Research Question 2 (Track 2: Composite Pollution Risk Index Modeling)
> **"Can heavy metal pollution risk indices (HPI and HEI) be accurately predicted from hydrogeochemical facies and major ion dynamics alone, while guaranteeing zero predictor-target leakage?"**

- **Scientific Rationale:** Heavy Metal Pollution Index ($\text{HPI}$) and Heavy Metal Evaluation Index ($\text{HEI}$) quantify integrated toxicological risks. By excluding component metals ($\text{As}, \text{Fe}, \text{Mn}, \text{Pb}, \text{Ni}, \text{Zn}$) from the predictor pool, we test if macro-hydrochemistry and ion-exchange dynamics encode the underlying geogenic metal risk.
- **Target Variables (2 separate continuous models):**
  1. Heavy Metal Pollution Index ($\text{HPI}$)
  2. Heavy Metal Evaluation Index ($\text{HEI}$)
- **Leakage Prevention Requirement:** Component heavy metals are **strictly excluded** from all Track 2 predictor matrices.

---

### Research Question 3 (Track 3: Risk Index Divergence & Hydrochemical Synthesis)
> **"How do toxicity-focused heavy metal indices (HPI/HEI) diverge from conventional general Water Quality Indices (WQI), and how do specific hydrogeochemical processes (e.g., reductive dissolution vs. ion exchange) explain these discrepancies?"**

- **Scientific Rationale:** General $\text{WQI}$ assesses overall potability ($85\%$ of North Bengal samples are "Poor Water Quality" driven by baseline $\text{Fe}/\text{Mn}$), whereas $\text{HPI}$ isolates severe toxicity ($5\%$ critical hotspots in Pabna & Bogra). Track 3 synthesizes ML predictions with SHAP explainability to map where and why $\text{WQI}$ and $\text{HPI}$ diverge.
- **Output:** Residual divergence mapping ($\Delta = \text{HPI}_{\text{norm}} - \text{WQI}_{\text{norm}}$) correlated with Chloro-Alkaline Indices ($\text{CAI-1}, \text{CAI-2}$) and redox proxies.

---

## 3. Predictor Feature Matrix Specification (15 Features)

To maintain a defensible predictor-to-sample ratio ($15:40 \approx 1:2.7$), predictors are selected based on field accessibility and domain hydrogeology:

| Feature Category | Parameter | Unit | Hydrogeochemical Rationale |
| :--- | :--- | :---: | :--- |
| **Field Metrics (4)** | `pH` | — | Controls heavy metal speciation, adsorption, and solubility |
| | `EC` / `TDS_calc` | $\mu\text{S/cm}$ / $\text{mg/L}$ | Reflects ionic strength and total dissolved solids |
| | `WELL_DEPTH` | $\text{m}$ | Distinguishes shallow (<30m) oxic/anoxic from deep aquifers |
| | `WELL_TYPE` | Categorical | Shallow Tube Well (STW) vs Hand Tube Well (HTW) |
| **Major Anions (4)** | `Cl_num` | $\text{mg/L}$ | Salinity indicator, halite dissolution proxy |
| | `HCO3_num` | $\text{mg/L}$ | Carbonate weathering & organic matter respiration proxy |
| | `SO4_num` | $\text{mg/L}$ | Pyrite oxidation & sulfate reduction marker |
| | `NO3-N_num` | $\text{mg/L}$ | Anthropogenic agricultural/sanitation contamination indicator |
| **Major Cations (4)** | `Ca_num` | $\text{mg/L}$ | Carbonate/silicate dissolution, ion exchange proxy |
| | `Mg_num` | $\text{mg/L}$ | Dolomite dissolution & clay mineral exchange |
| | `Na_num` | $\text{mg/L}$ | Halite dissolution & direct ion exchange product |
| | `K_num` | $\text{mg/L}$ | Silicate (K-feldspar) weathering |
| **Hydrochemical Context (3)** | `Facies` | Categorical | Hydrochemical facies ($\text{Ca-HCO}_3$, $\text{Mixed-HCO}_3$, etc.) |
| | `Ion_Exchange_Process` | Categorical | Direct ($\text{CAI}<0$) vs Reverse ($\text{CAI}>0$) Ion Exchange |
| | `Gibbs_Mechanism` | Categorical | Rock Weathering vs Evaporation dominance |

---

## 4. Machine Learning & Validation Strategy (Phase 4–6 Preview)

1. **Small-Sample Model Suite:**
   - Linear regularized models: **ElasticNet / Lasso**
   - Ensemble tree models: **Random Forest**, **Extra Trees**, **XGBoost**
   - Non-linear kernel models: **Support Vector Regression (SVR)**

2. **Validation Framework:**
   - **Repeated Nested Cross-Validation:** Outer 5-fold CV $\times$ 10 repeats (50 evaluation splits) for unbiased performance estimation ($R^2$, $\text{RMSE}$, $\text{MAE}$).
   - **Hyperparameter Optimization:** Inner 3-fold CV grid search strictly inside each outer training fold to prevent data leakage.

3. **Explainable AI (XAI):**
   - **SHAP (SHapley Additive exPlanations):** Global feature importance & dependence plots for hydrochemical mechanisms.
