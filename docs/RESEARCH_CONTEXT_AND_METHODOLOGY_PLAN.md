# Groundwater Quality & Heavy Metal Assessment (North Bengal)
## Project Research Context, Literature Matrix, and Methodology Roadmap

---

### 1. Project Overview & Title
* **Proposed Manuscript Title:** *An Explainable Small-Sample Machine Learning Framework for Groundwater Quality and Heavy Metal Risk Assessment in North Bengal, Bangladesh*
* **Target Domain:** Interdisciplinary Collaboration (Computer Science & Engineering + Water Resources/Management)
* **Dataset Location:** `/home/mosharrof/personal Doc/water jounal/Groundwater quality data_Northbengal.xlsx`
* **Reference Papers Location:** `/home/mosharrof/personal Doc/water jounal/ref pep/`

---

### 2. Primary Dataset Audit Summary
* **Total Samples:** 40 groundwater well samples across North Bengal districts (Panchagarh, Thakurgaon, Dinajpur, etc.).
* **Total Parameters:** 31 columns (Well Type, Well Depth, District, Thana, Union, Mouza, Geocode, Sample Code).
* **Cations & Anions:** $\text{Ca}^{2+}$, $\text{Mg}^{2+}$, $\text{Na}^+$, $\text{K}^+$, $\text{P}$, $\text{Cl}^-$, $\text{HCO}_3^-$, $\text{SO}_4^{2-}$, $\text{NH}_4\text{-N}$, $\text{NO}_2\text{-N}$, $\text{NO}_3\text{-N}$.
* **Heavy Metals (9 Parameters):** Lead ($\text{Pb}$), Nickel ($\text{Ni}$), Zinc ($\text{Zn}$), Arsenic ($\text{As}$), Cadmium ($\text{Cd}$), Chromium ($\text{Cr}$), Copper ($\text{Cu}$), Iron ($\text{Fe}$), Manganese ($\text{Mn}$).
* **Data Cleaning Required:**
  1. Removal of completely blank columns (Columns 8, 14, 21) and empty sheets (`Sheet2`, `Sheet3`).
  2. Addition of header for Sample ID column (Col 7).
  3. Conversion of Below Detection Limit (BDL) text values (e.g. `< 0.2`, `< 0.02`, `< 0.004`, `< 0.5`, `< 1`) using standard Half-Detection Limit ($DL/2$) imputation.
  4. Imputation of missing values in $\text{Zn}$ (12.5%) and well metadata.

---

### 3. Literature Matrix (Downloaded Reference Papers Summary)

| Paper / Citation | Journal & Year | Sample Size ($N$) | Core Method / Algorithm | Key Findings & Relevance |
| :--- | :--- | :--- | :--- | :--- |
| **Aldrees et al.** (`1-s2.0-S2214581826001564-main.pdf`) | *Journal of Hydrology: Regional Studies* (2026) | $N=39$ | Gaussian Mixture Models (GMM) + Gradient Boosting + SHAP | Augmented 39 samples to 700 synthetic samples for salinity prediction with low Jensen-Shannon divergence. |
| **Shyamala et al.** (`s41598-026-62033-0_reference.pdf`) | *Scientific Reports* (2026) | $N=135$ | Optuna-optimized Stacked Meta-Learner (XGBoost, CatBoost, LightGBM) + SHAP + LIME | Achieved $R^2=0.938$ for WQI prediction; identified $\text{NH}_3$, $\text{Fe}$, $\text{Cr}$ as main drivers. |
| **Huynh et al.** (`ijerph-19-12180-v2.pdf`) | *IJERPH* (2022) | Shallow Aquifer | Random Forest + Low-Cost Parameter Surrogate Modeling | Predicted heavy metals ($\text{As}$, $\text{Fe}$, $\text{Mn}$) from quick-measure field parameters. |
| **Relangi et al.** (`s41598-026-54560-7_reference.pdf`) | *Scientific Reports* (2026) | Small Water Dataset | SMOTE + GAN Synthetic Data + XGBoost / Gradient Boosting | Achieved $99.47\%$ accuracy and $MMD=0.0006$ on synthetic water datasets. |
| **Zhan et al.** (`es5c01025.pdf`) | *Environmental Science & Technology* (2025) | Small Data | Optimization Methods + SVM + Random Forests + GANs | Established small data principles for groundwater contamination monitoring. |
| **Chowdhury et al.** (`Chowdhury_2025_Environ._Res._Lett._20_033003.pdf`) | *Environmental Research Letters* (2025) | Systematic Review | Analytical Models + PBPMs + Machine Learning (RF, XGBoost, ANN) | Comprehensive systematic review of ML in groundwater quality under climate stressors. |

---

### 4. Core Novelties (Proposed Contributions)

1. **Geographical & Hydrochemical Novelty:** First Explainable AI and Small-Sample Data Augmentation study focusing specifically on the North Bengal alluvial groundwater aquifer system (Panchagarh, Thakurgaon, Dinajpur).
2. **Unified Multi-Output Architecture:** Simultaneous prediction of both overall Water Quality Index (WQI) AND Heavy Metal Pollution Index (HPI / Heavy Metal Risk) using low-cost field parameters in a single ML framework.
3. **Physics-Constrained Data Augmentation:** Synthetic data generation (Gaussian Copula / GMM) enforced with Ionic Charge Balance Error ($<5\%$) constraints to maintain hydrogeochemical realism.
4. **Dual-Level Explainable AI (XAI):** Global SHAP feature ranking for regional insights + Local Well-Level SHAP interpretability for site-specific well management.

---

### 5. Implementation Roadmap
* **Phase 1:** Data Cleaning, BDL Imputation ($DL/2$), and Header Standardization (`cleaned_northbengal_water_data.csv`).
* **Phase 2:** Exploratory Data Analysis, WQI Calculation, and Heavy Metal Risk Indexing.
* **Phase 3:** Virtual Sample Generation (VSG) via Gaussian Copula / GMM ($N=40 \rightarrow 700$).
* **Phase 4:** Multi-Output Machine Learning Model Training (XGBoost, Random Forest, CatBoost, SVR) & Evaluation ($R^2$, RMSE, MAE).
* **Phase 5:** SHAP Global & Local Interpretability Analysis & Manuscript Drafting.
