# North Bengal Groundwater Dataset — Phase 2 Risk Indices Computation & Phase 2.1 Spatial Hotspot Audit

**Date of Completion:** August 2026  
**Output Dataset:** `/home/mosharrof/personal Doc/water jounal/data/processed/phase2_risk_indices_results.csv`  
**Interactive GIS Hotspot Map:** `/home/mosharrof/personal Doc/water jounal/reports/north_bengal_groundwater_hotspots.html`  
**Sample Constraints:** $N = 40$ primary groundwater samples across 16 districts in North Bengal  

---

## Executive Summary

Phase 2 and 2.1 establish the comprehensive drinking water quality risk profile ($\text{WQI}$), heavy metal pollution burden ($\text{HPI}$, $\text{HEI}$, $C_d$), and spatial hotspot distribution for North Bengal groundwater ($N=40$).

### Core Findings:
1. **Water Quality Index (WQI):**
   - **Mean WQI:** $111.18 \pm 15.42$ (Range: $97.16 - 156.79$, Median: $107.62$).
   - **Categorization:** **$85.0\%$ of samples (34/40)** fall into the **"Poor Water Quality"** class ($100 \le \text{WQI} < 200$), driven by elevated baseline dissolved iron ($\text{Fe}$), manganese ($\text{Mn}$), and total ionic concentrations relative to WHO 2011 drinking guidelines.
   - **$15.0\%$ of samples (6/40)** qualify as **"Good Water Quality"** ($50 \le \text{WQI} < 100$). Zero samples are classified as "Unsuitable for Drinking" under general major-ion WQI.

2. **Heavy Metal Pollution Index (HPI) & Critical Hotspots:**
   - **Mean HPI:** $16.98$ (Median HPI: $5.94$). $95.0\%$ of North Bengal groundwater exhibits low baseline heavy metal contamination.
   - **Critical Polluted Hotspots ($\text{HPI} > 100$):** **$5.0\%$ of samples (2/40)** exceed the critical safety threshold ($\text{HPI} > 100$):
     - **Sample `S98_01828` (Bera, Pabna):** $\text{HPI} = 129.45$, $\text{As} = 28.7\ \mu\text{g/L}$ ($2.87\times$ WHO limit), $\text{Fe} = 3.75\text{ mg/L}$, $\text{Mn} = 2.21\text{ mg/L}$.
     - **Sample `S98_01834` (Sherpur, Bogra):** $\text{HPI} = 124.41$, $\text{As} = 27.8\ \mu\text{g/L}$ ($2.78\times$ WHO limit), $\text{Fe} = 19.60\text{ mg/L}$ ($65.3\times$ WHO limit), $\text{Mn} = 2.27\text{ mg/L}$.

3. **Geochemical Hotspot Driver:** Both critical heavy metal hotspots are geographically concentrated in the southeastern alluvial floodplain (Pabna-Bogra along the Jamuna/Padma river system), where active reductive dissolution of Fe-Mn oxyhydroxides releases co-precipitated geogenic arsenic into shallow aquifers.

---

## 1. Water Quality Index (WQI) Breakdown ($N=40$)

- **Methodology:** Weighted Arithmetic Index Method across 16 parameters ($\text{pH, TDS, Ca, Mg, Na, K, Cl, HCO}_3, \text{SO}_4, \text{NO}_3\text{-N, Fe, Mn, As, Pb, Ni, Zn}$).
- **Formula:**
  $$w_i = \frac{K}{S_i}, \quad K = \frac{1}{\sum (1/S_i)}, \quad q_i = \frac{C_i}{S_i} \times 100, \quad \text{WQI} = \sum w_i q_i$$

| WQI Range | Water Quality Classification | Sample Count | Percentage | Spatial Distribution |
| :---: | :--- | :---: | :---: | :--- |
| $< 50$ | Excellent Water | 0 | 0.0% | None |
| $50 \le \text{WQI} < 100$ | Good Water Quality | 6 | 15.0% | Panchagarh, Thakurgaon, Dinajpur (Upper Alluvial Fan) |
| $100 \le \text{WQI} < 200$ | Poor Water Quality | 34 | 85.0% | Widespread across Central & Southern North Bengal |
| $200 \le \text{WQI} < 300$ | Very Poor Water Quality | 0 | 0.0% | None |
| $\ge 300$ | Unsuitable for Drinking | 0 | 0.0% | None |

---

## 2. Heavy Metal Risk Indices ($\text{HPI}$, $\text{HEI}$, $C_d$)

| Index | Formula / Concept | Mean | Median | Range | Critical Threshold | Safety Assessment |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **HPI** | $\frac{\sum W_i Q_i}{\sum W_i}$ | **16.98** | **5.94** | $2.26 - 129.45$ | $> 100$ | 95% of samples safe; 2 critical hotspots in Pabna & Bogra |
| **HEI** | $\sum \frac{M_i}{S_i}$ | **0.44** | **0.20** | $0.07 - 3.03$ | $> 20$ | 100% Low Contamination ($\text{HEI} < 10$) |
| **$C_d$** | $\sum \left( \frac{M_i}{S_i} - 1 \right)$ | **-5.56** | **-5.80** | $-5.93 - (-2.97)$ | $> 3.0$ | Overall low composite contamination factor |

---

## 3. Spatial Hotspot Matrix & Interactive GIS Mapping

An interactive HTML map has been created at [`reports/north_bengal_groundwater_hotspots.html`](file:///home/mosharrof/personal%20Doc/water%20jounal/reports/north_bengal_groundwater_hotspots.html).

### Top 5 Highest Risk Hotspot Wells in North Bengal:

| Rank | Sample ID | District | Thana | WQI | HPI | Heavy Metal Profile | Hydrochemical Facies | Geochemical Driver |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| **1** | **S98_01828** | Pabna | Bera | **156.79** | **129.45** | $\text{As: } 28.7\ \mu\text{g/L, Fe: } 3.75\text{ mg/L, Mn: } 2.21\text{ mg/L}$ | $\text{Ca-HCO}_3$ | Reductive dissolution of Fe-Mn oxides |
| **2** | **S98_01834** | Bogra | Sherpur | **150.04** | **124.41** | $\text{As: } 27.8\ \mu\text{g/L, Fe: } 19.60\text{ mg/L, Mn: } 2.27\text{ mg/L}$ | $\text{Ca-HCO}_3$ | Extreme geogenic Iron-Arsenic release |
| **3** | **S98_01824** | Sirajganj | Sirajganj Sadar | **140.21** | **69.82** | $\text{As: } 15.4\ \mu\text{g/L, Fe: } 3.99\text{ mg/L, Mn: } 0.85\text{ mg/L}$ | $\text{Mixed-HCO}_3$ | Moderately reduced floodplain groundwater |
| **4** | **S98_01823** | Sirajganj | Raiganj | **138.15** | **55.91** | $\text{As: } 12.3\ \mu\text{g/L, Fe: } 3.99\text{ mg/L, Mn: } 0.52\text{ mg/L}$ | $\text{Mixed-HCO}_3$ | Floodplain arsenic enrichment |
| **5** | **S98_01796** | Nilphamari | Domar | **135.88** | **31.25** | $\text{As: } <1.0\ \mu\text{g/L, Fe: } 1.85\text{ mg/L, Mn: } 6.20\text{ mg/L}$ | $\text{Mixed-HCO}_3$ | Extreme localized Manganese hotspot |

---

## 4. Integration into Machine Learning Modeling (Phase 3–6)

With Phase 2 and 2.1 completed:
1. **Target Variables for Track 2 ML:** $\text{HPI}$ and $\text{HEI}$ are now fully computed and available in `data/processed/phase2_risk_indices_results.csv`.
2. **Leakage Prevention Check:** Predictors for Track 2 models will use strictly field parameters ($\text{pH, EC, Depth}$) and major ions ($\text{Ca, Mg, Na, K, Cl, HCO}_3, \text{SO}_4, \text{NO}_3$), with component heavy metals ($\text{As, Fe, Mn, Pb, Ni, Zn}$) **strictly excluded** to guarantee 0% data leakage.
3. We are ready to proceed to **Phase 3: Research Question Finalization & Target Definition**.
