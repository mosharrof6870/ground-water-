"""
North Bengal Groundwater Research Project — Phase 1.5 Hydrochemical Characterization
Target Journal Standards: Q1 (Journal of Hydrology / Water Research)

Author: Research Team
Date: August 2026

Description:
This script performs complete Phase 1.5 hydrochemical characterization:
1. Piper Trilinear Diagram Data & Water Type / Hydrochemical Facies Classification.
2. Gibbs Diagram Ratios & Geochemical Controlling Mechanisms (Rock Weathering vs Evaporation vs Precipitation).
3. Calculated TDS from major ion sum: TDS = Ca + Mg + Na + K + Cl + 0.5*HCO3 + SO4 + NO3-N (mg/L).
4. Diagnostic Ion Ratios:
   - Na+/Cl- (Halite Dissolution vs Silicate Weathering vs Ion Exchange)
   - (Ca2++Mg2+)/(HCO3-+SO42-) (Carbonate vs Silicate Weathering)
   - Ca2+/Mg2+ (Calcite vs Dolomite Dissolution)
   - Chloro-Alkaline Indices (CAI-1 & CAI-2) for Direct vs Reverse Ion Exchange.
5. Output dataset saved cleanly to ../data/processed/phase1_5_hydrochemistry_results.csv
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'raw', 'Groundwater_quality_data_Northbengal.xlsx'))
OUTPUT_CSV_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'processed', 'phase1_5_hydrochemistry_results.csv'))

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset file not found at expected path: {DATASET_PATH}")

def load_and_parse_data(file_path):
    """Loads raw Excel file and parses numeric concentrations using DL/2 for BDL values."""
    df_raw = pd.read_excel(file_path, header=None)
    
    col_names = list(df_raw.iloc[1].values)
    col_names[7] = 'SAMPLE_ID'
    
    df_data = df_raw.iloc[3:].copy()
    df_data.columns = col_names
    
    empty_cols = [col for col in df_data.columns if pd.isna(col) or str(col).startswith('Unnamed')]
    df_clean = df_data.drop(columns=empty_cols).reset_index(drop=True)
    
    def parse_dl2(val):
        if pd.isna(val) or str(val).strip() in ['nan', 'None', '']:
            return np.nan
        s = str(val).strip()
        if s.startswith('<'):
            return float(s.replace('<', '').strip()) / 2.0
        try:
            return float(s)
        except:
            return np.nan

    num_cols = ['Ca', 'Mg', 'Na', 'K', 'Cl', 'HCO3', 'SO4', 'NO3-N', 'Fe', 'Mn', 'As', 'Pb', 'Ni', 'Zn', 'Cd', 'Cr', 'Cu']
    for col in num_cols:
        if col in df_clean.columns:
            df_clean[col+'_num'] = df_clean[col].apply(parse_dl2)
            
    return df_clean

def compute_hydrochemistry(df):
    """Calculates meq/L equivalents, Piper percentages, Gibbs ratios, and diagnostic ion ratios."""
    df = df.copy()
    
    df['TDS_calc'] = (df['Ca_num'] + df['Mg_num'] + df['Na_num'] + df['K_num'] + 
                      df['Cl_num'] + 0.5 * df['HCO3_num'] + df['SO4_num'] + df['NO3-N_num'])

    # 1. meq/L Calculations
    df['Ca_meq']   = df['Ca_num']   * (2.0 / 40.078)
    df['Mg_meq']   = df['Mg_num']   * (2.0 / 24.305)
    df['Na_meq']   = df['Na_num']   * (1.0 / 22.990)
    df['K_meq']    = df['K_num']    * (1.0 / 39.098)

    df['Cl_meq']   = df['Cl_num']   * (1.0 / 35.453)
    df['HCO3_meq'] = df['HCO3_num'] * (1.0 / 61.016)
    df['SO4_meq']  = df['SO4_num']  * (2.0 / 96.06)
    df['NO3_meq']  = df['NO3-N_num'] * (1.0 / 14.007)

    df['Cat_sum'] = df['Ca_meq'] + df['Mg_meq'] + df['Na_meq'] + df['K_meq']
    df['An_sum']  = df['Cl_meq'] + df['HCO3_meq'] + df['SO4_meq'] + df['NO3_meq']
    df['CBE_pct'] = np.abs((df['Cat_sum'] - df['An_sum']) / (df['Cat_sum'] + df['An_sum'])) * 100

    # 2. Piper Percentages
    df['pct_Ca'] = (df['Ca_meq'] / df['Cat_sum']) * 100
    df['pct_Mg'] = (df['Mg_meq'] / df['Cat_sum']) * 100
    df['pct_Na_K'] = ((df['Na_meq'] + df['K_meq']) / df['Cat_sum']) * 100

    df['pct_HCO3'] = (df['HCO3_meq'] / df['An_sum']) * 100
    df['pct_Cl']   = (df['Cl_meq'] / df['An_sum']) * 100
    df['pct_SO4_NO3'] = ((df['SO4_meq'] + df['NO3_meq']) / df['An_sum']) * 100

    # 3. Hydrochemical Facies Classification (Piper Water Types)
    def classify_facies(row):
        cat_type = "Ca" if row['pct_Ca'] > 50 else ("Mg" if row['pct_Mg'] > 50 else ("Na" if row['pct_Na_K'] > 50 else "Mixed-Cation"))
        an_type  = "HCO3" if row['pct_HCO3'] > 50 else ("Cl" if row['pct_Cl'] > 50 else ("SO4" if row['pct_SO4_NO3'] > 50 else "Mixed-Anion"))
        return f"{cat_type}-{an_type}"

    df['Facies'] = df.apply(classify_facies, axis=1)

    # 4. Gibbs Ratios
    df['Gibbs_Cation'] = (df['Na_meq'] + df['K_meq']) / (df['Na_meq'] + df['K_meq'] + df['Ca_meq'])
    df['Gibbs_Anion']  = df['Cl_meq'] / (df['Cl_meq'] + df['HCO3_meq'])

    def classify_gibbs(row):
        tds = row['TDS_calc']
        c_ratio = row['Gibbs_Cation']
        if tds > 1000 or c_ratio > 0.7:
            return "Evaporation / Crystallization Dominance"
        elif tds < 10 and c_ratio > 0.5:
            return "Precipitation Dominance"
        else:
            return "Rock-Water Interaction Dominance"

    df['Gibbs_Mechanism'] = df.apply(classify_gibbs, axis=1)

    # 5. Key Diagnostic Ion Ratios
    df['Ratio_Na_Cl'] = df['Na_meq'] / df['Cl_meq']
    df['Ratio_Ca_Mg'] = df['Ca_meq'] / df['Mg_meq']
    df['Ratio_CaMg_HCO3SO4'] = (df['Ca_meq'] + df['Mg_meq']) / (df['HCO3_meq'] + df['SO4_meq'])

    # Chloro-Alkaline Indices (Schoeller CAI-1 & CAI-2)
    df['CAI_1'] = (df['Cl_meq'] - (df['Na_meq'] + df['K_meq'])) / df['Cl_meq']
    df['CAI_2'] = (df['Cl_meq'] - (df['Na_meq'] + df['K_meq'])) / (df['HCO3_meq'] + df['SO4_meq'] + df['NO3_meq'])

    def classify_ion_exchange(row):
        if row['CAI_1'] < 0 and row['CAI_2'] < 0:
            return "Direct Ion Exchange (Na matrix -> Ca solution)"
        elif row['CAI_1'] > 0 and row['CAI_2'] > 0:
            return "Reverse Ion Exchange (Ca matrix -> Na solution)"
        else:
            return "Equilibrium / Complex Exchange"

    df['Ion_Exchange_Process'] = df.apply(classify_ion_exchange, axis=1)

    return df

def main():
    print(f"Loading data from relative path: {DATASET_PATH}")
    df_clean = load_and_parse_data(DATASET_PATH)
    df_hydro = compute_hydrochemistry(df_clean)

    print(f"\n=== HYDROCHEMICAL FACIES SUMMARY (Piper Diagram Classification, N={len(df_hydro)}) ===")
    facies_counts = df_hydro['Facies'].value_counts()
    for facies, count in facies_counts.items():
        print(f"  {facies}: {count} samples ({count/len(df_hydro)*100:.1f}%)")

    print(f"\n=== GIBBS GEOCHEMICAL MECHANISM SUMMARY ===")
    gibbs_counts = df_hydro['Gibbs_Mechanism'].value_counts()
    for mech, count in gibbs_counts.items():
        print(f"  {mech}: {count} samples ({count/len(df_hydro)*100:.1f}%)")

    print(f"\n=== ION EXCHANGE PROCESS SUMMARY (Schoeller CAI) ===")
    cai_counts = df_hydro['Ion_Exchange_Process'].value_counts()
    for proc, count in cai_counts.items():
        print(f"  {proc}: {count} samples ({count/len(df_hydro)*100:.1f}%)")

    # Deep-Dive: Facies & Ratios of the 7 High-CBE (>10%) Samples
    high_cbe_ids = ['S98_01797', 'S98_01815', 'S98_01818', 'S98_01819', 'S98_01820', 'S98_01821', 'S98_01826']
    df_high = df_hydro[df_hydro['SAMPLE_ID'].isin(high_cbe_ids)].copy()

    print("\n=== PHASE 1.5 DEEP-DIVE: HIGH-CBE (>10%) SAMPLES GEOCHEMICAL FACIES & RATIOS ===")
    cols_to_print = ['SAMPLE_ID', 'DISTRICT', 'TDS_calc', 'CBE_pct', 'Facies', 'Ratio_Na_Cl', 'Ratio_CaMg_HCO3SO4', 'CAI_1', 'Ion_Exchange_Process']
    print(df_high[cols_to_print].to_string(index=False))

    # Ensure target output directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    df_hydro.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nPhase 1.5 Hydrochemical dataset successfully saved to: {OUTPUT_CSV_PATH}")

if __name__ == '__main__':
    main()
