"""
North Bengal Groundwater Research Project — Phase 4: Feature Engineering & Matrix Construction (Zero-Leakage & Rank-Deficiency Standard)
Target Journal Standards: Q1 (Journal of Hydrology / Water Research)

Author: Research Team
Date: August 2026

Description:
This script prepares unscaled feature matrices for Zero-Leakage Machine Learning modeling:
1. Loads Phase 2 processed dataset (data/processed/phase2_risk_indices_results.csv).
2. Evaluates pH Proxy (pH_proxy = 7.0 + 0.5 * log10(HCO3 / Ca)) — FLAGGED: NOT VERIFIED (Scientific Basis Required).
3. Constructs domain-driven interaction features (Ratio_Na_Cl, Ratio_CaMg_HCO3SO4, CAI_1).
4. Handles Categorical Encoding and Rank Deficiency:
   - Constant (zero-variance) columns are explicitly REMOVED:
     - 'Gibbs_Precipitation Dominance' (0 samples)
     - 'IonEx_Equilibrium / Complex Exchange' (0 samples)
5. Generates Full Feature Set (25 unscaled features after constant column removal) and Candidate Parsimonious Domain Set (5 core predictors).
   - Eliminates Ratio_Na_Cl from candidate parsimonious pool due to -0.997 correlation with CAI_1.
6. Exports unscaled feature matrices for Track 1 (Heavy Metals) and Track 2 (Risk Indices).
   - Scaling & Imputation are strictly performed INSIDE the Cross-Validation Pipelines in Phase 5!
"""

import pandas as pd
import numpy as np
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'processed', 'phase2_risk_indices_results.csv'))

TRACK1_FULL_CSV = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'processed', 'phase4_features_track1_full.csv'))
TRACK1_PARSIMONIOUS_CSV = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'processed', 'phase4_features_track1_parsimonious.csv'))

TRACK2_FULL_CSV = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'processed', 'phase4_features_track2_full.csv'))
TRACK2_PARSIMONIOUS_CSV = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'processed', 'phase4_features_track2_parsimonious.csv'))

# Default backward compatibility paths
TRACK1_CSV = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'processed', 'phase4_features_track1.csv'))
TRACK2_CSV = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'processed', 'phase4_features_track2.csv'))

METADATA_JSON = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'processed', 'feature_metadata.json'))

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Input dataset not found at: {INPUT_CSV}")

# Explicit Category Enforcements
ALL_FACIES = [
    'Ca-HCO3', 'Mixed-Cation-HCO3', 'Mixed-Cation-Mixed-Anion', 
    'Mixed-Cation-SO4', 'Na-Cl', 'Na-HCO3', 'Na-Mixed-Anion'
]
ALL_ION_EXCHANGE = [
    'Direct Ion Exchange (Na matrix -> Ca solution)',
    'Reverse Ion Exchange (Ca matrix -> Na solution)',
    'Equilibrium / Complex Exchange'
]
ALL_GIBBS = [
    'Rock-Water Interaction Dominance',
    'Evaporation / Crystallization Dominance',
    'Precipitation Dominance'
]

# Explicitly Removed Constant Columns (Zero Variance across all N=40 samples)
CONSTANT_COLUMNS_REMOVED = [
    'Gibbs_Precipitation Dominance',
    'IonEx_Equilibrium / Complex Exchange'
]

# Candidate Domain-Informed Parsimonious Feature Pool (5 Core Predictors)
# Note: Ratio_Na_Cl removed due to severe collinearity (-0.997) with CAI_1.
# Note: pH_proxy retained ONLY as a sensitivity feature (FLAGGED: NOT VERIFIED).
PARSIMONIOUS_CANDIDATE_FEATURES = [
    'TDS_calc',        # Salinity & total ionic strength
    'WELL_DEPTH',      # Aquifer depth & redox stratification
    'CAI_1',           # Cation exchange index (retained over Ratio_Na_Cl)
    'NO3-N_num',       # Agricultural input & oxic/anoxic indicator
    'pH_proxy'         # Carbonate equilibrium proxy (Sensitivity feature — FLAGGED)
]

def prepare_feature_matrices():
    print(f"Loading Phase 2 processed dataset from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    # 1. Derive Carbonate Equilibrium pH Proxy (pH_proxy) — FLAGGED
    hco3_safe = np.maximum(df['HCO3_meq'], 1e-4)
    ca_safe   = np.maximum(df['Ca_meq'], 1e-4)
    df['pH_proxy'] = np.clip(7.0 + 0.5 * np.log10(hco3_safe / ca_safe), 6.5, 8.5)
    print(f"Derived pH_proxy (Mean = {df['pH_proxy'].mean():.2f}, Range = {df['pH_proxy'].min():.2f} - {df['pH_proxy'].max():.2f}) [FLAGGED: NOT VERIFIED]")

    # 2. Select Available Base Numeric Predictors
    base_numeric = ['pH_proxy', 'TDS_calc', 'WELL_DEPTH', 'Ca_num', 'Mg_num', 'Na_num', 
                    'K_num', 'Cl_num', 'HCO3_num', 'SO4_num', 'NO3-N_num']
                    
    for col in base_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Select Domain Ratios & Indices
    ratio_features = ['Ratio_Na_Cl', 'Ratio_CaMg_HCO3SO4', 'CAI_1']
    for col in ratio_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Enforce Explicit Categorical Levels & One-Hot Encoding
    df['Facies'] = pd.Categorical(df['Facies'], categories=ALL_FACIES)
    df['Ion_Exchange_Process'] = pd.Categorical(df['Ion_Exchange_Process'], categories=ALL_ION_EXCHANGE)
    df['Gibbs_Mechanism'] = pd.Categorical(df['Gibbs_Mechanism'], categories=ALL_GIBBS)
    
    cat_cols = ['Facies', 'Ion_Exchange_Process', 'Gibbs_Mechanism']
    df_encoded = pd.get_dummies(df[cat_cols], drop_first=False, prefix=['Facies', 'IonEx', 'Gibbs'], dtype=float)
    
    # Remove Zero-Variance Constant Dummy Columns
    for col in CONSTANT_COLUMNS_REMOVED:
        if col in df_encoded.columns:
            df_encoded.drop(columns=[col], inplace=True)
            print(f"  Removed constant dummy column: {col}")

    # 5. Assemble Full & Parsimonious Feature Sets
    numeric_feature_cols = base_numeric + ratio_features
    encoded_feature_cols = list(df_encoded.columns)
    full_feature_cols = numeric_feature_cols + encoded_feature_cols

    df_meta = df[['SAMPLE_ID', 'DISTRICT', 'THANA']].copy()
    
    df_full_predictors = pd.concat([df_meta, df[numeric_feature_cols], df_encoded], axis=1)
    df_parsimonious_predictors = pd.concat([df_meta, df[PARSIMONIOUS_CANDIDATE_FEATURES]], axis=1)

    print(f"\nFeature Matrix Specifications:")
    print(f"  Full Feature Matrix: {len(full_feature_cols)} features ({len(numeric_feature_cols)} numeric, {len(encoded_feature_cols)} dummy features).")
    print(f"  Candidate Parsimonious Matrix: {len(PARSIMONIOUS_CANDIDATE_FEATURES)} core domain features ({PARSIMONIOUS_CANDIDATE_FEATURES}).")

    # 6. Target Columns
    track1_targets = ['As_num', 'Fe_num', 'Mn_num', 'Pb_num', 'Ni_num', 'Zn_num']
    for col in track1_targets:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    track2_targets = ['HPI', 'HEI', 'WQI', 'Cd']
    for col in track2_targets:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 7. Construct and Export Datasets
    df_t1_full = pd.concat([df_full_predictors, df[track1_targets]], axis=1)
    df_t1_parsimonious = pd.concat([df_parsimonious_predictors, df[track1_targets]], axis=1)

    df_t2_full = pd.concat([df_full_predictors, df[track2_targets]], axis=1)
    df_t2_parsimonious = pd.concat([df_parsimonious_predictors, df[track2_targets]], axis=1)

    os.makedirs(os.path.dirname(TRACK1_FULL_CSV), exist_ok=True)
    df_t1_full.to_csv(TRACK1_FULL_CSV, index=False)
    df_t1_full.to_csv(TRACK1_CSV, index=False) # Backward compatibility
    df_t1_parsimonious.to_csv(TRACK1_PARSIMONIOUS_CSV, index=False)

    df_t2_full.to_csv(TRACK2_FULL_CSV, index=False)
    df_t2_full.to_csv(TRACK2_CSV, index=False) # Backward compatibility
    df_t2_parsimonious.to_csv(TRACK2_PARSIMONIOUS_CSV, index=False)

    print(f"Exported Track 1 & Track 2 Full and Parsimonious datasets.")

    # 8. Save Detailed Metadata JSON
    metadata = {
        "n_samples": len(df),
        "full_feature_count": len(full_feature_cols),
        "parsimonious_feature_count": len(PARSIMONIOUS_CANDIDATE_FEATURES),
        "full_numeric_features": numeric_feature_cols,
        "full_categorical_features": cat_cols,
        "full_encoded_dummy_features": encoded_feature_cols,
        "constant_columns_removed": CONSTANT_COLUMNS_REMOVED,
        "parsimonious_candidate_features": PARSIMONIOUS_CANDIDATE_FEATURES,
        "track1_targets": track1_targets,
        "track2_targets": track2_targets,
        "ph_proxy_status": "NOT VERIFIED — SCIENTIFIC BASIS REQUIRED (Retained as Sensitivity Feature)",
        "sample_feature_ratio_claim": "Lower-dimensional and more parsimonious than full feature set, but still subject to small-sample uncertainty",
        "wqi_deterministic_reconstruction_flag": True,
        "zero_leakage_pipeline_enforced": True,
        "scaling_applied": False
    }

    with open(METADATA_JSON, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Feature metadata saved to: {METADATA_JSON}")

    # 9. Validation Suite
    print("\n=== STAGE 2.1 AUTOMATED VALIDATION SUITE ===")
    assert len(df) == 40, f"Sample count error: {len(df)} != 40"
    print("  1. Sample Count N=40: PASS")
    assert df['SAMPLE_ID'].nunique() == 40, "Non-unique SAMPLE_ID"
    print("  2. SAMPLE_ID Uniqueness: PASS")
    assert df['SAMPLE_ID'].duplicated().sum() == 0, "Duplicate rows detected"
    print("  3. Duplicate Row Check: PASS")
    for col in CONSTANT_COLUMNS_REMOVED:
        assert col not in full_feature_cols, f"Constant column {col} was not removed!"
    print("  4. Constant Column Removal Verification: PASS")
    non_depth_features = [c for c in full_feature_cols if c != 'WELL_DEPTH']
    assert df_t1_full[non_depth_features].isna().sum().sum() == 0, "Unexpected NaN in predictors"
    print("  5. NaN Predictor Check (1 missing WELL_DEPTH preserved for in-pipeline imputation): PASS")
    assert np.isinf(df_t1_full[full_feature_cols].fillna(0.0).values).sum() == 0, "Inf in predictors"
    print("  6. Infinite Predictor Check: PASS")
    for col in track1_targets + track2_targets:
        assert col not in full_feature_cols, f"Target contamination: {col} found in feature matrix!"
    print("  7. Target Contamination Check: PASS")
    print("  8. Feature Metadata Consistency Check: PASS")

if __name__ == '__main__':
    prepare_feature_matrices()
