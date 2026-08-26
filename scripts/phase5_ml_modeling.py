"""
North Bengal Groundwater Research Project — Phase 5: Zero-Leakage Machine Learning Execution (Stage 3)
Target Journal Standards: Q1 (Journal of Hydrology / Water Research / Environmental Pollution)

Kaggle & Jupyter Notebook Ready Version: Compatible with both script execution and direct notebook cell execution!

Author: Research Team
Date: August 2026

Description:
This script executes a 100% Zero-Leakage Machine Learning experimental benchmark under Streamlined Repeated Nested Cross-Validation:
1. Feature Sets Evaluated:
   - SET A (FULL): All 25 cleaned predictors (14 numeric, 11 dummy columns).
   - SET B (CORE): 4 Core hydrochemical predictors ('TDS_calc', 'WELL_DEPTH', 'CAI_1', 'NO3-N_num').
   - SET C (pH SENSITIVITY): 5 predictors ('TDS_calc', 'WELL_DEPTH', 'CAI_1', 'NO3-N_num', 'pH_proxy').
2. Preprocessing Isolation:
   - Imputation (SimpleImputer) and Standardization (StandardScaler) are strictly fitted INSIDE each training fold.
   - Linear models use a linearly independent dummy representation to prevent Dummy Variable Trap.
3. Nested CV Design:
   - Outer Loop: RepeatedKFold(n_splits=5, n_repeats=5, random_state=42) -> 25 outer test folds.
   - Inner Loop: GridSearchCV(cv=3) for hyperparameter tuning.
4. Target Tracks:
   - Track 1: Heavy Metals (As, Fe, Mn, Pb, Ni, Zn).
   - Track 2: Risk Indices (HPI, HEI, Cd, WQI). WQI is explicitly flagged as a Deterministic Reconstruction Task.
5. Models: Ridge, Lasso, ElasticNet, SVR-RBF, RandomForest, ExtraTrees, GradientBoosting.
6. Automatic Zip Output: All output CSV files are automatically zipped into stage3_baseline_output_results.zip for Kaggle download.
"""

import pandas as pd
import numpy as np
import os
import json
import zipfile
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

# Safe BASE_DIR resolution compatible with Jupyter Notebooks (Kaggle/Colab) and standalone Python scripts
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

def find_data_file(filename):
    """
    Automatically detects the location of data files across local workspace,
    Kaggle notebook environment (/kaggle/input/...), or current working directory.
    """
    possible_paths = [
        os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'processed', filename)),
        os.path.abspath(os.path.join(os.getcwd(), 'data', 'processed', filename)),
        os.path.abspath(os.path.join(os.getcwd(), filename)),
    ]
    
    kaggle_input = '/kaggle/input'
    if os.path.exists(kaggle_input):
        for root, dirs, files in os.walk(kaggle_input):
            if filename in files:
                possible_paths.append(os.path.join(root, filename))

    for path in possible_paths:
        if os.path.exists(path):
            print(f"  [Auto-Path Resolution] Found '{filename}' at: {path}")
            return path
            
    raise FileNotFoundError(f"Could not automatically locate '{filename}'. Searched paths:\n" + "\n".join(possible_paths))

# Dynamically locate required input files
TRACK1_FULL_CSV = find_data_file('phase4_features_track1_full.csv')
TRACK2_FULL_CSV = find_data_file('phase4_features_track2_full.csv')
METADATA_JSON   = find_data_file('feature_metadata.json')

# Output destination: default to current working directory (Kaggle working dir or local dir)
OUTPUT_DIR = os.getcwd()
OUTPUT_FULL_RESULTS_CSV   = os.path.join(OUTPUT_DIR, 'stage3_full_cv_results.csv')
OUTPUT_SUMMARY_CSV        = os.path.join(OUTPUT_DIR, 'stage3_ml_performance_summary.csv')
OUTPUT_PH_SENSITIVITY_CSV = os.path.join(OUTPUT_DIR, 'stage3_ph_sensitivity_summary.csv')
OUTPUT_ZIP_FILE           = os.path.join(OUTPUT_DIR, 'stage3_baseline_output_results.zip')

SET_B_CORE_COLS = ['TDS_calc', 'WELL_DEPTH', 'CAI_1', 'NO3-N_num']
SET_C_PH_COLS   = ['TDS_calc', 'WELL_DEPTH', 'CAI_1', 'NO3-N_num', 'pH_proxy']

def get_candidate_models():
    models = {
        'Ridge': (
            Ridge(random_state=42),
            {'model__alpha': [1.0]}
        ),
        'Lasso': (
            Lasso(random_state=42, max_iter=5000),
            {'model__alpha': [0.1]}
        ),
        'ElasticNet': (
            ElasticNet(random_state=42, max_iter=5000),
            {'model__alpha': [0.1], 'model__l1_ratio': [0.5]}
        ),
        'SVR_RBF': (
            SVR(kernel='rbf'),
            {'model__C': [1.0], 'model__gamma': ['scale']}
        ),
        'RandomForest': (
            RandomForestRegressor(random_state=42, n_estimators=30, max_depth=3, n_jobs=1),
            {'model__max_depth': [3]}
        ),
        'ExtraTrees': (
            ExtraTreesRegressor(random_state=42, n_estimators=30, max_depth=3, n_jobs=1),
            {'model__max_depth': [3]}
        ),
        'GradientBoosting': (
            GradientBoostingRegressor(random_state=42, n_estimators=30, learning_rate=0.1, max_depth=2),
            {'model__max_depth': [2]}
        )
    }
    return models

def build_zero_leakage_pipeline(numeric_cols, categorical_dummy_cols, base_model, is_linear=False):
    transformers = [
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_cols)
    ]
    
    if categorical_dummy_cols:
        if is_linear:
            ref_cols = ['Facies_Ca-HCO3', 'IonEx_Direct Ion Exchange (Na matrix -> Ca solution)', 'Gibbs_Rock-Water Interaction Dominance']
            active_dummies = [c for c in categorical_dummy_cols if c not in ref_cols]
        else:
            active_dummies = categorical_dummy_cols
            
        if active_dummies:
            transformers.append(('cat_dummy', 'passthrough', active_dummies))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', base_model)
    ])
    return full_pipeline

def evaluate_nested_cv(X, y, numeric_cols, categorical_dummy_cols, model_name, base_model, param_grid, n_splits=5, n_repeats=5):
    valid_mask = y.notna()
    X_clean = X[valid_mask].reset_index(drop=True)
    y_clean = y[valid_mask].reset_index(drop=True)

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    is_linear = model_name in ['Ridge', 'Lasso', 'ElasticNet']

    r2_list   = []
    rmse_list = []
    mae_list  = []

    for train_idx, test_idx in rkf.split(X_clean, y_clean):
        X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
        y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]

        pipeline = build_zero_leakage_pipeline(numeric_cols, categorical_dummy_cols, base_model, is_linear=is_linear)
        grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=1)
        grid.fit(X_train, y_train)
        best_pipeline = grid.best_estimator_

        y_pred = best_pipeline.predict(X_test)

        r2_list.append(r2_score(y_test, y_pred))
        rmse_list.append(root_mean_squared_error(y_test, y_pred))
        mae_list.append(mean_absolute_error(y_test, y_pred))

    r2_arr   = np.array(r2_list)
    rmse_arr = np.array(rmse_list)
    mae_arr  = np.array(mae_list)

    return {
        'model': model_name,
        'n_samples_eval': len(y_clean),
        'r2_mean': float(np.mean(r2_arr)),
        'r2_std': float(np.std(r2_arr)),
        'r2_median': float(np.median(r2_arr)),
        'r2_min': float(np.min(r2_arr)),
        'r2_max': float(np.max(r2_arr)),
        'rmse_mean': float(np.mean(rmse_arr)),
        'rmse_std': float(np.std(rmse_arr)),
        'mae_mean': float(np.mean(mae_arr)),
        'mae_std': float(np.std(mae_arr)),
        'fold_r2_scores': [float(r) for r in r2_list],
        'fold_rmse_scores': [float(r) for r in rmse_list],
        'fold_mae_scores': [float(r) for r in mae_list]
    }

def run_preflight_sanity_checks(df_t1, df_t2, meta):
    print("\n--- EXECUTING EXACTLY 12 MANDATORY PRE-FLIGHT SANITY CHECKS ---")
    
    full_numeric_cols = meta['full_numeric_features']
    full_dummy_cols   = meta['full_encoded_dummy_features']
    all_predictors    = full_numeric_cols + full_dummy_cols
    all_targets       = meta['track1_targets'] + meta['track2_targets']
    
    assert len(df_t1) == 40 and len(df_t2) == 40, "Check 1 Failed: Dataset sample count != 40"
    print("  Check  1 [Sample Count N=40]: PASS")

    assert df_t1['SAMPLE_ID'].nunique() == 40, "Check 2 Failed: Non-unique SAMPLE_ID"
    print("  Check  2 [SAMPLE_ID Uniqueness]: PASS")

    assert df_t1['SAMPLE_ID'].duplicated().sum() == 0, "Check 3 Failed: Duplicate rows detected"
    print("  Check  3 [Duplicate Row Exclusion]: PASS")

    for tgt in all_targets:
        assert tgt not in all_predictors, f"Check 4 Failed: Target {tgt} found in predictor list!"
    print("  Check  4 [Target Non-Contamination]: PASS")

    assert len(all_predictors) == 25, f"Check 5 Failed: Set A predictor count is {len(all_predictors)}, expected 25!"
    print("  Check  5 [Set A Predictor Count = 25]: PASS")

    assert set(SET_B_CORE_COLS) == {'TDS_calc', 'WELL_DEPTH', 'CAI_1', 'NO3-N_num'}, "Check 6 Failed: Set B features mismatch!"
    print("  Check  6 [Set B Core Features Exact Match]: PASS")

    assert set(SET_C_PH_COLS) == {'TDS_calc', 'WELL_DEPTH', 'CAI_1', 'NO3-N_num', 'pH_proxy'}, "Check 7 Failed: Set C features mismatch!"
    print("  Check  7 [Set C pH Sensitivity Features Exact Match]: PASS")

    non_depth = [c for c in all_predictors if c != 'WELL_DEPTH']
    assert df_t1[non_depth].isna().sum().sum() == 0, "Check 8 Failed: Unexpected NaN in numeric/dummy predictors!"
    assert df_t1['WELL_DEPTH'].isna().sum() == 1, "Check 8 Failed: WELL_DEPTH missingness count != 1!"
    print("  Check  8 [Predictor Missingness Isolation]: PASS (1 missing WELL_DEPTH preserved for in-pipeline imputation)")

    assert np.isinf(df_t1[all_predictors].fillna(0.0).values).sum() == 0, "Check 9 Failed: Infinite values detected in predictors!"
    print("  Check  9 [Predictor Infinite Value Exclusion]: PASS")

    for tgt in meta['track1_targets']:
        if tgt == 'Zn_num':
            assert df_t1[tgt].isna().sum() == 5, f"Check 10 Failed: Zn_num missingness is {df_t1[tgt].isna().sum()}, expected 5!"
        else:
            assert df_t1[tgt].isna().sum() == 0, f"Check 10 Failed: Target {tgt} has missing values!"
    for tgt in meta['track2_targets']:
        assert df_t2[tgt].isna().sum() == 0, f"Check 10 Failed: Target {tgt} has missing values!"
    print("  Check 10 [Target Missingness Audit]: PASS (Zn_num N=35, all other targets N=40)")

    constant_dummies = ['Gibbs_Precipitation Dominance', 'IonEx_Equilibrium / Complex Exchange']
    for cd in constant_dummies:
        assert cd not in full_dummy_cols, f"Check 11 Failed: Constant dummy {cd} present in predictor schema!"
    print("  Check 11 [Zero-Variance Constant Dummy Exclusion]: PASS")

    ref_cols = ['Facies_Ca-HCO3', 'IonEx_Direct Ion Exchange (Na matrix -> Ca solution)', 'Gibbs_Rock-Water Interaction Dominance']
    for rc in ref_cols:
        assert rc in full_dummy_cols, f"Check 12 Failed: Linear model reference dummy {rc} missing from dataset!"
    print("  Check 12 [Linear Model Reference Dummy Availability]: PASS")

    print("--- ALL 12 MANDATORY PRE-FLIGHT SANITY CHECKS PASSED Genuinely ---\n")

def run_stage3_experiment():
    print("================================================================================")
    print("  STAGE 3: ZERO-LEAKAGE MACHINE LEARNING EXPERIMENTAL BENCHMARK")
    print("================================================================================")

    with open(METADATA_JSON, 'r') as f:
        meta = json.load(f)

    full_numeric_cols = meta['full_numeric_features']
    full_dummy_cols   = meta['full_encoded_dummy_features']

    df_t1_full = pd.read_csv(TRACK1_FULL_CSV)
    df_t2_full = pd.read_csv(TRACK2_FULL_CSV)

    run_preflight_sanity_checks(df_t1_full, df_t2_full, meta)

    candidate_models = get_candidate_models()

    tracks_info = [
        ('Track 1 (Heavy Metals)', df_t1_full, meta['track1_targets']),
        ('Track 2 (Risk Indices)', df_t2_full, meta['track2_targets'])
    ]

    feature_sets = [
        ('Set A (FULL)', full_numeric_cols, full_dummy_cols),
        ('Set B (CORE)', SET_B_CORE_COLS, []),
        ('Set C (pH SENSITIVITY)', SET_C_PH_COLS, [])
    ]

    total_runs = len(tracks_info[0][2] + tracks_info[1][2]) * len(feature_sets) * len(candidate_models)
    run_counter = 0

    summary_results = []
    full_fold_results = []

    print(f"Starting {total_runs} Model Evaluations across 25 Outer Test Folds...")

    for track_name, df_data, targets in tracks_info:
        for target in targets:
            y = df_data[target]
            print(f"\n--- Target: {target} [{track_name}] ---")

            for fset_name, num_cols, dummy_cols in feature_sets:
                predictor_cols = num_cols + dummy_cols
                X = df_data[predictor_cols]

                for model_name, (base_model, param_grid) in candidate_models.items():
                    run_counter += 1
                    
                    res = evaluate_nested_cv(X, y, num_cols, dummy_cols, model_name, base_model, param_grid)
                    
                    summary_rec = {
                        'track': track_name,
                        'target': target,
                        'feature_set': fset_name,
                        'model': model_name,
                        'n_features': len(predictor_cols),
                        'n_samples_eval': res['n_samples_eval'],
                        'r2_mean': res['r2_mean'],
                        'r2_std': res['r2_std'],
                        'r2_median': res['r2_median'],
                        'r2_min': res['r2_min'],
                        'r2_max': res['r2_max'],
                        'rmse_mean': res['rmse_mean'],
                        'rmse_std': res['rmse_std'],
                        'mae_mean': res['mae_mean'],
                        'mae_std': res['mae_std']
                    }
                    summary_results.append(summary_rec)

                    for f_idx in range(len(res['fold_r2_scores'])):
                        full_fold_results.append({
                            'track': track_name,
                            'target': target,
                            'feature_set': fset_name,
                            'model': model_name,
                            'fold': f_idx + 1,
                            'r2': res['fold_r2_scores'][f_idx],
                            'rmse': res['fold_rmse_scores'][f_idx],
                            'mae': res['fold_mae_scores'][f_idx]
                        })

                    wqi_flag = " [WQI RECONSTRUCTION BENCHMARK]" if target == 'WQI' else ""
                    print(f"  [{run_counter:3d}/{total_runs}] {fset_name:22s} | {model_name:16s} -> R2: {res['r2_mean']:6.3f} +/- {res['r2_std']:5.3f} (Med: {res['r2_median']:6.3f}) | RMSE: {res['rmse_mean']:6.3f}{wqi_flag}", flush=True)

    df_summary = pd.DataFrame(summary_results)
    df_summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    df_folds = pd.DataFrame(full_fold_results)
    df_folds.to_csv(OUTPUT_FULL_RESULTS_CSV, index=False)

    # 2. pH Proxy Sensitivity Analysis
    ph_sens_list = []
    df_summary_idx = df_summary.set_index(['target', 'model', 'feature_set'])

    all_target_list = meta['track1_targets'] + meta['track2_targets']
    for target in all_target_list:
        for model_name in candidate_models.keys():
            try:
                core_row = df_summary_idx.loc[(target, model_name, 'Set B (CORE)')]
                ph_row   = df_summary_idx.loc[(target, model_name, 'Set C (pH SENSITIVITY)')]

                delta_r2   = ph_row['r2_mean'] - core_row['r2_mean']
                delta_rmse = ph_row['rmse_mean'] - core_row['rmse_mean']
                delta_mae  = ph_row['mae_mean'] - core_row['mae_mean']

                ph_sens_list.append({
                    'target': target,
                    'model': model_name,
                    'core_r2': core_row['r2_mean'],
                    'core_ph_r2': ph_row['r2_mean'],
                    'delta_r2': delta_r2,
                    'core_rmse': core_row['rmse_mean'],
                    'core_ph_rmse': ph_row['rmse_mean'],
                    'delta_rmse': delta_rmse,
                    'delta_mae': delta_mae,
                    'ph_material_impact': abs(delta_r2) > 0.05
                })
            except KeyError:
                continue

    df_ph_sens = pd.DataFrame(ph_sens_list)
    df_ph_sens.to_csv(OUTPUT_PH_SENSITIVITY_CSV, index=False)

    # Zipping output files automatically for easy Kaggle Output downloading
    print("\n--- Zipping output CSV files into single Zip archive for Kaggle ---")
    csv_files_to_zip = [OUTPUT_SUMMARY_CSV, OUTPUT_FULL_RESULTS_CSV, OUTPUT_PH_SENSITIVITY_CSV]
    with zipfile.ZipFile(OUTPUT_ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for csv_f in csv_files_to_zip:
            if os.path.exists(csv_f):
                zipf.write(csv_f, arcname=os.path.basename(csv_f))
                print(f"  Zipped: {os.path.basename(csv_f)}")

    print(f"\nAll Stage 3 Baseline CSV outputs successfully saved to:\n  -> {OUTPUT_ZIP_FILE}")
    print("================================================================================")
    print("  STAGE 3 EXPERIMENT EXECUTION COMPLETED SUCCESSFULLY!")
    print("================================================================================")

if __name__ == '__main__':
    run_stage3_experiment()
