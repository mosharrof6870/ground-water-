"""
North Bengal Groundwater Research Project — Stage 3.1: Controlled Target-Transformation Sensitivity Experiment
Target Journal Standards: Q1 (Journal of Hydrology / Water Research / Environmental Pollution)

Kaggle & Jupyter Notebook Ready Version (Robust against negative target domains like Degree of Contamination Cd)

Author: Research Team
Date: August 2026

Description:
Controlled comparative audit (Condition A: RAW TARGET vs Condition B: LOG1P TARGET) to evaluate:
1. Target Distribution Audit (N, Missing, Min, Max, Mean, Median, Std, Skewness, IQR, Max/Median Ratio).
2. Domain Safety check: Targets with min < 0 (e.g. Cd Degree of Contamination) are flagged as LOG1P INVALID.
3. Out-of-Sample Spearman Rank Association (rho) alongside out-of-fold R2, RMSE, MAE, NRMSE.
4. Zero-leakage target transformation strictly inside CV using TransformedTargetRegressor.
5. Auto-zipping all produced CSV results into stage3_1_output_results.zip for Kaggle output download.
"""

import pandas as pd
import numpy as np
import os
import json
import zipfile
import warnings
from scipy.stats import skew, spearmanr
warnings.filterwarnings('ignore')

from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
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

OUTPUT_DIR = os.getcwd()
OUTPUT_DISTRIBUTION_CSV    = os.path.join(OUTPUT_DIR, 'stage3_1_target_distribution_audit.csv')
OUTPUT_FULL_BENCHMARK_CSV  = os.path.join(OUTPUT_DIR, 'stage3_1_full_benchmark_results.csv')
OUTPUT_COMPARISON_CSV      = os.path.join(OUTPUT_DIR, 'stage3_1_raw_vs_log1p_comparison.csv')
OUTPUT_DECISION_MATRIX_CSV = os.path.join(OUTPUT_DIR, 'stage3_1_decision_matrix.csv')
OUTPUT_ZIP_FILE            = os.path.join(OUTPUT_DIR, 'stage3_1_output_results.zip')

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

def run_section1_distribution_audit(df_t1, df_t2, meta):
    print("\n================================================================================")
    print("  STAGE 3.1 — SECTION 1: TARGET DISTRIBUTION AUDIT")
    print("================================================================================")

    audit_records = []
    all_targets_info = [
        ('Track 1', df_t1, meta['track1_targets']),
        ('Track 2', df_t2, meta['track2_targets'])
    ]

    for track_name, df_data, targets in all_targets_info:
        for tgt in targets:
            series = df_data[tgt].dropna()
            n_obs = len(series)
            n_miss = df_data[tgt].isna().sum()
            min_val = series.min()
            max_val = series.max()
            mean_val = series.mean()
            median_val = series.median()
            std_val = series.std()
            skew_val = skew(series)
            q25 = series.quantile(0.25)
            q75 = series.quantile(0.75)
            iqr_val = q75 - q25
            max_med_ratio = max_val / median_val if median_val > 0 else np.nan
            
            # Log1p is empirically justified if skewness > 1.0 OR max/median ratio > 3.0 (and min >= 0)
            empirically_justified = bool((skew_val > 1.0 or (not np.isnan(max_med_ratio) and max_med_ratio > 3.0)) and min_val >= 0)

            audit_records.append({
                'target': tgt,
                'track': track_name,
                'N': n_obs,
                'missing_count': n_miss,
                'min': min_val,
                'max': max_val,
                'mean': mean_val,
                'median': median_val,
                'std': std_val,
                'skewness': skew_val,
                'q25': q25,
                'q75': q75,
                'IQR': iqr_val,
                'max_median_ratio': max_med_ratio,
                'log1p_empirically_justified': empirically_justified
            })

    df_audit = pd.DataFrame(audit_records)
    df_audit.to_csv(OUTPUT_DISTRIBUTION_CSV, index=False)
    print(f"\nTarget Distribution Audit Exported to: {OUTPUT_DISTRIBUTION_CSV}\n")
    print(df_audit[['target', 'N', 'min', 'mean', 'median', 'skewness', 'IQR', 'log1p_empirically_justified']].to_string(index=False))
    return df_audit

def build_zero_leakage_pipeline(numeric_cols, categorical_dummy_cols, base_model, is_linear=False, transform_target=False):
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

    feature_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', base_model)
    ])

    if transform_target:
        # Pass check_inverse=False to avoid unnecessary internal round-trip warnings
        return TransformedTargetRegressor(
            regressor=feature_pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False
        )
    return feature_pipeline

def evaluate_target_sensitivity(X, y, numeric_cols, categorical_dummy_cols, model_name, base_model, param_grid, transform_target=False, n_splits=5, n_repeats=5):
    valid_mask = y.notna()
    X_clean = X[valid_mask].reset_index(drop=True)
    y_clean = y[valid_mask].reset_index(drop=True)

    # Domain check: If y has values <= -1.0 or < 0.0, log1p(y) produces NaN or complex values.
    if transform_target and y_clean.min() < 0:
        return {
            'model': model_name,
            'transform': 'LOG1P',
            'n_samples_eval': len(y_clean),
            'r2_mean': np.nan,
            'r2_std': np.nan,
            'r2_median': np.nan,
            'r2_min': np.nan,
            'r2_max': np.nan,
            'rmse_mean': np.nan,
            'mae_mean': np.nan,
            'nrmse_iqr': np.nan,
            'spearman_rho': np.nan,
            'spearman_pval': np.nan,
            'fold_r2_scores': [],
            'skipped_reason': f"Log1p Mathematically Undefined (Target min={y_clean.min():.4f} < 0)"
        }

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    is_linear = model_name in ['Ridge', 'Lasso', 'ElasticNet']

    if transform_target:
        inner_param_grid = {f'regressor__{k}': v for k, v in param_grid.items()}
    else:
        inner_param_grid = param_grid

    r2_list   = []
    rmse_list = []
    mae_list  = []
    
    oof_predictions = np.zeros(len(y_clean))
    oof_counts      = np.zeros(len(y_clean))

    for train_idx, test_idx in rkf.split(X_clean, y_clean):
        X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
        y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]

        pipeline = build_zero_leakage_pipeline(numeric_cols, categorical_dummy_cols, base_model, is_linear=is_linear, transform_target=transform_target)
        grid = GridSearchCV(pipeline, inner_param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=1)
        grid.fit(X_train, y_train)
        best_pipeline = grid.best_estimator_

        y_pred = best_pipeline.predict(X_test)

        r2_list.append(r2_score(y_test, y_pred))
        rmse_list.append(root_mean_squared_error(y_test, y_pred))
        mae_list.append(mean_absolute_error(y_test, y_pred))

        oof_predictions[test_idx] += y_pred
        oof_counts[test_idx] += 1

    oof_preds_avg = oof_predictions / oof_counts
    r2_arr   = np.array(r2_list)
    rmse_arr = np.array(rmse_list)
    mae_arr  = np.array(mae_list)

    # Spearman rank correlation on concatenated out-of-fold predictions
    rho, pval = spearmanr(oof_preds_avg, y_clean)
    
    # Calculate IQR for NRMSE
    q75, q25 = np.percentile(y_clean, [75, 25])
    iqr_val = q75 - q25
    nrmse_val = np.mean(rmse_arr) / iqr_val if iqr_val != 0 else np.nan

    return {
        'model': model_name,
        'transform': 'LOG1P' if transform_target else 'RAW',
        'n_samples_eval': len(y_clean),
        'r2_mean': float(np.mean(r2_arr)),
        'r2_std': float(np.std(r2_arr)),
        'r2_median': float(np.median(r2_arr)),
        'r2_min': float(np.min(r2_arr)),
        'r2_max': float(np.max(r2_arr)),
        'rmse_mean': float(np.mean(rmse_arr)),
        'mae_mean': float(np.mean(mae_arr)),
        'nrmse_iqr': float(nrmse_val),
        'spearman_rho': float(rho),
        'spearman_pval': float(pval),
        'fold_r2_scores': [float(r) for r in r2_list],
        'skipped_reason': None
    }

def run_stage3_1_sensitivity_experiment():
    print("================================================================================")
    print("  STAGE 3.1 — SECTION 2 & 3: CONTROLLED TARGET-TRANSFORMATION EXPERIMENT")
    print("================================================================================")

    with open(METADATA_JSON, 'r') as f:
        meta = json.load(f)

    full_numeric_cols = meta['full_numeric_features']
    full_dummy_cols   = meta['full_encoded_dummy_features']

    df_t1_full = pd.read_csv(TRACK1_FULL_CSV)
    df_t2_full = pd.read_csv(TRACK2_FULL_CSV)

    # Section 1 Distribution Audit
    df_audit = run_section1_distribution_audit(df_t1_full, df_t2_full, meta)

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

    benchmark_records = []
    total_evals = len(tracks_info[0][2] + tracks_info[1][2]) * len(feature_sets) * len(candidate_models) * 2
    eval_counter = 0

    print(f"\nExecuting {total_evals} Model Evaluations across 25 Outer Test Folds (Condition A: RAW vs Condition B: LOG1P)...")

    for track_name, df_data, targets in tracks_info:
        for target in targets:
            y = df_data[target]
            print(f"\n--- Target: {target} [{track_name}] ---")

            for fset_name, num_cols, dummy_cols in feature_sets:
                predictor_cols = num_cols + dummy_cols
                X = df_data[predictor_cols]

                for model_name, (base_model, param_grid) in candidate_models.items():
                    for transform_flag in [False, True]:
                        eval_counter += 1
                        cond_label = "LOG1P" if transform_flag else "RAW"
                        
                        res = evaluate_target_sensitivity(
                            X, y, num_cols, dummy_cols, model_name, base_model, param_grid, transform_target=transform_flag
                        )

                        rec = {
                            'track': track_name,
                            'target': target,
                            'feature_set': fset_name,
                            'model': model_name,
                            'condition': cond_label,
                            'n_features': len(predictor_cols),
                            'n_samples_eval': res['n_samples_eval'],
                            'r2_mean': res['r2_mean'],
                            'r2_std': res['r2_std'],
                            'r2_median': res['r2_median'],
                            'r2_min': res['r2_min'],
                            'r2_max': res['r2_max'],
                            'rmse_mean': res['rmse_mean'],
                            'mae_mean': res['mae_mean'],
                            'nrmse_iqr': res['nrmse_iqr'],
                            'spearman_rho': res['spearman_rho'],
                            'spearman_pval': res['spearman_pval'],
                            'skipped_reason': res.get('skipped_reason', None)
                        }
                        benchmark_records.append(rec)

                        if res.get('skipped_reason'):
                            print(f"  [{eval_counter:3d}/{total_evals}] {cond_label:5s} | {fset_name:22s} | {model_name:16s} -> SKIPPED ({res['skipped_reason']})", flush=True)
                        else:
                            print(f"  [{eval_counter:3d}/{total_evals}] {cond_label:5s} | {fset_name:22s} | {model_name:16s} -> R2: {res['r2_mean']:6.3f} (Med: {res['r2_median']:6.3f}) | RMSE: {res['rmse_mean']:6.3f} | NRMSE: {res['nrmse_iqr']:5.2f} | Spearman Rho: {res['spearman_rho']:+5.3f}", flush=True)

    df_benchmark = pd.DataFrame(benchmark_records)
    df_benchmark.to_csv(OUTPUT_FULL_BENCHMARK_CSV, index=False)

    # Calculate Comparison & Delta CSV
    comp_list = []
    for (tgt, fset, mdl), group in df_benchmark.groupby(['target', 'feature_set', 'model']):
        raw_row = group[group['condition'] == 'RAW'].iloc[0]
        log_row = group[group['condition'] == 'LOG1P'].iloc[0]

        if pd.isna(log_row['r2_mean']):
            delta_r2 = np.nan
            delta_rmse = np.nan
            delta_mae = np.nan
            delta_rho = np.nan
        else:
            delta_r2 = log_row['r2_mean'] - raw_row['r2_mean']
            delta_rmse = log_row['rmse_mean'] - raw_row['rmse_mean']
            delta_mae = log_row['mae_mean'] - raw_row['mae_mean']
            delta_rho = log_row['spearman_rho'] - raw_row['spearman_rho']

        comp_list.append({
            'target': tgt,
            'feature_set': fset,
            'model': mdl,
            'raw_r2': raw_row['r2_mean'],
            'log1p_r2': log_row['r2_mean'],
            'delta_r2': delta_r2,
            'raw_rmse': raw_row['rmse_mean'],
            'log1p_rmse': log_row['rmse_mean'],
            'delta_rmse': delta_rmse,
            'raw_spearman_rho': raw_row['spearman_rho'],
            'log1p_spearman_rho': log_row['spearman_rho'],
            'delta_spearman_rho': delta_rho
        })

    df_comp = pd.DataFrame(comp_list)
    df_comp.to_csv(OUTPUT_COMPARISON_CSV, index=False)

    # Section 6: Generate Decision Matrix
    decision_records = []
    for tgt, group in df_comp.groupby('target'):
        justified_audit = df_audit[df_audit['target'] == tgt]['log1p_empirically_justified'].values[0]
        tgt_min = df_audit[df_audit['target'] == tgt]['min'].values[0]
        
        if tgt_min < 0:
            decision = 'KEEP RAW'
            reasoning = f'LOG1P INVALID: Target takes negative values (min={tgt_min:.4f} < 0) when concentrations are below evaluation limits; log1p is mathematically undefined.'
        else:
            avg_delta_r2 = group['delta_r2'].mean()
            avg_delta_rho = group['delta_spearman_rho'].mean()
            positive_rho_count = (group['log1p_spearman_rho'] > 0.2).sum()
            
            if avg_delta_r2 > 0.05 and avg_delta_rho > 0.05:
                decision = 'KEEP LOG1P'
                reasoning = 'Consistent out-of-sample improvement in both R2 and Spearman rank correlation.'
            elif avg_delta_rho > 0.05 and positive_rho_count >= 5:
                decision = 'LOG1P SENSITIVITY ONLY'
                reasoning = 'Improved rank correlation despite negative baseline R2 due to small N=40 noise.'
            elif avg_delta_r2 < -0.05:
                decision = 'KEEP RAW'
                reasoning = 'Log1p transformation deteriorated out-of-sample prediction stability.'
            else:
                decision = 'INSUFFICIENT EVIDENCE'
                reasoning = 'Negligible difference; small-sample instability dominates.'

        decision_records.append({
            'target': tgt,
            'log1p_empirically_justified': justified_audit,
            'target_min': tgt_min,
            'final_decision': decision,
            'decision_reasoning': reasoning
        })

    df_decision = pd.DataFrame(decision_records)
    df_decision.to_csv(OUTPUT_DECISION_MATRIX_CSV, index=False)

    print("\n================================================================================")
    print("  STAGE 3.1 FINAL DECISION MATRIX SUMMARY")
    print("================================================================================")
    print(df_decision[['target', 'target_min', 'log1p_empirically_justified', 'final_decision', 'decision_reasoning']].to_string(index=False))

    # Zipping output files automatically for easy Kaggle Output downloading
    print("\n--- Zipping output CSV files into single Zip archive for Kaggle Output section ---")
    csv_files_to_zip = [
        OUTPUT_DISTRIBUTION_CSV,
        OUTPUT_FULL_BENCHMARK_CSV,
        OUTPUT_COMPARISON_CSV,
        OUTPUT_DECISION_MATRIX_CSV
    ]
    with zipfile.ZipFile(OUTPUT_ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for csv_f in csv_files_to_zip:
            if os.path.exists(csv_f):
                zipf.write(csv_f, arcname=os.path.basename(csv_f))
                print(f"  Zipped: {os.path.basename(csv_f)}")

    print(f"\nAll Stage 3.1 CSV outputs successfully zipped to:\n  -> {OUTPUT_ZIP_FILE}")
    print("================================================================================")
    print("  STAGE 3.1 SENSITIVITY EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("================================================================================")

if __name__ == '__main__':
    run_stage3_1_sensitivity_experiment()
