"""
Configuration file for Groundwater AI Screening Web Application.
Centralizes research thresholds, model paths, validation rules, and training domain bounds.
"""
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 1. Regulatory Heavy Metal Screening Thresholds (µg/L)
THRESHOLDS = {
    "Ni": 20.0,  # Nickel research screening threshold (µg/L)
    "Cd": 3.0    # Cadmium research screening threshold (µg/L)
}

# 2. Model Files and Feature Contracts (Inference Only)
MODEL_CONFIG = {
    "ni": {
        "name": "Nickel (Ni)",
        "symbol": "Ni",
        "unit": "µg/L",
        "threshold": THRESHOLDS["Ni"],
        "model_type": "HuberRegressor",
        "model_path": os.path.join(MODELS_DIR, "ni_model", "FINAL_MODEL_NI.joblib"),
        "conformal_path": os.path.join(MODELS_DIR, "ni_model", "CONFORMAL_NI.csv"),
        "feature_order": ["pH_proxy", "TDS_calc", "NO3-N_num", "WELL_DEPTH"],
        "training_n_imputed": 40,
        "training_n_complete_case": 39,
        "oof_r2": 0.2240,               # Alias for backwards compatibility
        "oof_r2_primary": 0.2240,       # 5x5 Repeated Nested CV Mean (Primary Research Metric)
        "oof_r2_secondary": 0.2271,     # Single-Pass Global OOF Concatenation (Secondary Metric)
        "oof_r2_diff": 0.0031,          # Actual Difference (0.2271 - 0.2240)
        "oof_rmse": 1.771,
        "spearman": 0.633,
        "conformal_nominal_coverage": "90%",
        "conformal_observed_coverage": "90.0% (36/40 samples covered)",
        "conformal_wording": "90% nominal conformal prediction interval; observed coverage in evaluation artifact = 90.0% (36/40).",
        "primary_predictor": "TDS (mg/L)",
        "interpretation": "TDS is the primary predictive variable in the trained Ni HuberRegressor model.",
        "performance_note": "Limited quantitative predictive performance (Primary 5x5 CV OOF R² = 0.2240)",
        "confidence_level": "MODERATE SCREENING CONFIDENCE"
    },
    "cd": {
        "name": "Cadmium (Cd)",
        "symbol": "Cd",
        "unit": "µg/L",
        "threshold": THRESHOLDS["Cd"],
        "model_type": "LinearRegression",
        "model_path": os.path.join(MODELS_DIR, "cd_model", "FINAL_MODEL_CD.joblib"),
        "conformal_path": os.path.join(MODELS_DIR, "cd_model", "CONFORMAL_CD.csv"),
        "feature_order": ["pH_proxy", "TDS_calc", "NO3-N_num", "WELL_DEPTH"],
        "training_n_imputed": 40,
        "training_n_complete_case": 39,
        "sample_exclusion_note": "Sample S98_01798 has 1 missing WELL_DEPTH; complete-case domain profiling used N=39, while model estimator trained on N=40 via median imputer.",
        "oof_r2": 0.7061,               # Alias for backwards compatibility
        "oof_r2_primary": 0.7061,       # 5x5 Repeated Nested CV Mean (Primary Research Metric)
        "oof_r2_secondary": 0.7158,     # Single-Pass Global OOF Concatenation (Secondary Metric)
        "oof_r2_diff": 0.0097,          # Actual Difference (0.7158 - 0.7061)
        "oof_rmse": 0.066,
        "spearman": 0.685,
        "conformal_nominal_coverage": "90%",
        "conformal_observed_coverage": "90.0% (36/40 samples covered)",
        "conformal_wording": "90% nominal CV+ conformal prediction interval; observed coverage in evaluation artifact = 90.0% (36/40).",
        "primary_predictor": "NO3-N (mg/L)",
        "interpretation": "NO3-N is the primary predictive variable in the trained Cd LinearRegression model.",
        "performance_note": "Stronger quantitative predictive performance (Primary 5x5 CV OOF R² = 0.7061)",
        "confidence_level": "HIGHER SCREENING CONFIDENCE"
    }
}

# 3. Load Exact Empirical Training Domain Statistics dynamically
DOMAIN_BOUNDS_FILE = os.path.join(MODELS_DIR, "domain_bounds.json")

if os.path.exists(DOMAIN_BOUNDS_FILE):
    with open(DOMAIN_BOUNDS_FILE, "r") as f:
        TRAINING_DOMAIN = json.load(f)
else:
    TRAINING_DOMAIN = {
        "features": {
            "pH_proxy": {"min": 6.868, "max": 7.318, "mean": 7.086, "std": 0.102},
            "TDS_calc": {"min": 44.5, "max": 593.4, "mean": 229.776, "std": 173.861},
            "NO3-N_num": {"min": 0.10, "max": 12.50, "mean": 1.269, "std": 2.455},
            "WELL_DEPTH": {"min": 9.0, "max": 61.0, "mean": 22.615, "std": 9.450}
        },
        "mean_vector": [7.0859, 229.776, 1.2692, 22.615],
        "inv_covariance": None,
        "mahalanobis_threshold_95": 3.3347,
        "n_samples": 39
    }

# Extract Model Validated Input Range directly from training dataset statistics
if TRAINING_DOMAIN and "features" in TRAINING_DOMAIN:
    tf = TRAINING_DOMAIN["features"]
    VALIDATED_RANGES = {
        "pH": {"min": round(tf["pH_proxy"]["min"], 2), "max": round(tf["pH_proxy"]["max"], 2), "unit": "-"},
        "TDS": {"min": round(tf["TDS_calc"]["min"], 1), "max": round(tf["TDS_calc"]["max"], 1), "unit": "mg/L"},
        "NO3-N": {"min": round(tf["NO3-N_num"]["min"], 2), "max": round(tf["NO3-N_num"]["max"], 2), "unit": "mg/L"},
        "Well Depth": {"min": round(tf["WELL_DEPTH"]["min"], 1), "max": round(tf["WELL_DEPTH"]["max"], 1), "unit": "m"}
    }
else:
    VALIDATED_RANGES = None

# Separate Environmental Reference Ranges (Not to be confused with model domain)
ENVIRONMENTAL_RANGES = {
    "pH": {"range": "6.5 – 8.5", "unit": "-", "standard": "WHO / BIS Drinking Water Standard"},
    "TDS": {"range": "100 – 1000", "unit": "mg/L", "standard": "WHO / BIS Desirable-Permissible Limit"},
    "NO3-N": {"range": "< 10.0", "unit": "mg/L", "standard": "WHO / BIS Maximum Permissible Limit"},
    "Well Depth": {"range": "10 – 150", "unit": "m", "standard": "Regional Aquifer Context"}
}

# 4. Basic Input Validity Limits (Layer A Validation)
PHYSICAL_LIMITS = {
    "pH": {"min": 0.0, "max": 14.0},
    "TDS": {"min": 0.0, "max": 10000.0},
    "NO3-N": {"min": 0.0, "max": 500.0},
    "well_depth": {"min": 0.0, "max": 1000.0}
}

# 5. Standard Research Disclaimer Text
DISCLAIMER_TEXT = (
    "Research prototype — model-based preliminary groundwater screening. "
    "Results do not replace laboratory measurement and should not be interpreted as definitive regulatory compliance."
)
