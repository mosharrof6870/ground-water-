# CHANGELOG — Groundwater AI Screening Project

## [v1.0.0-final] - 2026-08-23

### Forensic Audit & Reproducibility Freeze
- **Scientific Audit Completed**: Verified zero retraining/data manipulation, confirmed preservation of pre-trained binary models `FINAL_MODEL_NI.joblib` and `FINAL_MODEL_CD.joblib`.
- **Conformal Prediction Terminology Hardening**: Replaced mislabeled "CV+" references with exact, defensible method descriptor: **"Out-of-Fold Residual Quantile Conformal Prediction"**.
- **Metrics Differentiation**: Formally separated primary 5×5 nested cross-validation out-of-fold metrics (Ni $R^2 = 0.2240$, Cd $R^2 = 0.7061$) from single-pass out-of-fold metrics (Ni $R^2 = 0.2271$, Cd $R^2 = 0.7158$), reporting actual non-zero differences (+0.0031 for Ni, +0.0097 for Cd).
- **Training $N$ Sample Trace Documented**: Explicitly documented that model estimators were fit on $N=40$ samples using `SimpleImputer(strategy='median')` (imputing sample `S98_01798`), while complete-case domain profiling evaluated $N=39$.
- **Project Structure Reorganization**: Created clean target directories (`data/`, `scripts/`, `models/`, `results/`, `reports/`, `docs/`, `archive/`) and cataloged all 418 files into `PROJECT_FILE_INVENTORY.csv` and `PROJECT_MANIFEST.csv`.
- **Safe Archiving**: Safely archived 120 intermediate stage outputs, debug logs, and obsolete archives into `archive/` without deleting active research or application files.
- **Validation**: Executed 12/12 automated test suite cases verifying Streamlit app inference, domain bounds enforcement, and decision engine logic.
