"""
Uncertainty Quantification Service.
Calculates Out-of-Fold Residual Quantile Conformal prediction intervals based on saved calibration artifacts.
Does NOT fabricate arbitrary percentage intervals if calibration artifacts are missing.
Explicitly distinguishes nominal coverage from empirical OOF coverage.
"""
import numpy as np
from services.model_loader import ModelLoader


class UncertaintyService:
    @staticmethod
    def compute_conformal_interval(metal_key, predicted_val, alpha=0.10):
        """
        Computes Out-of-Fold Residual Quantile 90% Conformal Prediction Interval for a given predicted concentration.
        
        Args:
            metal_key (str): 'ni' or 'cd'
            predicted_val (float): Point prediction from trained model
            alpha (float): Miscoverage rate (default 0.10 for 90% coverage)
            
        Returns:
            dict: {
                "available": bool,
                "lower_bound": float or None,
                "upper_bound": float or None,
                "interval_width": float or None,
                "quantile_q": float or None,
                "nominal_coverage": str,
                "observed_coverage": str,
                "message": str
            }
        """
        conf_df = ModelLoader.get_conformal_data(metal_key)

        if conf_df is None or conf_df.empty:
            return {
                "available": False,
                "lower_bound": None,
                "upper_bound": None,
                "interval_width": None,
                "quantile_q": None,
                "nominal_coverage": "90% nominal",
                "observed_coverage": "Unavailable",
                "message": "Prediction interval unavailable in deployment. Laboratory confirmation is recommended when uncertainty cannot be quantified."
            }

        try:
            # Calculate observed OOF empirical coverage if 'is_covered' column exists
            if "is_covered" in conf_df.columns:
                obs_cov_pct = float(conf_df["is_covered"].mean() * 100.0)
                obs_cov_str = f"{obs_cov_pct:.1f}% observed OOF coverage"
            else:
                obs_cov_str = "Empirical coverage unrecorded"

            # Check if precomputed residual / interval width exists or compute q from absolute residuals
            if "true_value" in conf_df.columns and "predicted_value" in conf_df.columns:
                abs_residuals = np.abs(conf_df["true_value"] - conf_df["predicted_value"])
                # 90th percentile quantile for 90% coverage
                q_90 = float(np.percentile(abs_residuals, (1 - alpha) * 100))
            elif "interval_width" in conf_df.columns:
                # Use half of average interval width
                q_90 = float(np.mean(conf_df["interval_width"])) / 2.0
            else:
                return {
                    "available": False,
                    "lower_bound": None,
                    "upper_bound": None,
                    "interval_width": None,
                    "quantile_q": None,
                    "nominal_coverage": "90% nominal",
                    "observed_coverage": obs_cov_str,
                    "message": "Prediction interval unavailable in deployment."
                }

            lower_b = max(0.0, float(predicted_val - q_90))
            upper_b = max(0.0, float(predicted_val + q_90))
            width = float(upper_b - lower_b)

            return {
                "available": True,
                "lower_bound": round(lower_b, 4),
                "upper_bound": round(upper_b, 4),
                "interval_width": round(width, 4),
                "quantile_q": round(q_90, 4),
                "nominal_coverage": "90% nominal conformal interval",
                "observed_coverage": obs_cov_str,
                "message": f"90% nominal conformal interval ({obs_cov_str})"
            }
        except Exception as e:
            return {
                "available": False,
                "lower_bound": None,
                "upper_bound": None,
                "interval_width": None,
                "quantile_q": None,
                "nominal_coverage": "90% nominal",
                "observed_coverage": "Error",
                "message": f"Conformal calculation error: {str(e)}"
            }
