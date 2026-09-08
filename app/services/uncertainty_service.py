"""
Uncertainty Service — OOF Conformal Prediction Intervals.
Priority order for interval calculation:
  1. precomputed lower_bound_90pct / upper_bound_90pct  (most accurate)
  2. residual-based: 90th-percentile of |true - predicted|
  3. interval_width median fallback (last resort)
"""
import numpy as np
from services.model_loader import ModelLoader
from config import MODEL_CONFIG


class UncertaintyService:

    @staticmethod
    def compute_interval(metal_key: str, predicted_val: float, alpha: float = 0.10) -> dict:
        """Returns 90% conformal prediction interval for a point prediction."""
        conf_df = ModelLoader.get_conformal(metal_key)
        cfg     = MODEL_CONFIG[metal_key]

        # Fallback when conformal data unavailable
        if conf_df is None or conf_df.empty:
            return UncertaintyService._unavailable(cfg["conformal_nominal_coverage"])

        try:
            # — Coverage string from stored data —
            if "is_covered" in conf_df.columns:
                obs_pct = float(conf_df["is_covered"].mean() * 100.0)
                obs_str = f"{obs_pct:.1f}% observed OOF coverage"
            else:
                obs_str = cfg.get("conformal_observed_coverage", "90.0%")

            # — Quantile (q_hat) selection —
            q_hat = None

            # Priority 1: precomputed bounds (most faithful to calibration)
            if "lower_bound_90pct" in conf_df.columns and "upper_bound_90pct" in conf_df.columns:
                widths = conf_df["upper_bound_90pct"] - conf_df["lower_bound_90pct"]
                q_hat  = float(np.quantile(widths, 1 - alpha)) / 2.0

            # Priority 2: raw residuals
            elif "true_value" in conf_df.columns and "predicted_value" in conf_df.columns:
                abs_resid = np.abs(conf_df["true_value"] - conf_df["predicted_value"])
                q_hat     = float(np.quantile(abs_resid, 1 - alpha))

            # Priority 3: median of interval_width (not mean — robust to outliers)
            elif "interval_width" in conf_df.columns:
                q_hat = float(np.median(conf_df["interval_width"])) / 2.0

            if q_hat is None:
                return UncertaintyService._unavailable(cfg["conformal_nominal_coverage"])

            lower = max(0.0, float(predicted_val - q_hat))
            upper = max(0.0, float(predicted_val + q_hat))

            return {
                "available":          True,
                "lower_bound":        round(lower, 4),
                "upper_bound":        round(upper, 4),
                "interval_width":     round(upper - lower, 4),
                "quantile_q":         round(q_hat, 4),
                "nominal_coverage":   cfg["conformal_nominal_coverage"],
                "observed_coverage":  obs_str,
                "message":            f"90% conformal interval ({obs_str})",
            }
        except Exception as e:
            return UncertaintyService._unavailable(
                cfg.get("conformal_nominal_coverage", "90%"),
                error=str(e)
            )

    @staticmethod
    def _unavailable(nominal: str = "90%", error: str = "") -> dict:
        msg = "Conformal interval unavailable — laboratory confirmation recommended."
        if error:
            msg += f" (Error: {error})"
        return {
            "available":         False,
            "lower_bound":       None,
            "upper_bound":       None,
            "interval_width":    None,
            "quantile_q":        None,
            "nominal_coverage":  nominal,
            "observed_coverage": "Unavailable",
            "message":           msg,
        }
