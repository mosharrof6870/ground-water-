"""
Prediction Service — orchestrates the full inference pipeline.
Each metal gets its own DataFrame built from its own feature_order.
"""
import numpy as np
import pandas as pd
from config import MODEL_CONFIG
from utils.validation import validate_groundwater_sample
from services.model_loader import ModelLoader
from services.uncertainty_service import UncertaintyService
from services.decision_engine import DecisionEngine


class PredictionService:

    @staticmethod
    def predict(ph: float, tds: float, no3: float, depth: float) -> dict:
        """
        Full inference pipeline for one groundwater sample.

        Returns dict with keys:
          success, domain_state, errors, ood_warnings,
          input_parameters, results, overall_recommendation
        """
        # ── Layer A + B validation ─────────────────────────────────────────────
        is_valid, domain_state, errors, ood_warnings, validated = \
            validate_groundwater_sample(ph, tds, no3, depth)

        if not is_valid:
            return {
                "success":              False,
                "domain_state":         "INVALID",
                "errors":               errors,
                "ood_warnings":         [],
                "input_parameters":     None,
                "results":              None,
                "overall_recommendation": "Prediction cancelled — invalid input parameters.",
            }

        # Lookup table: validated_dict key → model feature name
        FEAT_LOOKUP = {
            "pH":        "pH_proxy",
            "TDS":       "TDS_calc",
            "NO3-N":     "NO3-N_num",
            "well_depth":"WELL_DEPTH",
        }

        results       = {}
        status_codes  = []

        for metal_key in ["ni", "cd"]:
            cfg          = MODEL_CONFIG[metal_key]
            feature_order= cfg["feature_order"]   # ← each model's own order

            # Build DataFrame matching THIS model's feature contract
            feat_values = [validated[{v: k for k, v in FEAT_LOOKUP.items()}[f]] for f in feature_order]
            input_df    = pd.DataFrame([feat_values], columns=feature_order)

            # Load model
            try:
                model = ModelLoader.get_model(metal_key)
            except FileNotFoundError as e:
                return {
                    "success": False, "domain_state": domain_state,
                    "errors": [str(e)], "ood_warnings": ood_warnings,
                    "input_parameters": validated, "results": None,
                    "overall_recommendation": f"Model not available for {cfg['name']}.",
                }

            # Inference
            try:
                raw_pred = model.predict(input_df.values)[0]
                pred_val = float(np.maximum(0.0, raw_pred))   # non-negative physical constraint
            except Exception as e:
                return {
                    "success": False, "domain_state": domain_state,
                    "errors": [f"{cfg['name']} inference error: {e}"],
                    "ood_warnings": ood_warnings,
                    "input_parameters": validated, "results": None,
                    "overall_recommendation": f"Inference failed for {cfg['name']}.",
                }

            # Uncertainty quantification
            uncertainty = UncertaintyService.compute_interval(metal_key, pred_val)

            # Decision
            decision = DecisionEngine.evaluate(metal_key, pred_val, uncertainty, domain_state)
            status_codes.append(decision["status_code"])

            results[metal_key] = {
                "config":     cfg,
                "prediction": round(pred_val, 4),
                "uncertainty": uncertainty,
                "decision":    decision,
            }

        # ── Overall recommendation ─────────────────────────────────────────────
        if domain_state == "OUT_OF_DOMAIN" or "OUT_OF_DOMAIN" in status_codes:
            overall = ("At least one input is outside the validated model domain. "
                       "The prediction should not be used as a standalone screening decision. "
                       "Laboratory confirmation is required.")
        elif "POTENTIAL_EXCEEDANCE" in status_codes:
            overall = ("Potential threshold exceedance detected for at least one heavy metal. "
                       "High-priority laboratory confirmation is required.")
        elif any(s in status_codes for s in ["UNCERTAIN", "CONFIDENCE_UNAVAILABLE"]):
            overall = ("At least one target has uncertainty overlapping the screening threshold. "
                       "Laboratory confirmation is recommended.")
        else:
            overall = ("Both Ni and Cd prediction intervals remain below their configured screening thresholds. "
                       "The sample passes preliminary model-based screening. "
                       "Laboratory confirmation remains appropriate where definitive regulatory assessment is required.")

        return {
            "success":                True,
            "domain_state":           domain_state,
            "errors":                 [],
            "ood_warnings":           ood_warnings,
            "input_parameters":       validated,
            "results":                results,
            "overall_recommendation": overall,
        }
