"""
Main Prediction Service.
Orchestrates Input Validation -> Model Inference -> Uncertainty Service -> Decision Engine.
Implements strict 3-State Domain Handling (INVALID, OUT_OF_DOMAIN, IN_DOMAIN).
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
    def predict_sample(ph_val, tds_val, no3_val, depth_val):
        """
        Executes end-to-end heavy metal screening for a groundwater sample.
        
        Returns:
            dict: {
                "success": bool,
                "domain_state": str ("INVALID" | "OUT_OF_DOMAIN" | "IN_DOMAIN"),
                "errors": list,
                "ood_warnings": list,
                "input_parameters": dict,
                "results": dict,
                "overall_recommendation": str
            }
        """
        # 1. 2-Layer Input Validation & Domain Applicability Check
        is_valid, domain_state, errors, ood_warnings, validated_inputs = validate_groundwater_sample(
            ph_val, tds_val, no3_val, depth_val
        )
        
        # STATE 1: 🔴 INVALID -> PREDICTION CANCELLED IMMEDIATELY
        if not is_valid:
            return {
                "success": False,
                "domain_state": "INVALID",
                "errors": errors,
                "ood_warnings": [],
                "input_parameters": None,
                "results": None,
                "overall_recommendation": "Prediction cancelled. Input parameters fail basic physical validity checks."
            }

        # Build feature matrix matching exact training feature contract:
        # ['pH_proxy', 'TDS_calc', 'NO3-N_num', 'WELL_DEPTH']
        feature_order = MODEL_CONFIG["ni"]["feature_order"]
        raw_features = {
            "pH_proxy": validated_inputs["pH"],
            "TDS_calc": validated_inputs["TDS"],
            "NO3-N_num": validated_inputs["NO3-N"],
            "WELL_DEPTH": validated_inputs["well_depth"]
        }
        
        input_df = pd.DataFrame([[raw_features[feat] for feat in feature_order]], columns=feature_order)

        results = {}
        status_codes = []

        for metal_key in ["ni", "cd"]:
            cfg = MODEL_CONFIG[metal_key]
            
            # Load model artifact
            try:
                model_obj = ModelLoader.get_model(metal_key)
            except Exception as e:
                return {
                    "success": False,
                    "domain_state": domain_state,
                    "errors": [f"{cfg['name']} model error: {str(e)}"],
                    "ood_warnings": ood_warnings,
                    "input_parameters": validated_inputs,
                    "results": None,
                    "overall_recommendation": f"Model loading failed for {cfg['name']}."
                }

            # Inference (Inference-only deployment)
            try:
                pred_raw = model_obj.predict(input_df.values)[0]
                pred_val = float(np.maximum(0.0, pred_raw))  # Non-negative physical concentration
            except Exception as e:
                return {
                    "success": False,
                    "domain_state": domain_state,
                    "errors": [f"{cfg['name']} prediction failed: {str(e)}"],
                    "ood_warnings": ood_warnings,
                    "input_parameters": validated_inputs,
                    "results": None,
                    "overall_recommendation": f"Inference error on {cfg['name']} model."
                }

            # Uncertainty Quantification (Out-of-Fold Conformal Prediction Interval)
            uncertainty = UncertaintyService.compute_conformal_interval(metal_key, pred_val)

            # Decision Engine Evaluation
            decision = DecisionEngine.evaluate_metal_screening(
                metal_key, pred_val, uncertainty, domain_state=domain_state
            )

            status_codes.append(decision["status_code"])

            results[metal_key] = {
                "config": cfg,
                "prediction": round(pred_val, 4),
                "uncertainty": uncertainty,
                "decision": decision
            }

        # Dynamic Overall Recommendation Logic
        if domain_state == "OUT_OF_DOMAIN" or "OUT_OF_DOMAIN" in status_codes:
            overall_rec = (
                "At least one input lies outside the validated model domain. "
                "The prediction should not be used as a stand-alone screening decision. "
                "Laboratory confirmation is required."
            )
        elif "POTENTIAL_EXCEEDANCE" in status_codes:
            overall_rec = (
                "Potential threshold exceedance detected for at least one heavy metal target. "
                "High-priority laboratory confirmation is required."
            )
        elif "UNCERTAIN" in status_codes or "CONFIDENCE_UNAVAILABLE" in status_codes:
            overall_rec = (
                "At least one target has uncertainty overlapping the screening threshold. "
                "Laboratory confirmation is recommended."
            )
        else:
            overall_rec = (
                "Both Ni and Cd prediction intervals remain below their configured screening thresholds. "
                "The sample therefore passes preliminary model-based screening. "
                "Laboratory confirmation remains appropriate where definitive regulatory assessment is required."
            )

        return {
            "success": True,
            "domain_state": domain_state,
            "errors": [],
            "ood_warnings": ood_warnings,
            "input_parameters": validated_inputs,
            "results": results,
            "overall_recommendation": overall_rec
        }
