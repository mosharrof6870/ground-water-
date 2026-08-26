"""
Regulatory Decision Engine.
Evaluates model point prediction + uncertainty bounds against research threshold configuration.
Incorporates 3-State Domain State (IN_DOMAIN vs OUT_OF_DOMAIN) into screening recommendations.
Enforces strict rule: NEVER use the word "SAFE".
"""
from config import THRESHOLDS, MODEL_CONFIG


class DecisionEngine:
    @staticmethod
    def evaluate_metal_screening(metal_key, point_prediction, uncertainty_result, domain_state="IN_DOMAIN"):
        """
        Evaluates screening decision for a single heavy metal target.
        """
        metal_symbol = MODEL_CONFIG[metal_key]["symbol"]
        threshold = THRESHOLDS[metal_symbol]
        confidence_label = MODEL_CONFIG[metal_key].get("confidence_level", "SCREENING CONFIDENCE")
        distance = point_prediction - threshold

        # RULE 1: IF MODEL DOMAIN IS OUTSIDE (OUT_OF_DOMAIN)
        if domain_state == "OUT_OF_DOMAIN":
            return {
                "status_code": "OUT_OF_DOMAIN",
                "title": "OUT-OF-DOMAIN — LABORATORY CONFIRMATION REQUIRED",
                "description": (
                    "This input is outside the range represented in the training data. "
                    "The model prediction is extrapolative and should not be used as a stand-alone screening decision. "
                    "Laboratory confirmation is recommended."
                ),
                "recommended_action": "Laboratory confirmation is required due to out-of-domain input.",
                "threshold": threshold,
                "distance_from_threshold": round(distance, 4),
                "confidence_label": "OUTSIDE VALIDATED MODEL DOMAIN"
            }

        # RULE 2: IF CONFORMAL INTERVAL IS UNAVAILABLE
        if not uncertainty_result or not uncertainty_result.get("available", False):
            return {
                "status_code": "CONFIDENCE_UNAVAILABLE",
                "title": "SCREENING CONFIDENCE UNAVAILABLE — LABORATORY CONFIRMATION RECOMMENDED",
                "description": (
                    f"90% conformal interval unavailable for {metal_symbol}. "
                    f"Point prediction is {point_prediction:.2f} µg/L (Threshold: {threshold} µg/L). "
                    "Laboratory confirmation is recommended."
                ),
                "recommended_action": "Laboratory confirmation recommended.",
                "threshold": threshold,
                "distance_from_threshold": round(distance, 4),
                "confidence_label": "CONFORMAL INTERVAL UNAVAILABLE"
            }

        lower_b = uncertainty_result["lower_bound"]
        upper_b = uncertainty_result["upper_bound"]

        # RULE 3: IN-DOMAIN DECISION LOGIC BASED ON CONFORMAL INTERVAL OVERLAP
        # CASE 1: Upper interval strictly below threshold
        if upper_b < threshold:
            return {
                "status_code": "BELOW_THRESHOLD",
                "title": "PRELIMINARY SCREENING: BELOW THRESHOLD",
                "description": (
                    f"The 90% conformal upper prediction bound ({upper_b:.2f} µg/L) is strictly below "
                    f"the configured screening threshold ({threshold} µg/L)."
                ),
                "recommended_action": "Preliminary model-based screening pass. Routine monitoring advised.",
                "threshold": threshold,
                "distance_from_threshold": round(distance, 4),
                "confidence_label": confidence_label
            }

        # CASE 3: Lower interval strictly above threshold
        elif lower_b > threshold:
            return {
                "status_code": "POTENTIAL_EXCEEDANCE",
                "title": "POTENTIAL EXCEEDANCE — LABORATORY CONFIRMATION REQUIRED",
                "description": (
                    f"The 90% conformal lower prediction bound ({lower_b:.2f} µg/L) exceeds "
                    f"the configured screening threshold ({threshold} µg/L). Potential contamination detected."
                ),
                "recommended_action": "High-priority laboratory confirmation is required.",
                "threshold": threshold,
                "distance_from_threshold": round(distance, 4),
                "confidence_label": confidence_label
            }

        # CASE 2: Interval straddles threshold (lower_b <= threshold <= upper_b)
        else:
            return {
                "status_code": "UNCERTAIN",
                "title": "UNCERTAIN — LABORATORY CONFIRMATION RECOMMENDED",
                "description": (
                    f"The 90% conformal prediction interval [{lower_b:.2f} – {upper_b:.2f} µg/L] "
                    f"straddles the screening threshold ({threshold} µg/L). Point prediction ({point_prediction:.2f} µg/L) "
                    "cannot rule out potential threshold exceedance."
                ),
                "recommended_action": "Laboratory confirmation is recommended.",
                "threshold": threshold,
                "distance_from_threshold": round(distance, 4),
                "confidence_label": confidence_label
            }
