"""
Decision Engine — evaluates point prediction + conformal interval vs threshold.
State machine: BELOW_THRESHOLD | UNCERTAIN | POTENTIAL_EXCEEDANCE | CONFIDENCE_UNAVAILABLE | OUT_OF_DOMAIN
RULE: NEVER use the word 'safe'.
"""
from config import THRESHOLDS, MODEL_CONFIG


class DecisionEngine:

    @staticmethod
    def evaluate(metal_key: str, point_pred: float, uncertainty: dict, domain_state: str = "IN_DOMAIN") -> dict:
        symbol     = MODEL_CONFIG[metal_key]["symbol"]
        threshold  = THRESHOLDS[symbol]
        conf_label = MODEL_CONFIG[metal_key].get("confidence_level", "SCREENING CONFIDENCE")
        distance   = point_pred - threshold

        base = {"threshold": threshold, "distance_from_threshold": round(distance, 4)}

        # CASE 0: Out-of-domain
        if domain_state == "OUT_OF_DOMAIN":
            return {**base,
                "status_code":       "OUT_OF_DOMAIN",
                "title":             "OUT-OF-DOMAIN — LABORATORY CONFIRMATION REQUIRED",
                "description":       ("Input is outside the validated training range. "
                                      "Model prediction is extrapolative and must not be used as standalone screening."),
                "recommended_action":"Laboratory confirmation required (out-of-domain input).",
                "confidence_label":  "OUTSIDE VALIDATED MODEL DOMAIN",
            }

        # CASE 1: Conformal interval unavailable
        if not uncertainty or not uncertainty.get("available", False):
            return {**base,
                "status_code":       "CONFIDENCE_UNAVAILABLE",
                "title":             "SCREENING CONFIDENCE UNAVAILABLE — LABORATORY CONFIRMATION RECOMMENDED",
                "description":       (f"90% conformal interval unavailable for {symbol}. "
                                      f"Point prediction: {point_pred:.2f} µg/L (threshold: {threshold} µg/L)."),
                "recommended_action":"Laboratory confirmation recommended.",
                "confidence_label":  "CONFORMAL INTERVAL UNAVAILABLE",
            }

        lower = uncertainty["lower_bound"]
        upper = uncertainty["upper_bound"]

        # CASE 2: Entire interval BELOW threshold
        if upper < threshold:
            return {**base,
                "status_code":       "BELOW_THRESHOLD",
                "title":             "PRELIMINARY SCREENING: BELOW THRESHOLD",
                "description":       (f"90% conformal upper bound ({upper:.2f} µg/L) is strictly below "
                                      f"the screening threshold ({threshold} µg/L)."),
                "recommended_action":"Preliminary screening pass. Routine monitoring advised.",
                "confidence_label":  conf_label,
            }

        # CASE 3: Entire interval ABOVE threshold
        if lower > threshold:
            return {**base,
                "status_code":       "POTENTIAL_EXCEEDANCE",
                "title":             "POTENTIAL EXCEEDANCE — LABORATORY CONFIRMATION REQUIRED",
                "description":       (f"90% conformal lower bound ({lower:.2f} µg/L) exceeds "
                                      f"screening threshold ({threshold} µg/L). Potential contamination detected."),
                "recommended_action":"High-priority laboratory confirmation required.",
                "confidence_label":  conf_label,
            }

        # CASE 4: Interval STRADDLES threshold
        return {**base,
            "status_code":       "UNCERTAIN",
            "title":             "UNCERTAIN — LABORATORY CONFIRMATION RECOMMENDED",
            "description":       (f"90% conformal interval [{lower:.2f} – {upper:.2f} µg/L] straddles "
                                  f"the screening threshold ({threshold} µg/L). "
                                  f"Point prediction ({point_pred:.2f} µg/L) cannot rule out exceedance."),
            "recommended_action":"Laboratory confirmation is recommended.",
            "confidence_label":  conf_label,
        }
