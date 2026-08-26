"""
Input Validation and Domain Boundary Checker.
Implements a strict 2-Layer Validation Architecture:
  Layer A. Basic Input Validity (Reject physically invalid values: pH <= 0, TDS < 0, NO3-N < 0, Well Depth <= 0).
  Layer B. Range-Based Model Applicability Check (Compare against validated training data ranges).
"""
import numpy as np
from scipy.spatial.distance import mahalanobis
from config import PHYSICAL_LIMITS, TRAINING_DOMAIN, VALIDATED_RANGES


def validate_groundwater_sample(ph_val, tds_val, no3_val, depth_val):
    """
    Validates input sample against basic physical limits and model training applicability ranges.
    
    Returns:
        tuple: (
            is_valid: bool,
            domain_state: str ("INVALID" | "OUT_OF_DOMAIN" | "IN_DOMAIN"),
            errors: list,
            ood_warnings: list,
            validated_dict: dict
        )
    """
    errors = []
    ood_warnings = []

    # 1. Parse numeric inputs without modifying user input
    try:
        if ph_val is None or tds_val is None or no3_val is None or depth_val is None:
            return False, "INVALID", ["Missing required input parameter."], [], None
        ph = float(ph_val)
        tds = float(tds_val)
        no3 = float(no3_val)
        depth = float(depth_val)
    except (ValueError, TypeError):
        return False, "INVALID", ["All parameters (pH, TDS, NO3-N, Well Depth) must be valid numeric values."], [], None

    # --------------------------------------------------------------------------
    # LAYER A: BASIC INPUT VALIDITY CHECKS (Reject invalid values)
    # --------------------------------------------------------------------------
    if ph <= PHYSICAL_LIMITS["pH"]["min"] or ph > PHYSICAL_LIMITS["pH"]["max"]:
        errors.append(f"Invalid pH value: {ph:.2f}. Must be greater than 0 and within [0–14].")

    if tds < PHYSICAL_LIMITS["TDS"]["min"] or tds > PHYSICAL_LIMITS["TDS"]["max"]:
        errors.append(f"TDS cannot be negative: {tds:.1f} mg/L.")

    if no3 < PHYSICAL_LIMITS["NO3-N"]["min"] or no3 > PHYSICAL_LIMITS["NO3-N"]["max"]:
        errors.append(f"NO3-N cannot be negative: {no3:.2f} mg/L.")

    if depth <= PHYSICAL_LIMITS["well_depth"]["min"] or depth > PHYSICAL_LIMITS["well_depth"]["max"]:
        errors.append(f"Well depth must be greater than zero: {depth:.1f} m.")

    if len(errors) > 0:
        return False, "INVALID", errors, [], None

    validated_dict = {
        "pH": ph,
        "TDS": tds,
        "NO3-N": no3,
        "well_depth": depth
    }

    # --------------------------------------------------------------------------
    # LAYER B: MODEL DOMAIN / RANGE-BASED APPLICABILITY CHECK
    # --------------------------------------------------------------------------
    if VALIDATED_RANGES:
        ph_v = VALIDATED_RANGES["pH"]
        if ph < ph_v["min"] or ph > ph_v["max"]:
            ood_warnings.append(
                f"pH ({ph:.2f}) is outside the validated model training range [{ph_v['min']:.2f} – {ph_v['max']:.2f}]."
            )

        tds_v = VALIDATED_RANGES["TDS"]
        if tds < tds_v["min"] or tds > tds_v["max"]:
            ood_warnings.append(
                f"TDS ({tds:.1f} mg/L) is outside the validated model training range [{tds_v['min']:.1f} – {tds_v['max']:.1f} mg/L]."
            )

        no3_v = VALIDATED_RANGES["NO3-N"]
        if no3 < no3_v["min"] or no3 > no3_v["max"]:
            ood_warnings.append(
                f"NO3-N ({no3:.2f} mg/L) is outside the validated model training range [{no3_v['min']:.2f} – {no3_v['max']:.2f} mg/L]."
            )

        depth_v = VALIDATED_RANGES["Well Depth"]
        if depth < depth_v["min"] or depth > depth_v["max"]:
            ood_warnings.append(
                f"Well Depth ({depth:.1f} m) is outside the validated model training range [{depth_v['min']:.1f} – {depth_v['max']:.1f} m]."
            )

    # Optional Multivariate Check if covariance matrix is available
    if TRAINING_DOMAIN.get("mean_vector") and TRAINING_DOMAIN.get("inv_covariance"):
        try:
            mean_vec = np.array(TRAINING_DOMAIN["mean_vector"])
            inv_cov = np.array(TRAINING_DOMAIN["inv_covariance"])
            sample_vec = np.array([ph, tds, no3, depth])
            
            m_dist = float(mahalanobis(sample_vec, mean_vec, inv_cov))
            m_thresh = TRAINING_DOMAIN.get("mahalanobis_threshold_95", 3.33)

            if m_dist > m_thresh:
                ood_warnings.append(
                    f"Multivariate Mahalanobis distance ({m_dist:.2f}) exceeds training dataset 95th percentile threshold ({m_thresh:.2f})."
                )
        except Exception:
            pass

    domain_state = "OUT_OF_DOMAIN" if len(ood_warnings) > 0 else "IN_DOMAIN"

    return True, domain_state, [], ood_warnings, validated_dict
