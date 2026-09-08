"""
Input Validation — 2-Layer Architecture
Layer A: Physical limits (reject impossible values)
Layer B: Model domain applicability (warn if outside training range)
"""
import numpy as np
from scipy.spatial.distance import mahalanobis
from config import PHYSICAL_LIMITS, TRAINING_DOMAIN, VALIDATED_RANGES


def validate_groundwater_sample(ph_val, tds_val, no3_val, depth_val):
    """
    Returns: (is_valid, domain_state, errors, ood_warnings, validated_dict)
    domain_state ∈ {"INVALID", "OUT_OF_DOMAIN", "IN_DOMAIN"}
    """
    errors       = []
    ood_warnings = []

    # ── Parse inputs ──────────────────────────────────────────────────────────
    try:
        if any(v is None for v in [ph_val, tds_val, no3_val, depth_val]):
            return False, "INVALID", ["All four parameters are required."], [], None
        ph    = float(ph_val)
        tds   = float(tds_val)
        no3   = float(no3_val)
        depth = float(depth_val)
    except (ValueError, TypeError):
        return False, "INVALID", ["All parameters must be valid numeric values."], [], None

    # ── LAYER A: Physical validity ─────────────────────────────────────────────
    lim = PHYSICAL_LIMITS

    # pH: strict bounds [0, 14], zero excluded
    if ph <= lim["pH"]["min"] or ph > lim["pH"]["max"]:
        errors.append(
            f"Invalid pH ({ph:.2f}). Must be > {lim['pH']['min']} and ≤ {lim['pH']['max']}."
        )

    # TDS: separate messages for negative vs. too-high
    if tds < lim["TDS"]["min"]:
        errors.append(f"TDS cannot be negative ({tds:.1f} mg/L).")
    elif tds > lim["TDS"]["max"]:
        errors.append(f"TDS ({tds:.1f} mg/L) exceeds physical maximum ({lim['TDS']['max']:.0f} mg/L).")

    # NO3-N: separate messages
    if no3 < lim["NO3-N"]["min"]:
        errors.append(f"NO3-N cannot be negative ({no3:.2f} mg/L).")
    elif no3 > lim["NO3-N"]["max"]:
        errors.append(f"NO3-N ({no3:.2f} mg/L) exceeds physical maximum ({lim['NO3-N']['max']:.0f} mg/L).")

    # Well depth: must be > 0
    if depth <= lim["well_depth"]["min"]:
        errors.append(f"Well depth must be greater than zero ({depth:.1f} m).")
    elif depth > lim["well_depth"]["max"]:
        errors.append(f"Well depth ({depth:.1f} m) exceeds physical maximum ({lim['well_depth']['max']:.0f} m).")

    if errors:
        return False, "INVALID", errors, [], None

    validated_dict = {"pH": ph, "TDS": tds, "NO3-N": no3, "well_depth": depth}

    # ── LAYER B: Model domain applicability ───────────────────────────────────
    if VALIDATED_RANGES:
        checks = [
            ("pH",         ph,    "pH",        "{:.2f}", ""),
            ("TDS",        tds,   "TDS",       "{:.1f}", " mg/L"),
            ("NO3-N",      no3,   "NO3-N",     "{:.2f}", " mg/L"),
            ("Well Depth", depth, "Well Depth", "{:.1f}", " m"),
        ]
        for key, val, display, fmt, unit in checks:
            v_range = VALIDATED_RANGES[key]
            if val < v_range["min"] or val > v_range["max"]:
                ood_warnings.append(
                    f"{display} ({fmt.format(val)}{unit}) is outside the validated training "
                    f"range [{fmt.format(v_range['min'])}{unit} – {fmt.format(v_range['max'])}{unit}]."
                )

    # Optional multivariate Mahalanobis check
    mean_vec = TRAINING_DOMAIN.get("mean_vector")
    inv_cov  = TRAINING_DOMAIN.get("inv_covariance")
    if mean_vec and inv_cov:
        try:
            sample_vec = np.array([ph, tds, no3, depth])
            m_dist     = float(mahalanobis(sample_vec, np.array(mean_vec), np.array(inv_cov)))
            m_thresh   = TRAINING_DOMAIN.get("mahalanobis_threshold_95", 3.33)
            if m_dist > m_thresh:
                ood_warnings.append(
                    f"Multivariate Mahalanobis distance ({m_dist:.2f}) exceeds "
                    f"training 95th-percentile threshold ({m_thresh:.4f})."
                )
        except Exception:
            pass   # Mahalanobis check optional — silently skip on error

    domain_state = "OUT_OF_DOMAIN" if ood_warnings else "IN_DOMAIN"
    return True, domain_state, [], ood_warnings, validated_dict
