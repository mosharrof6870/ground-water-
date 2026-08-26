"""
UI Formatting and Visual Badge Helper.
Maps screening decision status codes to visually distinct status badges (Green, Yellow, Red).
NEVER uses the word 'SAFE'.
"""


def get_status_badge_style(status_code):
    """
    Returns visual badge styling dictionary for Streamlit UI.
    
    Status Codes:
        BELOW_THRESHOLD: Green (Below configured threshold & inside domain)
        UNCERTAIN: Yellow (Interval straddles threshold)
        CONFIDENCE_UNAVAILABLE: Yellow (Conformal interval unavailable)
        POTENTIAL_EXCEEDANCE: Red (Lower bound exceeds threshold)
        OUT_OF_DOMAIN: Red (Input outside validated model domain)
    """
    styles = {
        "BELOW_THRESHOLD": {
            "bg_color": "rgba(16, 185, 129, 0.15)",
            "text_color": "#34d399",
            "border_color": "rgba(16, 185, 129, 0.4)",
            "label": "🟢 BELOW CONFIGURED THRESHOLD",
            "theme": "success"
        },
        "UNCERTAIN": {
            "bg_color": "rgba(245, 158, 11, 0.15)",
            "text_color": "#fbbf24",
            "border_color": "rgba(245, 158, 11, 0.4)",
            "label": "🟡 UNCERTAIN — LAB CONFIRMATION RECOMMENDED",
            "theme": "warning"
        },
        "CONFIDENCE_UNAVAILABLE": {
            "bg_color": "rgba(245, 158, 11, 0.15)",
            "text_color": "#fbbf24",
            "border_color": "rgba(245, 158, 11, 0.4)",
            "label": "🟡 CONFIDENCE UNAVAILABLE — LAB CONFIRMATION RECOMMENDED",
            "theme": "warning"
        },
        "POTENTIAL_EXCEEDANCE": {
            "bg_color": "rgba(239, 68, 68, 0.15)",
            "text_color": "#f87171",
            "border_color": "rgba(239, 68, 68, 0.4)",
            "label": "🔴 POTENTIAL EXCEEDANCE — LAB CONFIRMATION REQUIRED",
            "theme": "error"
        },
        "OUT_OF_DOMAIN": {
            "bg_color": "rgba(239, 68, 68, 0.15)",
            "text_color": "#f87171",
            "border_color": "rgba(239, 68, 68, 0.4)",
            "label": "🔴 OUT-OF-DOMAIN — LAB CONFIRMATION REQUIRED",
            "theme": "error"
        }
    }

    return styles.get(status_code, {
        "bg_color": "rgba(148, 163, 184, 0.15)",
        "text_color": "#cbd5e1",
        "border_color": "rgba(148, 163, 184, 0.4)",
        "label": "⚪ UNKNOWN STATUS",
        "theme": "info"
    })


def format_threshold_distance(predicted_val, threshold_val, unit="µg/L"):
    """Formats distance from regulatory threshold cleanly."""
    diff = predicted_val - threshold_val
    if diff < 0:
        return f"{abs(diff):.2f} {unit} below threshold"
    elif diff > 0:
        return f"{diff:.2f} {unit} above threshold"
    else:
        return f"Exactly at threshold ({threshold_val} {unit})"
