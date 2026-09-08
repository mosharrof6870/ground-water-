"""Formatting helpers. NEVER uses the word 'SAFE'."""


def get_status_badge_style(status_code: str) -> dict:
    styles = {
        "BELOW_THRESHOLD":     {"bg": "rgba(16,185,129,0.15)",  "text": "#34d399", "border": "rgba(16,185,129,0.4)",  "label": "🟢 BELOW CONFIGURED THRESHOLD"},
        "UNCERTAIN":           {"bg": "rgba(245,158,11,0.15)",  "text": "#fbbf24", "border": "rgba(245,158,11,0.4)",  "label": "🟡 UNCERTAIN — LAB CONFIRMATION RECOMMENDED"},
        "CONFIDENCE_UNAVAILABLE":{"bg":"rgba(245,158,11,0.15)", "text": "#fbbf24", "border": "rgba(245,158,11,0.4)",  "label": "🟡 CONFIDENCE UNAVAILABLE — LAB RECOMMENDED"},
        "POTENTIAL_EXCEEDANCE":{"bg": "rgba(239,68,68,0.15)",   "text": "#f87171", "border": "rgba(239,68,68,0.4)",   "label": "🔴 POTENTIAL EXCEEDANCE — LAB CONFIRMATION REQUIRED"},
        "OUT_OF_DOMAIN":       {"bg": "rgba(239,68,68,0.15)",   "text": "#f87171", "border": "rgba(239,68,68,0.4)",   "label": "🔴 OUT-OF-DOMAIN — LAB CONFIRMATION REQUIRED"},
    }
    return styles.get(status_code, {
        "bg": "rgba(148,163,184,0.15)", "text": "#cbd5e1",
        "border": "rgba(148,163,184,0.4)", "label": "⚪ UNKNOWN STATUS",
    })


def format_threshold_distance(predicted: float, threshold: float, unit: str = "µg/L") -> str:
    diff = predicted - threshold
    if diff < 0:
        return f"{abs(diff):.2f} {unit} below threshold"
    elif diff > 0:
        return f"{diff:.2f} {unit} above threshold"
    return f"Exactly at threshold ({threshold} {unit})"
