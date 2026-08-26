"""
Model Applicability Panel Component.
Renders Range-Based Model Applicability Check table comparing user sample against validated training dataset bounds.
"""
import textwrap
import streamlit as st
from config import VALIDATED_RANGES, ENVIRONMENTAL_RANGES


def render_applicability_panel(input_params):
    """
    Renders Model Applicability check section for submitted sample parameters.
    """
    st.markdown("### 🎯 MODEL APPLICABILITY")
    st.caption("Range-based model applicability check comparing user sample parameters against empirical training data bounds.")

    if not VALIDATED_RANGES:
        st.warning("Validated model input ranges are unavailable.")
        return

    param_map = [
        ("pH", "pH", input_params["pH"], "-"),
        ("TDS", "TDS", input_params["TDS"], "mg/L"),
        ("NO3-N", "NO3-N", input_params["NO3-N"], "mg/L"),
        ("Well Depth", "well_depth", input_params["well_depth"], "m")
    ]

    cols = st.columns(4)

    for idx, (display_name, key_name, user_val, unit) in enumerate(param_map):
        v_bounds = VALIDATED_RANGES[display_name]
        v_min, v_max = v_bounds["min"], v_bounds["max"]
        in_range = v_min <= user_val <= v_max
        
        status_label = "✓ Within validated range" if in_range else "⚠️ Outside validated range"
        status_color = "#34d399" if in_range else "#f87171"
        bg_color = "rgba(16, 185, 129, 0.1)" if in_range else "rgba(239, 68, 68, 0.1)"
        border_color = "rgba(16, 185, 129, 0.3)" if in_range else "rgba(239, 68, 68, 0.3)"

        unit_str = f" {unit}" if unit != "-" else ""
        formatted_val = f"{user_val:.2f}" if display_name != "TDS" and display_name != "Well Depth" else f"{user_val:.1f}"

        card_html = textwrap.dedent(f"""
        <div style="background: {bg_color}; border: 1px solid {border_color}; border-radius: 10px; padding: 14px; text-align: center;">
            <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">{display_name}</div>
            <div style="font-size: 1.4rem; font-weight: 800; color: #f8fafc; margin-top: 4px;">{formatted_val}<span style="font-size: 0.85rem; color: #cbd5e1;">{unit_str}</span></div>
            <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">
                Validated Range:<br><strong style="color: #cbd5e1;">{v_min}{unit_str} – {v_max}{unit_str}</strong>
            </div>
            <div style="font-size: 0.78rem; font-weight: 700; color: {status_color}; margin-top: 8px;">
                {status_label}
            </div>
        </div>
        """)
        
        with cols[idx]:
            st.markdown(card_html, unsafe_allow_html=True)
