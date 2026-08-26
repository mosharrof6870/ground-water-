"""
Input Form Component for Streamlit Dashboard.
Provides user-friendly form fields, quick preset buttons, and validation guidance.
"""
import streamlit as st
from config import TRAINING_DOMAIN, PHYSICAL_LIMITS


def render_input_form():
    """
    Renders input form for groundwater screening parameters.
    
    Returns:
        tuple: (submitted: bool, ph: float, tds: float, no3: float, depth: float)
    """
    st.markdown("### 🧪 Water Sample Parameters")
    st.caption("Enter field parameters. Values outside empirical training range will trigger an out-of-domain warning.")

    # Preset Quick Select Buttons
    col_p1, col_p2, col_p3 = st.columns(3)
    preset_ph, preset_tds, preset_no3, preset_depth = 7.10, 220.0, 1.20, 25.0

    if col_p1.button("Standard Sample"):
        preset_ph, preset_tds, preset_no3, preset_depth = 7.10, 220.0, 1.20, 25.0
    if col_p2.button("High TDS (OOD)"):
        preset_ph, preset_tds, preset_no3, preset_depth = 7.25, 750.0, 2.50, 30.0
    if col_p3.button("High NO3-N (OOD)"):
        preset_ph, preset_tds, preset_no3, preset_depth = 6.95, 300.0, 15.00, 20.0

    tf = TRAINING_DOMAIN["features"]

    with st.form(key="groundwater_input_form"):
        ph_input = st.number_input(
            "pH",
            min_value=float(PHYSICAL_LIMITS["pH"]["min"]),
            max_value=float(PHYSICAL_LIMITS["pH"]["max"]),
            value=float(preset_ph),
            step=0.05,
            help=f"Groundwater pH (Training Range: {tf['pH_proxy']['min']:.2f} – {tf['pH_proxy']['max']:.2f})"
        )

        tds_input = st.number_input(
            "TDS (Total Dissolved Solids) [mg/L]",
            min_value=float(PHYSICAL_LIMITS["TDS"]["min"]),
            max_value=float(PHYSICAL_LIMITS["TDS"]["max"]),
            value=float(preset_tds),
            step=10.0,
            help=f"Total Dissolved Solids in mg/L (Training Range: {tf['TDS_calc']['min']:.1f} – {tf['TDS_calc']['max']:.1f} mg/L)"
        )

        no3_input = st.number_input(
            "NO3-N (Nitrate-Nitrogen) [mg/L]",
            min_value=float(PHYSICAL_LIMITS["NO3-N"]["min"]),
            max_value=float(PHYSICAL_LIMITS["NO3-N"]["max"]),
            value=float(preset_no3),
            step=0.5,
            help=f"Nitrate-Nitrogen in mg/L (Training Range: {tf['NO3-N_num']['min']:.2f} – {tf['NO3-N_num']['max']:.2f} mg/L)"
        )

        depth_input = st.number_input(
            "Well Depth [m]",
            min_value=float(PHYSICAL_LIMITS["well_depth"]["min"]),
            max_value=float(PHYSICAL_LIMITS["well_depth"]["max"]),
            value=float(preset_depth),
            step=1.0,
            help=f"Tubewell depth in meters (Training Range: {tf['WELL_DEPTH']['min']:.1f} – {tf['WELL_DEPTH']['max']:.1f} m)"
        )

        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "🔍 ANALYZE SAMPLE",
            type="primary"
        )

    return submitted, ph_input, tds_input, no3_input, depth_input
