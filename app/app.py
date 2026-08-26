"""
GROUNDWATER AI HEAVY-METAL SCREENING WEB APPLICATION
Production-Style Responsive Research Prototype (Streamlit)
"""
import os
import sys
import textwrap

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import pandas as pd
import streamlit as st

from config import MODEL_CONFIG, THRESHOLDS, DISCLAIMER_TEXT
from services.model_loader import ModelLoader
from services.prediction_service import PredictionService
from components.input_form import render_input_form
from components.result_card import render_result_card
from components.threshold_chart import render_threshold_chart
from components.applicability_panel import render_applicability_panel
from components.interpretation_panel import render_interpretation_panel
from components.report_generator import render_report_download_button


# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Groundwater AI Heavy-Metal Screening",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Scientific UI styling & font refinement
st.markdown(
    textwrap.dedent("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;700;800&display=swap');

        .main-header {
            border-bottom: 2px solid rgba(59, 130, 246, 0.3);
            padding-bottom: 18px;
            margin-bottom: 24px;
        }
        .header-title {
            font-family: 'Outfit', 'Inter', sans-serif;
            font-size: 2.3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #2563eb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .header-subtitle {
            font-family: 'Inter', sans-serif;
            font-size: 1.05rem;
            color: #cbd5e1;
            margin-top: 6px;
            font-weight: 400;
        }
        .status-pill {
            display: inline-block;
            background: rgba(59, 130, 246, 0.15);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.35);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.82rem;
            font-weight: 600;
            margin-top: 10px;
        }
        .rec-container {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-left: 5px solid #3b82f6;
            border-radius: 12px;
            padding: 18px 22px;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .footer-disclaimer {
            margin-top: 40px;
            padding: 18px 22px;
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 12px;
            font-size: 0.85rem;
            color: #cbd5e1;
            text-align: center;
            line-height: 1.5;
        }
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
        }
    </style>
    """),
    unsafe_allow_html=True
)

# Initialize Session History
if "history" not in st.session_state:
    st.session_state["history"] = []

# Pre-load models at app startup once
try:
    ModelLoader.load_all_models()
except Exception as e:
    st.error(f"Startup Model Loading Error: {str(e)}")


# ------------------------------------------------------------------------------
# APP HEADER
# ------------------------------------------------------------------------------
st.markdown(
    textwrap.dedent("""
    <div class="main-header">
        <div class="header-title">🧪 GROUNDWATER AI SCREENING</div>
        <div class="header-subtitle">Hydrochemistry-informed heavy-metal screening prototype (North Bengal Aquifer Framework)</div>
        <div class="status-pill">🔬 Research Prototype — Inference Only</div>
    </div>
    """),
    unsafe_allow_html=True
)

st.info(
    "Screening thresholds are based on the research configuration (Ni = 20 µg/L, Cd = 3 µg/L) "
    "and should be verified against the applicable current standard before regulatory use."
)


# ------------------------------------------------------------------------------
# LAYOUT: DESKTOP 2-COLUMN / MOBILE AUTO-STACK
# ------------------------------------------------------------------------------
left_col, right_col = st.columns([1, 1.4], gap="large")

# ------------------------------------------------------------------------------
# LEFT COLUMN: INPUT FORM & HISTORY
# ------------------------------------------------------------------------------
with left_col:
    submitted, ph_in, tds_in, no3_in, depth_in = render_input_form()

    # Trigger Analysis
    if submitted:
        with st.spinner("Analyzing groundwater sample..."):
            pred_response = PredictionService.predict_sample(ph_in, tds_in, no3_in, depth_in)
            st.session_state["last_prediction"] = pred_response

            if pred_response["success"]:
                st.session_state["history"].append({
                    "pH": ph_in,
                    "TDS": tds_in,
                    "NO3-N": no3_in,
                    "Depth": depth_in,
                    "Domain_State": pred_response["domain_state"],
                    "Ni_pred": pred_response["results"]["ni"]["prediction"],
                    "Cd_pred": pred_response["results"]["cd"]["prediction"],
                    "Status": pred_response["overall_recommendation"]
                })

    # Session History Expander
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📜 LOCAL SESSION HISTORY", expanded=False):
        if len(st.session_state["history"]) > 0:
            hist_df = pd.DataFrame(st.session_state["history"])
            st.dataframe(hist_df)
            if st.button("🗑️ CLEAR HISTORY"):
                st.session_state["history"] = []
                st.rerun()
        else:
            st.caption("No sample analyses recorded in current browser session.")


# ------------------------------------------------------------------------------
# RIGHT COLUMN: RESULTS & INTERPRETATION
# ------------------------------------------------------------------------------
with right_col:
    if "last_prediction" in st.session_state:
        res_payload = st.session_state["last_prediction"]

        # STATE 1: 🔴 INVALID INPUTS -> REJECTED COMPLETELY
        if not res_payload["success"]:
            st.error("❌ Sample Input Rejected: Invalid Environmental Parameters")
            for err in res_payload["errors"]:
                st.markdown(f"• **{err}**")
            st.info("Please enter valid parameters to proceed.")

        else:
            st.markdown("### 📊 Screening Results")
            domain_state = res_payload["domain_state"]

            # STATE 2: 🟡 OUT_OF_DOMAIN WARNING BANNER
            if domain_state == "OUT_OF_DOMAIN":
                st.warning(
                    "⚠️ **OUTSIDE VALIDATED MODEL DOMAIN**\n\n"
                    "This input is outside the range represented in the training data. "
                    "The model prediction is extrapolative and should not be used as a stand-alone screening decision.\n\n"
                    "**Domain Anomaly Details:**\n" +
                    "\n".join([f"• {w}" for w in res_payload["ood_warnings"]]) +
                    "\n\n*Laboratory confirmation is recommended.*"
                )

            # STATE 3: 🟢 IN_DOMAIN SUCCESS BANNER
            else:
                st.success(
                    "🟢 **IN-DOMAIN SAMPLE**: All input parameters are within the empirical training dataset distribution."
                )

            # Render Model Applicability Panel
            render_applicability_panel(res_payload["input_parameters"])

            st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)

            results = res_payload["results"]

            # Render Ni Result Card & Chart
            render_result_card(results["ni"])
            render_threshold_chart(results["ni"])

            st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)

            # Render Cd Result Card & Chart
            render_result_card(results["cd"])
            render_threshold_chart(results["cd"])

            st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)

            # Render Scientific Interpretation & Model Validation Information
            render_interpretation_panel(results)

            # Render Dynamic Overall Recommendation (Rendered ONLY ONCE!)
            rec_html = textwrap.dedent(f"""
            <div class="rec-container">
                <div style="font-size: 0.95rem; font-weight: 700; color: #60a5fa; text-transform: uppercase; letter-spacing: 0.5px;">Recommended Screening Action</div>
                <div style="font-size: 1.05rem; font-weight: 600; color: #f8fafc; margin-top: 8px; line-height: 1.5;">
                    {res_payload['overall_recommendation']}
                </div>
            </div>
            """)
            st.markdown(rec_html, unsafe_allow_html=True)

            # Render Report Download Button
            render_report_download_button(
                res_payload["input_parameters"], results, res_payload["overall_recommendation"], domain_state=domain_state
            )

    else:
        st.markdown(
            textwrap.dedent("""
            <div style="border: 2px dashed rgba(148, 163, 184, 0.3); border-radius: 16px; padding: 48px 24px; text-align: center; background: rgba(30, 41, 59, 0.4); margin-top: 10px;">
                <div style="font-size: 2.8rem; margin-bottom: 12px;">🧪</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc;">No Sample Analyzed Yet</div>
                <div style="font-size: 0.92rem; margin-top: 8px; color: #cbd5e1; max-width: 460px; margin-left: auto; margin-right: auto; line-height: 1.5;">
                    Enter pH, TDS, NO3-N, and Well Depth on the left panel and click <strong style="color: #60a5fa;">ANALYZE SAMPLE</strong> to generate preliminary heavy-metal screening predictions.
                </div>
            </div>
            """),
            unsafe_allow_html=True
        )


# ------------------------------------------------------------------------------
# PERSISTENT RESEARCH DISCLAIMER FOOTER
# ------------------------------------------------------------------------------
st.markdown(
    textwrap.dedent(f"""
    <div class="footer-disclaimer">
        ⚠️ <strong>RESEARCH DISCLAIMER:</strong> {DISCLAIMER_TEXT}
    </div>
    """),
    unsafe_allow_html=True
)
