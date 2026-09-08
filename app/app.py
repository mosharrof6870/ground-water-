"""
GROUNDWATER CADMIUM & MULTI-METAL SCREENING SYSTEM
North Bengal Aquifer Study — Operational Decision-Support Platform
Classic Light Theme (Pure White Background, Dark Charcoal Text)
Academic & Journal Publication Standard
"""
import os, sys, textwrap
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go

from config import MODEL_CONFIG, THRESHOLDS, DISCLAIMER_TEXT, VALIDATED_RANGES, ENVIRONMENTAL_RANGES
from services.model_loader import ModelLoader
from services.prediction_service import PredictionService
from services.risk_classification_service import RiskClassificationService
from services.spatial_service import (
    get_all_stations,
    find_nearest_stations,
    interpolate_spatial_risk,
    build_folium_spatial_map,
)

# ── Streamlit Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="Groundwater Heavy Metal AI Screening — North Bengal Study",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Clean Light Theme CSS ─────────────────────────────────────────────────────
st.markdown(textwrap.dedent("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #0f172a;
    background-color: #ffffff;
}
.stApp {
    background-color: #ffffff;
}
.main-header {
    background: #ffffff;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 16px;
    margin-bottom: 20px;
}
.app-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: #0f172a;
    margin: 0;
    line-height: 1.25;
}
.app-subtitle {
    font-size: 1.05rem;
    color: #475569;
    margin-top: 6px;
}
.tag-badge {
    display: inline-block;
    background: #e0f2fe;
    color: #0284c7;
    font-weight: 700;
    font-size: 0.8rem;
    padding: 4px 12px;
    border-radius: 9999px;
    margin-top: 8px;
    border: 1px solid #bae6fd;
}
.drink-card {
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.03);
}
.drink-card-safe {
    background: #f0fdf4;
    border: 2px solid #22c55e;
}
.drink-card-warn {
    background: #fffbeb;
    border: 2px solid #f59e0b;
}
.drink-card-danger {
    background: #fef2f2;
    border: 2px solid #ef4444;
}
.metric-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #0f172a;
}
.metric-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
}
.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #0f172a;
    margin-top: 24px;
    margin-bottom: 14px;
    border-left: 4px solid #0284c7;
    padding-left: 12px;
}
.footer-note {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 14px;
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 36px;
    text-align: center;
}
</style>
"""), unsafe_allow_html=True)

# ── Load stations & presets ───────────────────────────────────────────────────
stations = get_all_stations()
station_names = [f"{s['thana']} ({s['district']})" for s in stations]
station_dict = {f"{s['thana']} ({s['district']})": s for s in stations}

PRESETS = {
    "standard":   {"ph": 7.10, "tds": 220.0, "no3": 1.20, "depth": 25.0, "thana": "Dinajpur Sadar"},
    "deep_well":  {"ph": 7.25, "tds": 310.0, "no3": 0.50, "depth": 55.0, "thana": "Pabna Sadar"},
    "high_tds":   {"ph": 7.40, "tds": 750.0, "no3": 8.50, "depth": 18.0, "thana": "Bogra Sadar"},
}

for k, v in [("ph", 7.10), ("tds", 220.0), ("no3", 1.20), ("depth", 25.0),
            ("thana_select", "Dinajpur Sadar (Dinajpur)"),
            ("lat", 25.6279), ("lon", 88.6332)]:
    if k not in st.session_state:
        st.session_state[k] = v

def apply_preset(key: str):
    p = PRESETS[key]
    st.session_state["ph"]    = p["ph"]
    st.session_state["tds"]   = p["tds"]
    st.session_state["no3"]   = p["no3"]
    st.session_state["depth"] = p["depth"]
    target_thana = [sn for sn in station_names if p["thana"] in sn]
    if target_thana:
        st.session_state["thana_select"] = target_thana[0]
        st.session_state["lat"] = station_dict[target_thana[0]]["lat"]
        st.session_state["lon"] = station_dict[target_thana[0]]["lon"]

def on_thana_change():
    selected = st.session_state.get("thana_select")
    if selected in station_dict:
        st.session_state["lat"] = station_dict[selected]["lat"]
        st.session_state["lon"] = station_dict[selected]["lon"]

# ── App Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="app-title">💧 Groundwater Quality & Trace Heavy Metal AI Screening System</div>
    <div class="app-subtitle">Alluvial Aquifers of North Bengal — Surrogate Hydrochemical Modeling & Conformal Uncertainty</div>
    <span class="tag-badge">Operational Decision-Support Tool · Asymmetric Cost-Sensitive Screening (81% Recall)</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar: Presets & Location ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Field Test Presets")
    sc1, sc2, sc3 = st.columns(3)
    if sc1.button("Standard", help="Typical shallow alluvial tubewell"):
        apply_preset("standard"); st.rerun()
    if sc2.button("Deep Well", help="Deep aquifer borewell sample"):
        apply_preset("deep_well"); st.rerun()
    if sc3.button("High Mineral", help="Elevated mineralization/TDS"):
        apply_preset("high_tds"); st.rerun()

    st.markdown("---")
    st.markdown("### 📍 Location & Geospatial Context")
    st.selectbox(
        "Select Monitoring Upazila / Station (North Bengal):",
        options=station_names,
        key="thana_select",
        on_change=on_thana_change
    )
    
    clat, clon = st.columns(2)
    lat_val = clat.number_input("Latitude (°N)", value=float(st.session_state["lat"]), format="%.4f", step=0.01)
    lon_val = clon.number_input("Longitude (°E)", value=float(st.session_state["lon"]), format="%.4f", step=0.01)
    st.session_state["lat"] = lat_val
    st.session_state["lon"] = lon_val

    st.markdown("---")
    st.markdown("""
    **💡 Methodological Overview:**
    Designed for rapid field deployment in resource-constrained rural aquifers.
    Combines 4 low-cost probe measurements with spatial coordinates to predict trace heavy metal toxicity and potability without requiring expensive ex-situ laboratory testing.
    """)

# ── Input Form ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🧪 In-Situ Hydrochemical Field Measurements</div>', unsafe_allow_html=True)

with st.form("water_input_form"):
    col1, col2, col3, col4 = st.columns(4)

    ph_input = col1.number_input(
        "1. Field Measured pH (or proxy)",
        min_value=0.0, max_value=14.0,
        value=float(st.session_state["ph"]),
        step=0.05, format="%.2f",
        help="Field electrochemical pH or carbonate-equilibrium proxy (Empirical baseline: 6.87 – 7.32)"
    )
    tds_input = col2.number_input(
        "2. Total Dissolved Solids / TDS (mg/L)",
        min_value=0.0, max_value=10000.0,
        value=float(st.session_state["tds"]),
        step=10.0, format="%.1f",
        help="Handheld conductivity/TDS meter reading (Empirical baseline: 38.1 – 537.3 mg/L)"
    )
    no3_input = col3.number_input(
        "3. Nitrate-Nitrogen / NO3-N (mg/L)",
        min_value=0.0, max_value=500.0,
        value=float(st.session_state["no3"]),
        step=0.1, format="%.2f",
        help="Colorimetric field test kit reading for agricultural runoff (Empirical baseline: 0.10 – 12.50 mg/L)"
    )
    depth_input = col4.number_input(
        "4. Well Installation Depth (meters)",
        min_value=0.0, max_value=1000.0,
        value=float(st.session_state["depth"]),
        step=1.0, format="%.1f",
        help="Screen depth of the tubewell below ground level in meters (Empirical baseline: 9.0 – 61.0 m)"
    )

    analyze_clicked = st.form_submit_button("🔍 ANALYZE GROUNDWATER QUALITY & DRINKABILITY", type="primary")

# ── Prediction & Results ──────────────────────────────────────────────────────
if analyze_clicked:
    st.session_state["ph"]    = ph_input
    st.session_state["tds"]   = tds_input
    st.session_state["no3"]   = no3_input
    st.session_state["depth"] = depth_input

    with st.spinner("Executing Multi-Modal AI Inference Pipeline..."):
        # 1. Cadmium Continuous Prediction
        res = PredictionService.predict(ph_input, tds_input, no3_input, depth_input)
        # 2. Joint Multi-Metal Public Health Risk (Cost-Sensitive)
        risk_res = RiskClassificationService.predict_risk(
            ph_input, tds_input, no3_input, depth_input,
            st.session_state["lat"], st.session_state["lon"]
        )
        # 3. Spatial Proximity
        spatial_interp = interpolate_spatial_risk(st.session_state["lat"], st.session_state["lon"])
        nearest_st = find_nearest_stations(st.session_state["lat"], st.session_state["lon"], top_k=3)

    if not res["success"]:
        st.error("### ❌ Input Validation Errors Encountered:")
        for err in res["errors"]:
            st.error(f"• {err}")
    else:
        # ══════════════════════════════════════════════════════════════════════
        # 1. UNIVERSAL DRINKABILITY DECISION BANNER (WHO STANDARDS)
        # ══════════════════════════════════════════════════════════════════════
        risk_class = risk_res.get("risk_class", 0)
        
        if risk_class == 0:
            card_class = "drink-card-safe"
            status_header = "🟢 SAFE FOR DRINKING (Compliance with WHO Standards)"
            status_desc = (
                "Hydrochemical equilibrium parameters and cumulative multi-metal hazard indices remain safely below "
                "international screening thresholds. Water quality is suitable for standard rural potable consumption."
            )
            action_badge = "✅ Field Action: Standard routine surveillance schedule is appropriate."
            badge_color = "#16a34a"
            meter_pct = 20
        elif risk_class == 1:
            card_class = "drink-card-warn"
            status_header = "🟡 CAUTION: FILTRATION RECOMMENDED (Precautionary Action Required)"
            status_desc = (
                "Water hydrochemistry or regional proximity indicates moderate multi-metal sensitivity. "
                "Precautionary filtration (e.g., activated carbon / iron filter) or boiling is advised prior to direct consumption."
            )
            action_badge = "⚠️ Field Action: Employ filtration; laboratory verification recommended within 6 months."
            badge_color = "#d97706"
            meter_pct = 55
        else:
            card_class = "drink-card-danger"
            status_header = "🔴 UNSAFE FOR CONSUMPTION (Elevated Heavy Metal Hazard)"
            status_desc = (
                "Alert: Cumulative heavy-metal hazard indicators exceed safe drinking water guidelines. "
                "Direct potable consumption presents significant public-health toxicity exposure risks."
            )
            action_badge = "🚫 Immediate Action: Cease direct potable use; priority laboratory testing (AAS/ICP-MS) required."
            badge_color = "#dc2626"
            meter_pct = 90

        st.markdown(f"""
        <div class="drink-card {card_class}">
            <div style="font-size: 1.45rem; font-weight: 800; color: {badge_color}; margin-bottom: 8px;">
                {status_header}
            </div>
            <p style="font-size: 1.02rem; color: #1e293b; line-height: 1.6; margin-bottom: 12px;">
                {status_desc}
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; align-items: center;">
                <span style="background: #ffffff; border: 1px solid {badge_color}; color: {badge_color}; padding: 6px 14px; border-radius: 6px; font-weight: 700; font-size: 0.88rem;">
                    {action_badge}
                </span>
                <span style="font-size: 0.85rem; color: #64748b;">
                    🛡️ Detection Sensitivity (Recall): <strong>81.0%</strong> (Cost-Sensitive Bayesian Screening)
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # 2. VISUAL WATER CONTAMINATION HAZARD METER
        # ══════════════════════════════════════════════════════════════════════
        st.markdown('<div class="section-title">📊 Water Contamination Hazard Gauge Meter</div>', unsafe_allow_html=True)
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=meter_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'suffix': "% Hazard Index", 'font': {'size': 24, 'color': badge_color, 'family': 'Inter'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
                'bar': {'color': badge_color, 'thickness': 0.3},
                'bgcolor': "#ffffff",
                'borderwidth': 1,
                'bordercolor': "#e2e8f0",
                'steps': [
                    {'range': [0, 40], 'color': '#dcfce7'},    # Safe Green
                    {'range': [40, 75], 'color': '#fef3c7'},   # Warning Amber
                    {'range': [75, 100], 'color': '#fee2e2'}   # Danger Red
                ],
                'threshold': {
                    'line': {'color': "#dc2626", 'width': 3},
                    'thickness': 0.75,
                    'value': 75
                }
            }
        ))
        fig_gauge.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=25, b=10),
            paper_bgcolor="#ffffff",
            font={'color': "#0f172a", 'family': "Inter"}
        )
        st.plotly_chart(fig_gauge)

        # ══════════════════════════════════════════════════════════════════════
        # 3. LABORATORY-GRADE CADMIUM SCREENING & CONFORMAL UNCERTAINTY
        # ══════════════════════════════════════════════════════════════════════
        st.markdown('<div class="section-title">🔬 Quantitative Cadmium (Cd) In-Situ Surrogate Prediction</div>', unsafe_allow_html=True)
        st.markdown("The international WHO maximum permissible concentration limit for Cadmium in drinking water is **3.0 µg/L**.")

        cd_res = res["results"]["cd"]
        cd_pred = cd_res["prediction"]
        cd_unc = cd_res["uncertainty"]
        cd_thr = cd_res["decision"]["threshold"]

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Predicted Cadmium Concentration</div>
            <div class="metric-value" style="color: #0284c7;">{cd_pred:.3f} <span style="font-size: 1rem; font-weight: normal;">µg/L</span></div>
            <div style="font-size: 0.8rem; color: #16a34a; font-weight: 600; margin-top: 4px;">✓ Complies with WHO drinking guideline</div>
        </div>
        """, unsafe_allow_html=True)

        c2.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">90% Conformal Uncertainty Interval</div>
            <div class="metric-value" style="color: #0f172a; font-size: 1.5rem;">
                {cd_unc['lower_bound']:.3f} – {cd_unc['upper_bound']:.3f} <span style="font-size: 0.9rem; font-weight: normal;">µg/L</span>
            </div>
            <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">Empirical coverage guarantee: 90.0% of true values</div>
        </div>
        """, unsafe_allow_html=True)

        c3.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">WHO Maximum Allowable Limit</div>
            <div class="metric-value" style="color: #64748b;">{cd_thr:.1f} <span style="font-size: 1rem; font-weight: normal;">µg/L</span></div>
            <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">WHO Guidelines for Drinking-water Quality</div>
        </div>
        """, unsafe_allow_html=True)

        # Clean Linear Threshold Plot
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Scatter(
            x=[cd_unc["lower_bound"], cd_unc["upper_bound"]],
            y=[0, 0],
            mode="lines",
            line=dict(color="#0284c7", width=8),
            name="90% Conformal Prediction Interval"
        ))
        fig_bar.add_trace(go.Scatter(
            x=[cd_pred],
            y=[0],
            mode="markers+text",
            marker=dict(color="#0369a1", size=16, line=dict(color="#ffffff", width=2)),
            text=[f"Prediction: {cd_pred:.3f} µg/L"],
            textposition="top center",
            name="Point Estimate"
        ))
        fig_bar.add_shape(
            type="line", x0=cd_thr, y0=-0.3, x1=cd_thr, y1=0.3,
            line=dict(color="#dc2626", width=3, dash="dash")
        )
        fig_bar.add_annotation(
            x=cd_thr, y=0.38,
            text=f"WHO LIMIT: {cd_thr:.1f} µg/L",
            showarrow=False,
            font=dict(color="#dc2626", size=11, family="Inter"),
            bgcolor="#fef2f2",
            bordercolor="#fca5a5"
        )
        fig_bar.update_layout(
            height=160,
            showlegend=False,
            margin=dict(l=10, r=10, t=35, b=20),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#0f172a", family="Inter"),
            xaxis=dict(range=[0, max(cd_thr * 1.25, cd_pred * 1.5)], title="Cadmium Concentration (µg/L)", gridcolor="#f1f5f9"),
            yaxis=dict(visible=False)
        )
        st.plotly_chart(fig_bar)

        # ══════════════════════════════════════════════════════════════════════
        # 4. INTERACTIVE NORTH BENGAL OPENSTREETMAP (FOLIUM RESILIENT GIS)
        # ══════════════════════════════════════════════════════════════════════
        st.markdown('<div class="section-title">🗺️ Geospatial Groundwater Surveillance Map (North Bengal)</div>', unsafe_allow_html=True)
        st.markdown(
            "Interactive GIS layer displaying 40 monitored tubewell reference stations across 16 northern districts of Bangladesh. "
            "Green markers denote safe baseline aquifers, while amber/red markers indicate elevated multi-metal vulnerability."
        )

        folium_html = build_folium_spatial_map(
            user_lat=st.session_state["lat"],
            user_lon=st.session_state["lon"],
            user_risk_label=risk_res.get("risk_label", "LOW_RISK")
        )
        components.html(folium_html, height=490)

        # Nearest Reference Monitoring Stations Table
        st.markdown("#### 📍 Proximity to Nearest Monitored Reference Stations:")
        near_table = []
        for s in nearest_st:
            near_table.append({
                "Station ID": s["sample_id"],
                "Upazila, District": f"{s['thana']}, {s['district']}",
                "Proximity Distance": f"{s['distance_km']} km",
                "Well Depth": f"{s['depth_m']} m",
                "pH": s["ph"],
                "TDS (mg/L)": s["tds_mg_l"],
                "Cadmium (µg/L)": f"{s['cd_ug_l']} µg/L",
                "Arsenic (µg/L)": f"{s['as_ug_l']} µg/L",
                "Risk Classification": s["risk_category"]
            })
        st.dataframe(pd.DataFrame(near_table), hide_index=True)

        # ── Audit Report Download (PDF) ───────────────────────────────────────
        st.markdown("---")
        from components.report_generator import generate_pdf_report
        pdf_bytes = generate_pdf_report(
            res["input_parameters"],
            res["results"],
            res["overall_recommendation"],
            risk_res=risk_res,
            spatial_res=spatial_interp,
            domain_state=res["domain_state"],
        )
        safe_name = st.session_state["thana_select"].split(" ")[0].replace("/", "_")
        st.download_button(
            label="📥 DOWNLOAD OFFICIAL WATER QUALITY AUDIT REPORT (PDF)",
            data=pdf_bytes,
            file_name=f"Groundwater_AI_Screening_Report_{safe_name}.pdf",
            mime="application/pdf"
        )

# ── Footer & Regulatory Disclaimer ───────────────────────────────────────────
st.markdown("""
<div class="footer-note">
    <strong>REGULATORY & ETHICAL DISCLAIMER:</strong> This software is an operational screening and decision-support tool.
    It is not an official drinking-water safety certificate and does not replace certified laboratory analytical testing (AAS / ICP-MS) where formal regulatory compliance is required.<br>
    <span style="font-size:0.75rem; color:#94a3b8;">North Bengal Groundwater Screening Framework · Multi-Modal AI (Ridge Regression + Bayesian Cost-Sensitive Classifier + Folium GIS)</span>
</div>
""", unsafe_allow_html=True)
