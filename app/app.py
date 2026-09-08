"""
GROUNDWATER AI MULTI-MODAL SCREENING ARCHITECTURE
North Bengal Study — Regression + Joint Classification + Geospatial Intelligence
"""
import os, sys, textwrap
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import streamlit as st
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
    build_plotly_spatial_map,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Groundwater AI Multi-Modal Screening — North Bengal",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(textwrap.dedent("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@700;800&display=swap');
.main-title { font-family: 'Outfit',sans-serif; font-size:2.2rem; font-weight:800;
    background:linear-gradient(135deg,#60a5fa,#2563eb); -webkit-background-clip:text;
    -webkit-text-fill-color:transparent; }
.subtitle { font-family:'Inter',sans-serif; font-size:1rem; color:#cbd5e1; margin-top:4px; }
.badge { display:inline-block; background:rgba(59,130,246,0.15); color:#60a5fa;
    border:1px solid rgba(59,130,246,0.35); padding:5px 14px; border-radius:20px;
    font-size:0.82rem; font-weight:600; margin-top:8px; }
.rec-box { background:rgba(30,41,59,0.6); border:1px solid rgba(59,130,246,0.3);
    border-left:5px solid #3b82f6; border-radius:12px; padding:16px 20px; margin-top:16px; }
.footer { margin-top:36px; padding:16px; background:rgba(30,41,59,0.4);
    border:1px solid rgba(148,163,184,0.2); border-radius:10px;
    font-size:0.83rem; color:#cbd5e1; text-align:center; }
</style>
"""), unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
models, conformal, load_errors = ModelLoader.load_all()

# Preset dictionary
PRESETS = {
    "standard":   {"ph": 7.10, "tds": 220.0, "no3": 1.20, "depth": 25.0, "thana": "Dinajpur Sadar"},
    "deep_well":  {"ph": 7.25, "tds": 310.0, "no3": 0.50, "depth": 55.0, "thana": "Pabna Sadar"},
    "high_tds":   {"ph": 7.40, "tds": 750.0, "no3": 8.50, "depth": 18.0, "thana": "Bogra Sadar"},
}

stations = get_all_stations()
station_names = [f"{s['thana']} ({s['district']})" for s in stations]
station_dict = {f"{s['thana']} ({s['district']})": s for s in stations}

# ── Session state init ────────────────────────────────────────────────────────
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

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Groundwater AI Multi-Modal Screening</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">North Bengal Aquifer Study — Continuous Regression · Multi-Metal Classification · Geospatial Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="badge">Hybrid Architecture · Conformal 90% Coverage · Spatial Proximity Interpolation</div>', unsafe_allow_html=True)

st.markdown("<hr style='margin:18px 0 24px 0;'>", unsafe_allow_html=True)

# ── Sidebar: Presets & Geographical Settings ─────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Sample Presets")
    pcols = st.columns(3)
    if pcols[0].button("Standard", use_container_width=True):
        apply_preset("standard"); st.rerun()
    if pcols[1].button("Deep Well", use_container_width=True):
        apply_preset("deep_well"); st.rerun()
    if pcols[2].button("High TDS", use_container_width=True):
        apply_preset("high_tds"); st.rerun()

    st.markdown("---")
    st.markdown("### 📍 Location & Geospatial Context")
    selected_thana = st.selectbox(
        "Nearest Upazila / Station (North Bengal):",
        options=station_names,
        key="thana_select",
        on_change=on_thana_change
    )
    
    col_lat, col_lon = st.columns(2)
    lat_input = col_lat.number_input("Latitude (°N)", value=float(st.session_state["lat"]), format="%.4f", step=0.01)
    lon_input = col_lon.number_input("Longitude (°E)", value=float(st.session_state["lon"]), format="%.4f", step=0.01)
    
    st.session_state["lat"] = lat_input
    st.session_state["lon"] = lon_input

    st.markdown("---")
    st.info(
        "**Geospatial Mode Active**\n\n"
        "Combines 4 field chemical inputs with GIS coordinates for Tobler-law spatial hazard weighting across 40 regional monitoring wells."
    )

# ── Input Form ────────────────────────────────────────────────────────────────
st.markdown("### 🧪 Field Hydrochemistry Parameters")
with st.form("prediction_form"):
    col1, col2, col3, col4 = st.columns(4)

    ph_val = col1.number_input(
        "pH (proxy / measured)",
        min_value=0.0, max_value=14.0,
        value=float(st.session_state["ph"]),
        step=0.05, format="%.2f",
        help="Carbonate-equilibrium proxy or field electrochemical pH (empirical range: 6.87 – 7.32)"
    )
    tds_val = col2.number_input(
        "TDS (mg/L)",
        min_value=0.0, max_value=10000.0,
        value=float(st.session_state["tds"]),
        step=10.0, format="%.1f",
        help="Total Dissolved Solids by handheld conductivity meter (empirical range: 38 – 537 mg/L)"
    )
    no3_val = col3.number_input(
        "NO3-N (mg/L)",
        min_value=0.0, max_value=500.0,
        value=float(st.session_state["no3"]),
        step=0.1, format="%.2f",
        help="Nitrate-nitrogen concentration (empirical range: 0.10 – 12.50 mg/L)"
    )
    depth_val = col4.number_input(
        "Well Depth (m)",
        min_value=0.0, max_value=1000.0,
        value=float(st.session_state["depth"]),
        step=1.0, format="%.1f",
        help="Tubewell installation depth below ground level (empirical range: 9 – 61 m)"
    )

    submitted = st.form_submit_button("🔍 ANALYZE MULTI-MODAL WATER QUALITY", use_container_width=True)

# ── Prediction execution ──────────────────────────────────────────────────────
if submitted:
    st.session_state["ph"]    = ph_val
    st.session_state["tds"]   = tds_val
    st.session_state["no3"]   = no3_val
    st.session_state["depth"] = depth_val

    with st.spinner("Running Multi-Modal Inference Pipeline..."):
        # 1. Continuous Regression
        res = PredictionService.predict(ph_val, tds_val, no3_val, depth_val)
        # 2. Joint Risk Classification
        risk_res = RiskClassificationService.predict_risk(
            ph_val, tds_val, no3_val, depth_val,
            st.session_state["lat"], st.session_state["lon"]
        )
        # 3. Geospatial Proximity & Interpolation
        spatial_interp = interpolate_spatial_risk(st.session_state["lat"], st.session_state["lon"])
        nearest_st = find_nearest_stations(st.session_state["lat"], st.session_state["lon"], top_k=3)

    if not res["success"]:
        st.error("### ❌ Validation Error")
        for err in res["errors"]:
            st.error(f"• {err}")
    else:
        st.markdown("<hr style='margin:20px 0;'>", unsafe_allow_html=True)
        
        # ══════════════════════════════════════════════════════════════════════
        # MODALITY 1: JOINT MULTI-METAL RISK CATEGORY (CLASSIFICATION)
        # ══════════════════════════════════════════════════════════════════════
        badge_bg = {
            "LOW_RISK": "rgba(16, 185, 129, 0.15)",
            "MODERATE_RISK": "rgba(245, 158, 11, 0.15)",
            "ELEVATED_RISK": "rgba(239, 68, 68, 0.15)"
        }.get(risk_res["risk_label"], "rgba(59, 130, 246, 0.15)")

        badge_border = {
            "LOW_RISK": "#10b981",
            "MODERATE_RISK": "#f59e0b",
            "ELEVATED_RISK": "#ef4444"
        }.get(risk_res["risk_label"], "#3b82f6")

        st.markdown(textwrap.dedent(f"""
        <div style="background:{badge_bg};border:2px solid {badge_border};border-radius:14px;padding:22px;margin-bottom:24px;">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
                <div>
                    <span style="font-size:0.85rem;color:#94a3b8;text-transform:uppercase;font-weight:700;letter-spacing:1px;">Joint Multi-Metal Risk Classifier</span>
                    <h2 style="margin:4px 0;font-size:1.8rem;color:#f8fafc;">{risk_res['risk_badge']}</h2>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:0.85rem;color:#94a3b8;">Classification Confidence</div>
                    <div style="font-size:1.6rem;font-weight:800;color:{badge_border};">{risk_res['confidence_percent']}%</div>
                </div>
            </div>
            <p style="color:#e2e8f0;font-size:0.95rem;margin:12px 0 8px 0;line-height:1.5;">{risk_res['risk_description']}</p>
            <div style="background:rgba(15,23,42,0.6);border-left:4px solid {badge_border};padding:10px 16px;border-radius:8px;font-size:0.88rem;color:#cbd5e1;">
                <strong>Field Advisory:</strong> {risk_res['recommended_action']}
            </div>
            <div style="margin-top:12px;font-size:0.82rem;color:#94a3b8;">
                📍 <strong>Spatial Anchor:</strong> Nearest monitored station is <strong>{spatial_interp.get('nearest_station', 'Unknown')}</strong> ({spatial_interp.get('nearest_distance_km', 0.0)} km away) | Regional Interpolated Hazard Index: <strong>{spatial_interp.get('interpolated_mhi', 0.0)}</strong>
            </div>
        </div>
        """), unsafe_allow_html=True)

        # ── Domain applicability banner ───────────────────────────────────────
        if res["domain_state"] == "OUT_OF_DOMAIN":
            st.warning(
                "⚠️ **OUTSIDE EMPIRICAL TRAINING DISTRIBUTION**\n\n"
                + "\n".join(f"• {w}" for w in res["ood_warnings"])
                + "\n\n*Prediction confidence may be attenuated. Confirmatory testing advised.*"
            )
        else:
            st.success("🟢 **IN-DOMAIN SAMPLE**: Chemical parameters match empirical training distribution.")

        # ══════════════════════════════════════════════════════════════════════
        # MODALITY 2: CONTINUOUS REGRESSION SCREENING (NI & CD)
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("### 📊 Specific Trace Metal Predictions (Continuous Regression)")
        
        reg_cols = st.columns(2)
        for idx, metal_key in enumerate(["ni", "cd"]):
            mr   = res["results"][metal_key]
            cfg  = mr["config"]
            pred = mr["prediction"]
            unc  = mr["uncertainty"]
            dec  = mr["decision"]
            thr  = dec["threshold"]

            from utils.formatting import get_status_badge_style, format_threshold_distance
            badge = get_status_badge_style(dec["status_code"])
            conf_text = (f"{unc['lower_bound']:.3f} – {unc['upper_bound']:.3f} {cfg['unit']}"
                         if unc["available"] else "Unavailable")
            width_text = (f"±{unc['quantile_q']:.3f} {cfg['unit']}" if unc["available"] else "N/A")
            dist_str   = format_threshold_distance(pred, thr, cfg["unit"])

            with reg_cols[idx]:
                st.markdown(textwrap.dedent(f"""
                <div style="border:1px solid rgba(148,163,184,0.25);border-radius:14px;
                     padding:18px;background:rgba(30,41,59,0.5);margin-bottom:16px;">
                  <div style="display:flex;justify-content:space-between;align-items:center;
                       border-bottom:1px solid rgba(148,163,184,0.18);padding-bottom:10px;">
                    <span style="font-size:1.25rem;font-weight:700;color:#f8fafc;">{cfg['name']}</span>
                    <span style="background:{badge['bg']};color:{badge['text']};border:1px solid {badge['border']};
                          padding:3px 10px;border-radius:20px;font-size:0.75rem;font-weight:700;">{dec['confidence_label']}</span>
                  </div>
                  <div style="display:flex;flex-wrap:wrap;margin-top:12px;gap:12px;">
                    <div style="flex:1;min-width:110px;">
                      <div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;">Predicted Value</div>
                      <div style="font-size:1.8rem;font-weight:800;color:#60a5fa;">{pred:.3f} <span style="font-size:0.85rem;color:#cbd5e1;">{cfg['unit']}</span></div>
                    </div>
                    <div style="flex:1;min-width:140px;">
                      <div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;">90% Conformal Interval</div>
                      <div style="font-size:1.05rem;font-weight:700;color:#38bdf8;margin-top:4px;">{conf_text}</div>
                      <div style="font-size:0.72rem;color:#94a3b8;">{width_text} · Empirical 90% coverage</div>
                    </div>
                    <div style="flex:1;min-width:110px;">
                      <div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;">Screening Limit</div>
                      <div style="font-size:1.05rem;font-weight:700;color:#cbd5e1;margin-top:4px;">{thr:.1f} {cfg['unit']}</div>
                      <div style="font-size:0.72rem;color:#94a3b8;">{dist_str}</div>
                    </div>
                  </div>
                  <div style="margin-top:12px;padding:10px 14px;border-radius:8px;
                       background:{badge['bg']};border-left:4px solid {badge['border']};">
                    <div style="font-size:0.85rem;font-weight:700;color:{badge['text']};">{badge['label']}</div>
                  </div>
                </div>
                """), unsafe_allow_html=True)

                # Threshold chart
                max_x  = max(thr * 1.3, (unc["upper_bound"] or pred) * 1.25, pred * 1.3)
                fig_t = go.Figure()
                if unc["available"]:
                    fig_t.add_trace(go.Scatter(x=[unc["lower_bound"], unc["upper_bound"]], y=[0,0],
                        mode="lines", line=dict(color="#3b82f6", width=7), name="90% Interval"))
                fig_t.add_trace(go.Scatter(x=[pred], y=[0], mode="markers+text",
                    marker=dict(color="#2563eb", size=14, line=dict(color="#fff", width=2)),
                    text=[f"{pred:.3f}"], textposition="top center", name="Prediction"))
                fig_t.add_shape(type="line", x0=thr, y0=-0.3, x1=thr, y1=0.3,
                    line=dict(color="#dc2626", width=3, dash="dash"))
                fig_t.add_annotation(x=thr, y=0.38, text=f"LIMIT: {thr} {cfg['unit']}",
                    showarrow=False, font=dict(color="#dc2626", size=10),
                    bgcolor="#fef2f2", bordercolor="#f87171", borderwidth=1, borderpad=2)
                fig_t.update_layout(
                    height=160, showlegend=False,
                    margin=dict(l=5, r=5, t=30, b=20),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0"),
                    xaxis=dict(range=[0, max_x], title=f"Concentration ({cfg['unit']})", gridcolor="rgba(148,163,184,0.15)"),
                    yaxis=dict(visible=False)
                )
                st.plotly_chart(fig_t, use_container_width=True)

        # ══════════════════════════════════════════════════════════════════════
        # MODALITY 3: GEOSPATIAL MAP & REGIONAL SURVEILLANCE
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("<hr style='margin:20px 0;'>", unsafe_allow_html=True)
        st.markdown("### 🗺️ Geospatial Groundwater Surveillance Map")
        st.markdown(
            "Interactive GIS layer displaying 40 monitored tubewell stations across 16 North Bengal districts. "
            "Click on stations to inspect localized trace metal concentrations and multi-metal vulnerability ratings."
        )

        fig_map = build_plotly_spatial_map(
            user_lat=st.session_state["lat"],
            user_lon=st.session_state["lon"],
            user_risk_label=risk_res["risk_label"]
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Nearest Reference Monitoring Wells Table
        st.markdown("#### 📍 Proximity to Monitored Reference Wells")
        near_records = []
        for s in nearest_st:
            near_records.append({
                "Station ID": s["sample_id"],
                "Upazila / District": f"{s['thana']}, {s['district']}",
                "Distance (km)": f"{s['distance_km']} km",
                "Well Depth": f"{s['depth_m']} m",
                "pH": s["ph"],
                "TDS (mg/L)": s["tds_mg_l"],
                "Ni (µg/L)": s["ni_ug_l"],
                "Cd (µg/L)": s["cd_ug_l"],
                "As (µg/L)": s["as_ug_l"],
                "Hazard Score (MHI)": s["mhi_score"],
                "Risk Status": s["risk_category"]
            })
        st.dataframe(pd.DataFrame(near_records), use_container_width=True, hide_index=True)

        # ── Overall recommendation & Report ───────────────────────────────────
        st.markdown(textwrap.dedent(f"""
        <div class="rec-box">
          <div style="font-size:0.8rem;color:#94a3b8;font-weight:700;text-transform:uppercase;">Overall Screening Synthesis</div>
          <div style="font-size:1rem;color:#f8fafc;margin-top:6px;line-height:1.6;">{res['overall_recommendation']}</div>
        </div>
        """), unsafe_allow_html=True)

        # Download report
        st.markdown("<br>", unsafe_allow_html=True)
        from components.report_generator import generate_html_report
        html_content = generate_html_report(
            res["input_parameters"],
            res["results"],
            res["overall_recommendation"]
        )
        st.download_button(
            label="📥 DOWNLOAD VERIFIED SCREENING AUDIT REPORT (HTML)",
            data=html_content.encode("utf-8"),
            file_name="Groundwater_AI_Screening_Report.html",
            mime="text/html",
            use_container_width=True,
        )

# ── Footer & Regulatory Disclaimer ───────────────────────────────────────────
st.markdown(textwrap.dedent(f"""
<div class="footer">
  <div style="font-weight:700;color:#94a3b8;margin-bottom:6px;">REGULATORY DISCLAIMER & ETHICAL USAGE NOTICE</div>
  <div style="line-height:1.5;">{DISCLAIMER_TEXT}</div>
  <div style="margin-top:8px;color:#64748b;font-size:0.75rem;">
    Architecture: Multi-Modal (HuberRegressor + Ridge + Conformal Calibration + Spatial KNN + OpenStreetMap)
  </div>
</div>
"""), unsafe_allow_html=True)
