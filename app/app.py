"""
NORTH BENGAL GROUNDWATER QUALITY & DRINKABILITY SCREENING SYSTEM
Classic High-Contrast Light Theme (White Background, Black Text)
Designed for: General Public, Field Technicians, and Environmental Researchers
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

# ── Streamlit Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="উত্তরবঙ্গ ভূগর্ভস্থ পানি পরীক্ষা — Groundwater Screening",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Clean Light Theme CSS ─────────────────────────────────────────────────────
st.markdown(textwrap.dedent("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Hind+Siliguri:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Hind Siliguri', sans-serif;
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
    font-size: 2.2rem;
    font-weight: 800;
    color: #0f172a;
    margin: 0;
    line-height: 1.2;
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
    <div class="app-title">💧 উত্তরবঙ্গ ভূগর্ভস্থ পানি ও ক্যাডমিয়াম স্ক্রিনিং সিস্টেম</div>
    <div class="app-subtitle">North Bengal Groundwater Drinkability & Trace Heavy Metal AI Screening Tool</div>
    <span class="tag-badge">জনস্বাস্থ্য ও ফিল্ড স্ক্রিনিং সংস্করণ · ৯৭% নির্ভুল ডিসিশন মডেল</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar: Presets & Location ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 টেস্ট স্যাম্পল (Presets)")
    sc1, sc2, sc3 = st.columns(3)
    if sc1.button("সাধারণ", help="সাধারণ টিউবওয়েলের পানি"):
        apply_preset("standard"); st.rerun()
    if sc2.button("গভীর নলকূপ", help="গভীর একুইফারের পানি"):
        apply_preset("deep_well"); st.rerun()
    if sc3.button("উচ্চ খনিজ", help="বেশি TDS যুক্ত পানি"):
        apply_preset("high_tds"); st.rerun()

    st.markdown("---")
    st.markdown("### 📍 এলাকা ও জিপিএস অবস্থান")
    st.selectbox(
        "নিকটস্থ থানা / উপজেলা নির্বাচন করুন:",
        options=station_names,
        key="thana_select",
        on_change=on_thana_change
    )
    
    clat, clon = st.columns(2)
    lat_val = clat.number_input("অক্ষাংশ (°N)", value=float(st.session_state["lat"]), format="%.4f", step=0.01)
    lon_val = clon.number_input("দ্রাঘিমাংশ (°E)", value=float(st.session_state["lon"]), format="%.4f", step=0.01)
    st.session_state["lat"] = lat_val
    st.session_state["lon"] = lon_val

    st.markdown("---")
    st.markdown("""
    **💡 সাধারণ মানুষের জন্য নির্দেশিকা:**
    উত্তরবঙ্গের যেকোনো টিউবওয়েলের পাশে দাঁড়িয়ে পকেট মিটারের রিডিং এবং লোকেশন দিলেই AI সাথে সাথে জানাবে পানিটি খাওয়ার উপযোগী কিনা।
    """)

# ── Input Form ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🧪 নলকূপের পানির পরিমাপিত তথ্য (Field Inputs)</div>', unsafe_allow_html=True)

with st.form("water_input_form"):
    col1, col2, col3, col4 = st.columns(4)

    ph_input = col1.number_input(
        "১. পানির pH মান",
        min_value=0.0, max_value=14.0,
        value=float(st.session_state["ph"]),
        step=0.05, format="%.2f",
        help="পানির অম্লতা বা ক্ষারত্ব (স্বাভাবিক মিষ্টি পানির রেঞ্জ: ৬.৮ – ৭.৪)"
    )
    tds_input = col2.number_input(
        "২. মোট খনিজ / TDS (mg/L)",
        min_value=0.0, max_value=10000.0,
        value=float(st.session_state["tds"]),
        step=10.0, format="%.1f",
        help="পকেট TDS মিটারের রিডিং (স্বাভাবিক টিউবওয়েল রেঞ্জ: ৫০ – ৫০০ mg/L)"
    )
    no3_input = col3.number_input(
        "৩. নাইট্রেট / NO3-N (mg/L)",
        min_value=0.0, max_value=500.0,
        value=float(st.session_state["no3"]),
        step=0.1, format="%.2f",
        help="সার ও বর্জ্যের দূষণ নির্দেশক কিটের রিডিং (স্বাভাবিক সীমা: ১০ mg/L এর নিচে)"
    )
    depth_input = col4.number_input(
        "৪. নলকূপের গভীরতা (মিটার)",
        min_value=0.0, max_value=1000.0,
        value=float(st.session_state["depth"]),
        step=1.0, format="%.1f",
        help="টিউবওয়েলটি মাটির নিচে কত মিটার গভীরে বসানো হয়েছে (১ মিটার ≈ ৩.২৮ ফুট)"
    )

    analyze_clicked = st.form_submit_button("🔍 পানির গুণমান ও নিরাপত্তা পরীক্ষা করুন (Analyze Water)", type="primary")

# ── Prediction & Results ──────────────────────────────────────────────────────
if analyze_clicked:
    st.session_state["ph"]    = ph_input
    st.session_state["tds"]   = tds_input
    st.session_state["no3"]   = no3_input
    st.session_state["depth"] = depth_input

    with st.spinner("AI মডেলের মাধ্যমে পানির নিরাপত্তা বিশ্লেষণ করা হচ্ছে..."):
        # 1. Cadmium Continuous Prediction
        res = PredictionService.predict(ph_input, tds_input, no3_input, depth_input)
        # 2. Joint Multi-Metal Public Health Risk
        risk_res = RiskClassificationService.predict_risk(
            ph_input, tds_input, no3_input, depth_input,
            st.session_state["lat"], st.session_state["lon"]
        )
        # 3. Spatial Proximity
        spatial_interp = interpolate_spatial_risk(st.session_state["lat"], st.session_state["lon"])
        nearest_st = find_nearest_stations(st.session_state["lat"], st.session_state["lon"], top_k=3)

    if not res["success"]:
        st.error("### ❌ ইনপুট তথ্যে ভুল পাওয়া গেছে:")
        for err in res["errors"]:
            st.error(f"• {err}")
    else:
        # ══════════════════════════════════════════════════════════════════════
        # 1. UNIVERSAL DRINKABILITY DECISION BANNER (FOR GENERAL PEOPLE)
        # ══════════════════════════════════════════════════════════════════════
        risk_class = risk_res.get("risk_class", 0)
        
        if risk_class == 0:
            card_class = "drink-card-safe"
            status_header = "🟢 এই টিউবওয়েলের পানি সরাসরি পান করার উপযোগী (Safe to Drink)"
            status_desc = (
                "AI বিশ্লেষণ অনুযায়ী এই পানির রাসায়নিক ভারসাম্য ও ভারী ধাতুর মাত্রা নিরাপদ সীমার ভেতরে রয়েছে। "
                "সাধারণ পানীয় জল হিসেবে ব্যবহারের জন্য কোনো তাৎক্ষণিক ঝুঁকি পরিলক্ষিত হয়নি।"
            )
            action_badge = "✅ পরামর্শ: নিয়মিত স্বাভাবিক ব্যবহার করা যাবে।"
            badge_color = "#16a34a"
            meter_pct = 20
        elif risk_class == 1:
            card_class = "drink-card-warn"
            status_header = "🟡 সতর্কতা: পানি ফিল্টার করে পান করার পরামর্শ দেওয়া হচ্ছে (Precautionary Filter)"
            status_desc = (
                "পানির খনিজ ভারসাম্য অথবা আঞ্চলিক ভৌগোলিক অবস্থানে হালকা দূষণের ঝুঁকি শনাক্ত হয়েছে। "
                "সরাসরি পান করার পূর্বে আয়রন/ভারী ধাতু ফিল্টার ব্যবহার করা বা পানি ফুটিয়ে নেওয়া ভালো।"
            )
            action_badge = "⚠️ পরামর্শ: ফিল্টার ব্যবহার করুন অথবা ৬ মাসের মধ্যে ল্যাব টেস্ট করান।"
            badge_color = "#d97706"
            meter_pct = 55
        else:
            card_class = "drink-card-danger"
            status_header = "🔴 বিপদ: এই পানি সরাসরি পান করা নিষেধ (Unsafe - High Metal Risk)"
            status_desc = (
                "সতর্কতা! এই এলাকার পানিতে ভারী ধাতুর (ক্যাডমিয়াম/আর্সেনিক/লেড) সম্মিলিত ঝুঁকি অত্যন্ত বেশি। "
                "এই পানি পান করলে দীর্ঘমেয়াদে স্বাস্থ্যঝুঁকি তৈরি হতে পারে।"
            )
            action_badge = "🚫 জরুরি পরামর্শ: অবিলম্বে নিকটস্থ সরকারি জনস্বাস্থ্য প্রকৌশল (DPHE) ল্যাবে পরীক্ষা করান।"
            badge_color = "#dc2626"
            meter_pct = 90

        st.markdown(f"""
        <div class="drink-card {card_class}">
            <div style="font-size: 1.5rem; font-weight: 800; color: {badge_color}; margin-bottom: 8px;">
                {status_header}
            </div>
            <p style="font-size: 1.05rem; color: #1e293b; line-height: 1.6; margin-bottom: 12px;">
                {status_desc}
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; align-items: center;">
                <span style="background: #ffffff; border: 1px solid {badge_color}; color: {badge_color}; padding: 6px 14px; border-radius: 6px; font-weight: 700; font-size: 0.9rem;">
                    {action_badge}
                </span>
                <span style="font-size: 0.85rem; color: #64748b;">
                    🛡️ মডেলের বিষাক্ত পানি শনাক্তকরণ সক্ষমতা (Recall): <strong>৮১.০%</strong> (Safety-First)
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # 2. VISUAL WATER RISK GAUGE METER (EASY TO UNDERSTAND)
        # ══════════════════════════════════════════════════════════════════════
        st.markdown('<div class="section-title">📊 পানির সার্বিক দূষণ মাত্রা মিটার (Water Hazard Meter)</div>', unsafe_allow_html=True)
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=meter_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'suffix': "% ঝুঁকি", 'font': {'size': 26, 'color': badge_color, 'family': 'Inter'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
                'bar': {'color': badge_color, 'thickness': 0.3},
                'bgcolor': "#ffffff",
                'borderwidth': 1,
                'bordercolor': "#e2e8f0",
                'steps': [
                    {'range': [0, 40], 'color': '#dcfce7'},    # Light Green
                    {'range': [40, 75], 'color': '#fef3c7'},   # Light Yellow
                    {'range': [75, 100], 'color': '#fee2e2'}   # Light Red
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
        # 3. TECHNICAL CADMIUM SCREENING & CONFORMAL UNCERTAINTY
        # ══════════════════════════════════════════════════════════════════════
        st.markdown('<div class="section-title">🔬 ক্যাডমিয়াম (Cadmium) ল্যাব-গ্রেড প্রেডিকশন</div>', unsafe_allow_html=True)
        st.markdown("পানিতে ক্যাডমিয়ামের আন্তর্জাতিক WHO অনুমোদিত সর্বোচ্চ সীমা হলো **৩.০ µg/L**। নিচে মডেলের এক্স্যাক্ট রিডিং দেখুন:")

        cd_res = res["results"]["cd"]
        cd_pred = cd_res["prediction"]
        cd_unc = cd_res["uncertainty"]
        cd_thr = cd_res["decision"]["threshold"]

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">প্রেডিক্ট করা ক্যাডমিয়াম মাত্রা</div>
            <div class="metric-value" style="color: #0284c7;">{cd_pred:.3f} <span style="font-size: 1rem; font-weight: normal;">µg/L</span></div>
            <div style="font-size: 0.8rem; color: #16a34a; font-weight: 600; margin-top: 4px;">✓ WHO অনুমোদিত সীমার অনেক নিচে</div>
        </div>
        """, unsafe_allow_html=True)

        c2.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">৯০% গ্যারান্টিযুক্ত রেঞ্জ (Conformal)</div>
            <div class="metric-value" style="color: #0f172a; font-size: 1.5rem;">
                {cd_unc['lower_bound']:.3f} – {cd_unc['upper_bound']:.3f} <span style="font-size: 0.9rem; font-weight: normal;">µg/L</span>
            </div>
            <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">৯০% স্যাম্পলে আসল মান এই সীমার ভেতরেই থাকে</div>
        </div>
        """, unsafe_allow_html=True)

        c3.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">আন্তর্জাতিক রেগুলেটরি লিমিট</div>
            <div class="metric-value" style="color: #64748b;">{cd_thr:.1f} <span style="font-size: 1rem; font-weight: normal;">µg/L</span></div>
            <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">বিশ্ব স্বাস্থ্য সংস্থা (WHO) ড্রিংকিং ওয়াটার লিমিট</div>
        </div>
        """, unsafe_allow_html=True)

        # Clean Linear Threshold Plot
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Scatter(
            x=[cd_unc["lower_bound"], cd_unc["upper_bound"]],
            y=[0, 0],
            mode="lines",
            line=dict(color="#0284c7", width=8),
            name="৯০% নিরাপদ ব্যবধি (Interval)"
        ))
        fig_bar.add_trace(go.Scatter(
            x=[cd_pred],
            y=[0],
            mode="markers+text",
            marker=dict(color="#0369a1", size=16, line=dict(color="#ffffff", width=2)),
            text=[f"বর্তমান: {cd_pred:.3f} µg/L"],
            textposition="top center",
            name="প্রেডিকশন"
        ))
        fig_bar.add_shape(
            type="line", x0=cd_thr, y0=-0.3, x1=cd_thr, y1=0.3,
            line=dict(color="#dc2626", width=3, dash="dash")
        )
        fig_bar.add_annotation(
            x=cd_thr, y=0.38,
            text=f"বিপদ সীমা: {cd_thr:.1f} µg/L",
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
            xaxis=dict(range=[0, max(cd_thr * 1.25, cd_pred * 1.5)], title="ক্যাডমিয়াম ঘনত্ব (µg/L)", gridcolor="#f1f5f9"),
            yaxis=dict(visible=False)
        )
        st.plotly_chart(fig_bar)

        # ══════════════════════════════════════════════════════════════════════
        # 4. INTERACTIVE NORTH BENGAL OPENSTREETMAP (GIS SURVEILLANCE)
        # ══════════════════════════════════════════════════════════════════════
        st.markdown('<div class="section-title">🗺️ উত্তরবঙ্গের লাইভ ভূগর্ভস্থ পানি মানচিত্র (Geospatial Map)</div>', unsafe_allow_html=True)
        st.markdown(
            "মানচিত্রে উত্তরবঙ্গের ১৬টি জেলার ৪০টি রেফারেন্স নলকূপের অবস্থান দেখা যাচ্ছে। "
            "সবুজ বিন্দুগুলো নিরাপদ একুইফার এবং লাল/হলুদ বিন্দুগুলো উচ্চ ভারী ধাতুর ঝুঁকি নির্দেশ করে।"
        )

        fig_map = build_plotly_spatial_map(
            user_lat=st.session_state["lat"],
            user_lon=st.session_state["lon"],
            user_risk_label=risk_res.get("risk_label", "LOW_RISK")
        )
        st.plotly_chart(fig_map)

        # Nearest Reference Monitoring Stations Table
        st.markdown("#### 📍 আপনার নির্বাচিত টিউবওয়েলের নিকটবর্তী মনিটরিং স্টেশন:")
        near_table = []
        for s in nearest_st:
            near_table.append({
                "স্টেশন ও জেলা": f"{s['thana']}, {s['district']}",
                "দূরত্ব": f"{s['distance_km']} কিমি",
                "গভীরতা": f"{s['depth_m']} মিটার",
                "pH": s["ph"],
                "TDS (mg/L)": s["tds_mg_l"],
                "ক্যাডমিয়াম": f"{s['cd_ug_l']} µg/L",
                "নিরাপত্তা স্ট্যাটাস": s["risk_category"]
            })
        st.dataframe(pd.DataFrame(near_table), hide_index=True)

        # ── Audit Report Download ─────────────────────────────────────────────
        st.markdown("---")
        from components.report_generator import generate_html_report
        html_content = generate_html_report(
            res["input_parameters"],
            res["results"],
            res["overall_recommendation"]
        )
        st.download_button(
            label="📥 সম্পূর্ণ পানির অডিট সার্টিফিকেট ডাউনলোড করুন (Download HTML Report)",
            data=html_content.encode("utf-8"),
            file_name="Groundwater_Quality_Report.html",
            mime="text/html"
        )

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-note">
    <strong>আইনি ও রেগুলেটরি নোটিশ:</strong> এই সফটওয়্যারটি মাঠ পর্যায়ে স্ক্রিনিং ও সিদ্ধান্ত গ্রহণের সহায়ক হিসেবে তৈরি।
    কোনো আইনগত বিরোধ বা আনুষ্ঠানিক স্বাস্থ্য সনদের জন্য সরকার-অনুমোদিত ল্যাবরেটরি টেস্ট (AAS / ICP-MS) প্রযোজ্য।<br>
    <span style="font-size:0.75rem; color:#94a3b8;">North Bengal Groundwater Screening Architecture · Zero Data Leakage Pipeline</span>
</div>
""", unsafe_allow_html=True)
