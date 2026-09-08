"""
GROUNDWATER AI HEAVY-METAL SCREENING — Clean Streamlit App
Pipeline: 06_app (auto-configured from training artifacts)
"""
import os, sys, textwrap
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import streamlit as st
import pandas as pd

from config import MODEL_CONFIG, THRESHOLDS, DISCLAIMER_TEXT, VALIDATED_RANGES, ENVIRONMENTAL_RANGES
from services.model_loader import ModelLoader
from services.prediction_service import PredictionService

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Groundwater AI Screening",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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

# ── Startup: load models ──────────────────────────────────────────────────────
_, _, load_errors = ModelLoader.load_all()
if load_errors:
    for k, msg in load_errors.items():
        st.error(f"⚠️ [{k}] {msg}")
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state["history"] = []
if "preset" not in st.session_state:
    st.session_state["preset"] = {"ph": 7.10, "tds": 220.0, "no3": 1.20, "depth": 25.0}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(textwrap.dedent("""
<div style="border-bottom:2px solid rgba(59,130,246,0.3); padding-bottom:16px; margin-bottom:20px;">
  <div class="main-title">🧪 GROUNDWATER AI HEAVY-METAL SCREENING</div>
  <div class="subtitle">Hydrochemistry-informed screening · North Bengal Aquifer Framework</div>
  <div class="badge">🔬 Research Prototype — Inference Only</div>
</div>
"""), unsafe_allow_html=True)

st.info(
    f"Screening thresholds: Ni = {THRESHOLDS['Ni']} µg/L · Cd = {THRESHOLDS['Cd']} µg/L  "
    f"(verify against applicable current standard before regulatory use)"
)

# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.4], gap="large")

# ── LEFT: Input form ──────────────────────────────────────────────────────────
with left:
    st.markdown("### 🧪 Water Sample Parameters")
    st.caption("Enter field parameters. Values outside training range trigger out-of-domain warning.")

    # Preset buttons — use session_state so values persist across rerun
    c1, c2, c3 = st.columns(3)
    if c1.button("Standard Sample"):
        st.session_state["preset"] = {"ph": 7.10, "tds": 220.0, "no3": 1.20, "depth": 25.0}
        st.rerun()
    if c2.button("High TDS (OOD)"):
        st.session_state["preset"] = {"ph": 7.25, "tds": 750.0, "no3": 2.50, "depth": 30.0}
        st.rerun()
    if c3.button("High NO3-N (OOD)"):
        st.session_state["preset"] = {"ph": 6.95, "tds": 300.0, "no3": 15.0, "depth": 20.0}
        st.rerun()

    pv = st.session_state["preset"]
    vr = VALIDATED_RANGES or {}

    with st.form("gw_form"):
        ph_in    = st.number_input("pH",                        min_value=0.01, max_value=14.0,   value=float(pv["ph"]),    step=0.05,  help=f"Training range: {vr.get('pH', {}).get('min','?')} – {vr.get('pH', {}).get('max','?')}")
        tds_in   = st.number_input("TDS [mg/L]",               min_value=0.0,  max_value=10000.0, value=float(pv["tds"]),   step=10.0,  help=f"Training range: {vr.get('TDS', {}).get('min','?')} – {vr.get('TDS', {}).get('max','?')} mg/L")
        no3_in   = st.number_input("NO3-N [mg/L]",             min_value=0.0,  max_value=500.0,   value=float(pv["no3"]),   step=0.5,   help=f"Training range: {vr.get('NO3-N', {}).get('min','?')} – {vr.get('NO3-N', {}).get('max','?')} mg/L")
        depth_in = st.number_input("Well Depth [m]",           min_value=0.1,  max_value=1000.0,  value=float(pv["depth"]), step=1.0,   help=f"Training range: {vr.get('Well Depth', {}).get('min','?')} – {vr.get('Well Depth', {}).get('max','?')} m")
        submitted = st.form_submit_button("🔍 ANALYZE SAMPLE", type="primary")

    if submitted:
        with st.spinner("Analyzing groundwater sample..."):
            result = PredictionService.predict(ph_in, tds_in, no3_in, depth_in)
            st.session_state["last_result"] = result
            if result["success"]:
                st.session_state["history"].append({
                    "pH": ph_in, "TDS": tds_in, "NO3-N": no3_in, "Depth": depth_in,
                    "Domain":  result["domain_state"],
                    "Ni (µg/L)": result["results"]["ni"]["prediction"],
                    "Cd (µg/L)": result["results"]["cd"]["prediction"],
                    "Status":  result["overall_recommendation"][:60] + "...",
                })

    # History
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📜 SESSION HISTORY", expanded=False):
        if st.session_state["history"]:
            st.dataframe(pd.DataFrame(st.session_state["history"]))
            if st.button("🗑️ Clear History"):
                st.session_state["history"] = []
                st.rerun()
        else:
            st.caption("No analyses in this session.")

# ── RIGHT: Results ────────────────────────────────────────────────────────────
with right:
    if "last_result" not in st.session_state:
        st.markdown(textwrap.dedent("""
        <div style="border:2px dashed rgba(148,163,184,0.3); border-radius:16px;
             padding:48px 24px; text-align:center; background:rgba(30,41,59,0.4);">
          <div style="font-size:2.8rem; margin-bottom:12px;">🧪</div>
          <div style="font-size:1.2rem; font-weight:700; color:#f8fafc;">No Sample Analyzed Yet</div>
          <div style="font-size:0.9rem; color:#cbd5e1; margin-top:8px;">
            Enter parameters on the left and click <strong>ANALYZE SAMPLE</strong>.
          </div>
        </div>
        """), unsafe_allow_html=True)
    else:
        res = st.session_state["last_result"]

        if not res["success"]:
            st.error("❌ Input Rejected")
            for e in res["errors"]:
                st.markdown(f"• **{e}**")
        else:
            st.markdown("### 📊 Screening Results")
            domain = res["domain_state"]

            if domain == "OUT_OF_DOMAIN":
                st.warning(
                    "⚠️ **OUTSIDE VALIDATED MODEL DOMAIN**\n\n"
                    + "\n".join(f"• {w}" for w in res["ood_warnings"])
                    + "\n\n*Laboratory confirmation recommended.*"
                )
            else:
                st.success("🟢 **IN-DOMAIN**: All inputs within empirical training distribution.")

            # — Applicability table —
            st.markdown("### 🎯 Model Applicability")
            if VALIDATED_RANGES:
                param_map = [
                    ("pH",        res["input_parameters"]["pH"],         "pH",         "-"),
                    ("TDS",       res["input_parameters"]["TDS"],        "TDS",        "mg/L"),
                    ("NO3-N",     res["input_parameters"]["NO3-N"],      "NO3-N",      "mg/L"),
                    ("Well Depth",res["input_parameters"]["well_depth"], "Well Depth", "m"),
                ]
                acols = st.columns(4)
                for i, (label, val, rkey, unit) in enumerate(param_map):
                    vr_entry = VALIDATED_RANGES.get(rkey, {})
                    v_min, v_max = vr_entry.get("min", 0), vr_entry.get("max", 9999)
                    in_rng = v_min <= val <= v_max
                    color  = "#34d399" if in_rng else "#f87171"
                    bg     = "rgba(16,185,129,0.1)" if in_rng else "rgba(239,68,68,0.1)"
                    border = "rgba(16,185,129,0.3)" if in_rng else "rgba(239,68,68,0.3)"
                    icon   = "✓" if in_rng else "⚠️"
                    u_str  = f" {unit}" if unit != "-" else ""
                    fmt_v  = f"{val:.2f}" if rkey in ("pH","NO3-N") else f"{val:.1f}"
                    acols[i].markdown(
                        f'<div style="background:{bg};border:1px solid {border};border-radius:10px;'
                        f'padding:12px;text-align:center;">'
                        f'<div style="font-size:0.78rem;color:#94a3b8;font-weight:600;text-transform:uppercase;">{label}</div>'
                        f'<div style="font-size:1.3rem;font-weight:800;color:#f8fafc;margin:4px 0;">{fmt_v}<span style="font-size:0.8rem;color:#cbd5e1;">{u_str}</span></div>'
                        f'<div style="font-size:0.72rem;color:#94a3b8;">Range: {v_min}{u_str}–{v_max}{u_str}</div>'
                        f'<div style="font-size:0.75rem;font-weight:700;color:{color};margin-top:6px;">{icon} {"Within" if in_rng else "Outside"} range</div>'
                        f'</div>', unsafe_allow_html=True
                    )

            st.markdown("<hr style='margin:20px 0;'>", unsafe_allow_html=True)

            # — Result cards for each metal —
            import plotly.graph_objects as go
            for metal_key in ["ni", "cd"]:
                mr   = res["results"][metal_key]
                cfg  = mr["config"]
                pred = mr["prediction"]
                unc  = mr["uncertainty"]
                dec  = mr["decision"]
                thr  = dec["threshold"]

                # Badge
                from utils.formatting import get_status_badge_style, format_threshold_distance
                badge = get_status_badge_style(dec["status_code"])
                conf_text = (f"{unc['lower_bound']:.2f} – {unc['upper_bound']:.2f} {cfg['unit']}"
                             if unc["available"] else "Unavailable")
                width_text = (f"±{unc['quantile_q']:.2f} {cfg['unit']}" if unc["available"] else "N/A")
                dist_str   = format_threshold_distance(pred, thr, cfg["unit"])

                st.markdown(textwrap.dedent(f"""
                <div style="border:1px solid rgba(148,163,184,0.25);border-radius:14px;
                     padding:20px;background:rgba(30,41,59,0.5);margin-bottom:16px;">
                  <div style="display:flex;justify-content:space-between;align-items:center;
                       border-bottom:1px solid rgba(148,163,184,0.18);padding-bottom:12px;">
                    <span style="font-size:1.3rem;font-weight:700;color:#f8fafc;">{cfg['name']} ({cfg['symbol']})</span>
                    <span style="background:{badge['bg']};color:{badge['text']};border:1px solid {badge['border']};
                          padding:4px 12px;border-radius:20px;font-size:0.78rem;font-weight:700;">{dec['confidence_label']}</span>
                  </div>
                  <div style="display:flex;flex-wrap:wrap;margin-top:14px;gap:16px;">
                    <div style="flex:1;min-width:130px;">
                      <div style="font-size:0.78rem;color:#94a3b8;text-transform:uppercase;">Predicted</div>
                      <div style="font-size:2rem;font-weight:800;color:#60a5fa;">{pred:.2f} <span style="font-size:0.9rem;color:#cbd5e1;">{cfg['unit']}</span></div>
                    </div>
                    <div style="flex:1;min-width:160px;">
                      <div style="font-size:0.78rem;color:#94a3b8;text-transform:uppercase;">90% Conformal Interval</div>
                      <div style="font-size:1.15rem;font-weight:700;color:#38bdf8;margin-top:4px;">{conf_text}</div>
                      <div style="font-size:0.75rem;color:#94a3b8;">{width_text} · {unc.get("observed_coverage","90.0%")}</div>
                    </div>
                    <div style="flex:1;min-width:120px;">
                      <div style="font-size:0.78rem;color:#94a3b8;text-transform:uppercase;">Threshold</div>
                      <div style="font-size:1.15rem;font-weight:700;color:#cbd5e1;margin-top:4px;">{thr:.1f} {cfg['unit']}</div>
                      <div style="font-size:0.75rem;color:#94a3b8;">{dist_str}</div>
                    </div>
                  </div>
                  <div style="margin-top:14px;padding:12px 16px;border-radius:10px;
                       background:{badge['bg']};border-left:4px solid {badge['border']};">
                    <div style="font-size:0.9rem;font-weight:700;color:{badge['text']};">{badge['label']}</div>
                    <div style="font-size:0.83rem;color:{badge['text']};opacity:0.9;margin-top:4px;">{dec['description']}</div>
                  </div>
                </div>
                """), unsafe_allow_html=True)

                # Threshold chart (responsive)
                max_x  = max(thr * 1.4, (unc["upper_bound"] or pred) * 1.25, pred * 1.3)
                fig    = go.Figure()
                if unc["available"]:
                    fig.add_trace(go.Scatter(x=[unc["lower_bound"], unc["upper_bound"]], y=[0,0],
                        mode="lines", line=dict(color="#3b82f6", width=8), name="90% Interval",
                        hovertext=f"[{unc['lower_bound']:.2f} – {unc['upper_bound']:.2f}] {cfg['unit']}"))
                fig.add_trace(go.Scatter(x=[pred], y=[0], mode="markers+text",
                    marker=dict(color="#1e40af", size=16, line=dict(color="#fff", width=2)),
                    text=[f"{pred:.2f}"], textposition="top center", name="Prediction"))
                fig.add_shape(type="line", x0=thr, y0=-0.3, x1=thr, y1=0.3,
                    line=dict(color="#dc2626", width=3, dash="dash"))
                fig.add_annotation(x=thr, y=0.38, text=f"THRESHOLD: {thr} {cfg['unit']}",
                    showarrow=False, font=dict(color="#dc2626", size=11),
                    bgcolor="#fef2f2", bordercolor="#f87171", borderwidth=1, borderpad=3)
                fig.update_layout(
                    height=175, showlegend=False,
                    margin=dict(l=10, r=10, t=35, b=25),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0"),
                    xaxis=dict(range=[0, max_x], title=f"Concentration ({cfg['unit']})",
                               gridcolor="rgba(148,163,184,0.15)", tickfont=dict(color="#94a3b8")),
                    yaxis=dict(showticklabels=False, range=[-0.5, 0.5], showgrid=False),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)

            # — Scientific interpretation (from config, not hardcoded) —
            st.markdown("### 💡 Scientific Interpretation")
            ic1, ic2 = st.columns(2)
            for col_w, mkey in [(ic1, "ni"), (ic2, "cd")]:
                c = MODEL_CONFIG[mkey]
                col_w.markdown(
                    f'<div style="background:rgba(30,41,59,0.5);border:1px solid rgba(148,163,184,0.2);'
                    f'border-radius:10px;padding:14px;">'
                    f'<div style="font-weight:700;color:#f8fafc;">{c["name"]} Model</div>'
                    f'<div style="font-size:0.83rem;color:#cbd5e1;margin-top:6px;">'
                    f'<strong style="color:#60a5fa;">Primary predictor:</strong> {c["primary_predictor"]}<br>'
                    f'<em>{c["interpretation"]}</em></div></div>',
                    unsafe_allow_html=True
                )
            st.caption("Predictive variables represent empirical statistical association — not physical causation.")

            # — Model validation expander (from config, no hardcoding) —
            with st.expander("ℹ️ MODEL VALIDATION METRICS", expanded=False):
                vc1, vc2 = st.columns(2)
                for col_w, mkey in [(vc1, "ni"), (vc2, "cd")]:
                    c = MODEL_CONFIG[mkey]
                    col_w.markdown(
                        f'<div style="background:rgba(15,23,42,0.6);border:1px solid rgba(148,163,184,0.2);'
                        f'border-radius:8px;padding:12px;font-size:0.83rem;color:#f8fafc;">'
                        f'<strong style="color:#60a5fa;">{c["name"]}</strong><br><br>'
                        f'• Algorithm: <code style="color:#38bdf8;">{c["model_type"]}</code><br>'
                        f'• Pipeline: <code style="color:#38bdf8;">{c["pipeline_description"]}</code><br>'
                        f'• 5×5 CV OOF R²: <code>{c["oof_r2_primary"]:.4f}</code><br>'
                        f'• Global OOF R²: <code>{c["oof_r2_secondary"]:.4f}</code><br>'
                        f'• OOF RMSE: <code>{c["oof_rmse"]:.4f} {c["unit"]}</code><br>'
                        f'• Spearman r: <code>{c["spearman"]:.4f}</code><br>'
                        f'• Conformal: <em style="color:#fbbf24;">{c["performance_note"]}</em>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                st.warning("Research validation metrics — do not guarantee accuracy for individual samples.")

            # — Overall recommendation —
            st.markdown(
                f'<div class="rec-box">'
                f'<div style="font-size:0.9rem;font-weight:700;color:#60a5fa;text-transform:uppercase;">Recommended Screening Action</div>'
                f'<div style="font-size:1rem;font-weight:600;color:#f8fafc;margin-top:6px;line-height:1.5;">{res["overall_recommendation"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            # — Download report —
            st.markdown("### 📄 Export Report")
            from datetime import datetime, timezone
            from components.report_generator import generate_html_report
            ts  = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            sid = f"GW-{ts}"
            html_content = generate_html_report(
                res["input_parameters"], res["results"],
                res["overall_recommendation"], domain_state=domain, sample_id=sid
            )
            st.download_button(
                label="📥 DOWNLOAD SCREENING REPORT (HTML)",
                data=html_content,
                file_name=f"Groundwater_Screening_{sid}.html",
                mime="text/html",
                type="secondary",
            )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="footer">⚠️ <strong>RESEARCH DISCLAIMER:</strong> {DISCLAIMER_TEXT}</div>',
    unsafe_allow_html=True
)
