"""
Result Card Component.
Displays metal predictions, conformal prediction intervals, screening badges, and threshold distances.
"""
import textwrap
import streamlit as st
from utils.formatting import get_status_badge_style, format_threshold_distance


def render_result_card(metal_res):
    """
    Renders screening result card for a single heavy metal target.
    """
    cfg = metal_res["config"]
    pred = metal_res["prediction"]
    uncertainty = metal_res["uncertainty"]
    decision = metal_res["decision"]

    symbol = cfg["symbol"]
    name = cfg["name"]
    unit = cfg["unit"]
    threshold = decision["threshold"]
    distance_str = format_threshold_distance(pred, threshold, unit)

    badge_style = get_status_badge_style(decision["status_code"])
    conf_label = decision["confidence_label"]

    conformal_text = (
        f"{uncertainty['lower_bound']:.2f} – {uncertainty['upper_bound']:.2f} {unit}"
        if uncertainty['available'] else "Unavailable"
    )
    width_text = (
        f"Width: ±{uncertainty['quantile_q']:.2f} {unit}"
        if uncertainty['available'] else "Interval unavailable"
    )

    card_html = textwrap.dedent(f"""
<div style="border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 14px; padding: 22px; background: rgba(30, 41, 59, 0.5); backdrop-filter: blur(10px); box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2); margin-bottom: 20px;">
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(148, 163, 184, 0.18); padding-bottom: 14px;">
        <div>
            <span style="font-size: 1.35rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.3px;">{name} ({symbol})</span>
            <span style="font-size: 0.82rem; color: #94a3b8; margin-left: 8px; font-weight: 500;">Target</span>
        </div>
        <div style="background-color: {badge_style['bg_color']}; color: {badge_style['text_color']}; border: 1px solid {badge_style['border_color']}; padding: 5px 14px; border-radius: 20px; font-size: 0.78rem; font-weight: 700;">
            {conf_label}
        </div>
    </div>
    <div style="display: flex; flex-wrap: wrap; margin-top: 18px; gap: 18px;">
        <div style="flex: 1; min-width: 140px;">
            <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;">Predicted Concentration</div>
            <div style="font-size: 2.1rem; font-weight: 800; color: #60a5fa; line-height: 1.2; margin-top: 4px;">
                {pred:.2f} <span style="font-size: 1rem; font-weight: 600; color: #cbd5e1;">{unit}</span>
            </div>
        </div>
        <div style="flex: 1; min-width: 170px;">
            <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;">90% Nominal Conformal Interval</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #38bdf8; margin-top: 6px;">
                {conformal_text}
            </div>
            <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 2px;">
                {width_text} ({uncertainty.get('observed_coverage', '90.0% observed OOF coverage')})
            </div>
        </div>
        <div style="flex: 1; min-width: 130px;">
            <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;">Regulatory Benchmark</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #cbd5e1; margin-top: 6px;">
                {threshold:.1f} {unit}
            </div>
            <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 2px;">
                {distance_str}
            </div>
        </div>
    </div>
    <div style="margin-top: 18px; padding: 14px 18px; border-radius: 10px; background-color: {badge_style['bg_color']}; border-left: 4px solid {badge_style['border_color']};">
        <div style="font-size: 0.92rem; font-weight: 700; color: {badge_style['text_color']};">
            {badge_style['label']}
        </div>
        <div style="font-size: 0.85rem; color: {badge_style['text_color']}; opacity: 0.9; margin-top: 4px; line-height: 1.4;">
            {decision['description']}
        </div>
    </div>
</div>
""")

    st.markdown(card_html, unsafe_allow_html=True)
