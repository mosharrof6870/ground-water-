"""
Interpretation Panel Component.
Displays non-causal scientific feature interpretation and collapsible research model validation information.
"""
import textwrap
import streamlit as st
from config import MODEL_CONFIG


def render_interpretation_panel(results_dict):
    """
    Renders scientific interpretation and research model metadata panel.
    """
    st.markdown("### 💡 Scientific Interpretation")

    col1, col2 = st.columns(2)

    with col1:
        html1 = textwrap.dedent(f"""
<div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 10px; padding: 16px;">
    <div style="font-weight: 700; color: #f8fafc; font-size: 0.95rem;">Nickel (Ni) Model Factor</div>
    <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 6px; line-height: 1.4;">
        <strong style="color: #60a5fa;">Primary Predictive Variable:</strong> TDS (mg/L)<br>
        <em style="color: #cbd5e1;">{MODEL_CONFIG['ni']['interpretation']}</em>
    </div>
</div>
""")
        st.markdown(html1, unsafe_allow_html=True)

    with col2:
        html2 = textwrap.dedent(f"""
<div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 10px; padding: 16px;">
    <div style="font-weight: 700; color: #f8fafc; font-size: 0.95rem;">Cadmium (Cd) Model Factor</div>
    <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 6px; line-height: 1.4;">
        <strong style="color: #60a5fa;">Primary Predictive Variable:</strong> NO3-N (mg/L)<br>
        <em style="color: #cbd5e1;">{MODEL_CONFIG['cd']['interpretation']}</em>
    </div>
</div>
""")
        st.markdown(html2, unsafe_allow_html=True)

    st.caption("Note: Variables represent empirical statistical predictors in the trained models and do not imply direct physical causation.")

    # Collapsible Model Validation Information
    with st.expander("ℹ️ MODEL VALIDATION INFORMATION", expanded=False):
        exp_html = textwrap.dedent("""
<div style="font-size: 0.85rem; color: #cbd5e1; margin-bottom: 14px;">
    <p><strong>Validation Protocol:</strong> 5×5 Repeated Nested Cross-Validation (25 outer folds × 5 inner folds)</p>
    <p><strong>Training Dataset:</strong> N = 40 real groundwater samples from North Bengal aquifers</p>
</div>
""")
        st.markdown(exp_html, unsafe_allow_html=True)

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            cfg_ni = MODEL_CONFIG["ni"]
            ni_oof_r2 = cfg_ni.get("oof_r2", cfg_ni.get("oof_r2_primary", 0.224))
            ni_meta = textwrap.dedent(f"""
<div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 8px; padding: 14px; font-size: 0.85rem; color: #f8fafc;">
    <strong style="color: #60a5fa;">{cfg_ni['name']} ({cfg_ni['symbol']}) Model Specs</strong><br><br>
    • Algorithm: <code style="color: #38bdf8;">{cfg_ni['model_type']}</code><br>
    • Out-Of-Fold R²: <code>{ni_oof_r2:.3f}</code><br>
    • Out-Of-Fold RMSE: <code>{cfg_ni['oof_rmse']:.3f} µg/L</code><br>
    • Spearman Correlation: <code>{cfg_ni['spearman']:.3f}</code><br>
    • Interpretation: <em style="color: #fbbf24;">{cfg_ni['performance_note']}</em>
</div>
""")
            st.markdown(ni_meta, unsafe_allow_html=True)

        with m_col2:
            cfg_cd = MODEL_CONFIG["cd"]
            cd_oof_r2 = cfg_cd.get("oof_r2", cfg_cd.get("oof_r2_primary", 0.706))
            cd_meta = textwrap.dedent(f"""
<div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 8px; padding: 14px; font-size: 0.85rem; color: #f8fafc;">
    <strong style="color: #60a5fa;">{cfg_cd['name']} ({cfg_cd['symbol']}) Model Specs</strong><br><br>
    • Algorithm: <code style="color: #38bdf8;">{cfg_cd['model_type']}</code><br>
    • Out-Of-Fold R²: <code>{cd_oof_r2:.3f}</code><br>
    • Out-Of-Fold RMSE: <code>{cfg_cd['oof_rmse']:.3f} µg/L</code><br>
    • Spearman Correlation: <code>{cfg_cd['spearman']:.3f}</code><br>
    • Interpretation: <em style="color: #34d399;">{cfg_cd['performance_note']}</em>
</div>
""")
            st.markdown(cd_meta, unsafe_allow_html=True)

        st.warning("These are research validation metrics and do not guarantee prediction accuracy for an individual new sample.")
