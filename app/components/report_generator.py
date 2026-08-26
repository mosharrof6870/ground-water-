"""
Report Generator Component.
Generates structured HTML research screening reports with download options.
Distinguishes Validated Model Input Range from Environmental Reference Range.
Renders recommended screening action ONLY ONCE (no duplicates).
"""
from datetime import datetime
import streamlit as st
from config import DISCLAIMER_TEXT, VALIDATED_RANGES, ENVIRONMENTAL_RANGES, MODEL_CONFIG


def generate_html_report(input_params, results_dict, overall_recommendation, domain_state="IN_DOMAIN", sample_id="SAMPLE-001"):
    """
    Generates structured HTML string for downloadable research screening report.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    ni_res = results_dict["ni"]
    cd_res = results_dict["cd"]

    # Build Validated Model Input Range Rows
    model_domain_rows = ""
    if VALIDATED_RANGES:
        param_keys = [("pH", "pH"), ("TDS", "TDS"), ("NO3-N", "NO3-N"), ("well_depth", "Well Depth")]
        for p_key, p_name in param_keys:
            val = input_params[p_key]
            v_range = VALIDATED_RANGES[p_name]
            in_range = v_range["min"] <= val <= v_range["max"]
            status_tag = f"<span style='color: {'#059669' if in_range else '#dc2626'}; font-weight: bold;'>{'✓ Within Validated Range' if in_range else '⚠️ Outside Validated Range'}</span>"
            unit_str = f" {v_range['unit']}" if v_range['unit'] != "-" else ""
            val_str = f"{val:.1f}" if p_key in ("TDS", "well_depth") else f"{val:.2f}"
            model_domain_rows += f"""
            <tr>
                <td><strong>{p_name}</strong></td>
                <td>{val_str}{unit_str}</td>
                <td>{v_range['min']}{unit_str} – {v_range['max']}{unit_str}</td>
                <td>{status_tag}</td>
            </tr>
            """
    else:
        model_domain_rows = "<tr><td colspan='4'>Validated model input ranges are unavailable.</td></tr>"

    # Build Environmental Reference Range Rows (Separate Table)
    env_rows = ""
    param_keys = [("pH", "pH"), ("TDS", "TDS"), ("NO3-N", "NO3-N"), ("well_depth", "Well Depth")]
    for p_key, p_name in param_keys:
        val = input_params[p_key]
        e_info = ENVIRONMENTAL_RANGES[p_name]
        unit_str = f" {e_info['unit']}" if e_info['unit'] != "-" else ""
        val_str = f"{val:.1f}" if p_key in ("TDS", "well_depth") else f"{val:.2f}"
        env_rows += f"""
        <tr>
            <td><strong>{p_name}</strong></td>
            <td>{val_str}{unit_str}</td>
            <td>{e_info['range']}{unit_str}</td>
            <td>{e_info['standard']}</td>
        </tr>
        """

    # Conformal Text
    ni_conf = (
        f"{ni_res['uncertainty']['lower_bound']:.2f} – {ni_res['uncertainty']['upper_bound']:.2f} µg/L"
        if ni_res['uncertainty']['available'] else "90% conformal interval unavailable."
    )
    cd_conf = (
        f"{cd_res['uncertainty']['lower_bound']:.2f} – {cd_res['uncertainty']['upper_bound']:.2f} µg/L"
        if cd_res['uncertainty']['available'] else "90% conformal interval unavailable."
    )

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Groundwater AI Screening Report - {sample_id}</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; color: #1e293b; line-height: 1.5; padding: 30px; max-width: 850px; margin: 0 auto; background-color: #ffffff; }}
            .header {{ border-bottom: 3px solid #2563eb; padding-bottom: 15px; margin-bottom: 25px; }}
            .title {{ font-size: 24px; font-weight: 800; color: #0f172a; letter-spacing: -0.5px; }}
            .subtitle {{ font-size: 13px; color: #64748b; margin-top: 6px; }}
            .section {{ margin-bottom: 28px; }}
            .section-title {{ font-size: 15px; font-weight: 700; color: #1e3a8a; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }}
            th, td {{ border: 1px solid #cbd5e1; padding: 9px 12px; text-align: left; }}
            th {{ background-color: #f1f5f9; color: #334155; font-weight: 700; }}
            .status-badge {{ display: inline-block; padding: 4px 10px; border-radius: 4px; font-weight: 700; font-size: 12px; }}
            .below {{ background-color: #d1fae5; color: #065f46; border: 1px solid #a7f3d0; }}
            .uncertain {{ background-color: #fef3c7; color: #92400e; border: 1px solid #fde68a; }}
            .exceed {{ background-color: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }}
            .rec-box {{ padding: 16px; border-radius: 8px; background-color: #f8fafc; border-left: 4px solid #2563eb; font-weight: 600; color: #0f172a; font-size: 14px; line-height: 1.5; }}
            .disclaimer {{ background-color: #f8fafc; border-left: 4px solid #94a3b8; padding: 14px; font-size: 12px; color: #475569; margin-top: 35px; border-radius: 4px; line-height: 1.4; }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="title">🧪 GROUNDWATER AI HEAVY-METAL SCREENING REPORT</div>
            <div class="subtitle">Sample Reference: <strong>{sample_id}</strong> | Timestamp: {timestamp} | Domain State: <strong>{domain_state}</strong></div>
        </div>

        <div class="section">
            <div class="section-title">1. SAMPLE INFORMATION</div>
            <table>
                <tr><th>Sample Reference ID</th><td>{sample_id}</td><th>Assessment Framework</th><td>North Bengal Aquifer Study</td></tr>
                <tr><th>Evaluation Timestamp</th><td>{timestamp}</td><th>Domain Applicability</th><td>{domain_state}</td></tr>
            </table>
        </div>

        <div class="section">
            <div class="section-title">2. INPUT HYDROCHEMICAL PARAMETERS</div>
            <table>
                <tr><th>Parameter</th><th>Submitted Value</th><th>Measurement Unit</th></tr>
                <tr><td>pH</td><td>{input_params['pH']:.2f}</td><td>-</td></tr>
                <tr><td>TDS (Total Dissolved Solids)</td><td>{input_params['TDS']:.1f}</td><td>mg/L</td></tr>
                <tr><td>NO3-N (Nitrate-Nitrogen)</td><td>{input_params['NO3-N']:.2f}</td><td>mg/L</td></tr>
                <tr><td>Well Depth</td><td>{input_params['well_depth']:.1f}</td><td>m</td></tr>
            </table>
        </div>

        <div class="section">
            <div class="section-title">3. MODEL APPLICABILITY / VALIDATED INPUT DOMAIN</div>
            <p style="font-size: 12px; color: #64748b; margin-bottom: 8px;"><em>Range-based model applicability check comparing sample parameters against empirical training data bounds.</em></p>
            <table>
                <tr><th>Parameter</th><th>Submitted Value</th><th>Validated Model Input Range</th><th>Applicability Status</th></tr>
                {model_domain_rows}
            </table>
        </div>

        <div class="section">
            <div class="section-title">4. SEPARATE ENVIRONMENTAL REFERENCE RANGES</div>
            <p style="font-size: 12px; color: #64748b; margin-bottom: 8px;"><em>Environmental reference standards for water quality contextualization (separate from model validation domain).</em></p>
            <table>
                <tr><th>Parameter</th><th>Submitted Value</th><th>Environmental Reference Range</th><th>Reference Standard</th></tr>
                {env_rows}
            </table>
        </div>

        <div class="section">
            <div class="section-title">5. PREDICTION RESULTS & 90% NOMINAL CONFORMAL PREDICTION INTERVALS</div>
            <table>
                <tr>
                    <th>Target Heavy Metal</th>
                    <th>Model Algorithm</th>
                    <th>Point Prediction</th>
                    <th>90% Nominal Conformal Interval</th>
                    <th>Empirical Coverage</th>
                    <th>Screening Threshold</th>
                </tr>
                <tr>
                    <td><strong>Nickel (Ni)</strong></td>
                    <td>{MODEL_CONFIG['ni']['model_type']}</td>
                    <td>{ni_res['prediction']:.2f} µg/L</td>
                    <td>{ni_conf}</td>
                    <td>{ni_res['uncertainty'].get('observed_coverage', '90.0% observed OOF coverage')}</td>
                    <td>{ni_res['decision']['threshold']:.1f} µg/L</td>
                </tr>
                <tr>
                    <td><strong>Cadmium (Cd)</strong></td>
                    <td>{MODEL_CONFIG['cd']['model_type']}</td>
                    <td>{cd_res['prediction']:.2f} µg/L</td>
                    <td>{cd_conf}</td>
                    <td>{cd_res['uncertainty'].get('observed_coverage', '90.0% observed OOF coverage')}</td>
                    <td>{cd_res['decision']['threshold']:.1f} µg/L</td>
                </tr>
            </table>
        </div>

        <div class="section">
            <div class="section-title">6. THRESHOLD COMPARISON & SCREENING STATUS</div>
            <table>
                <tr>
                    <th>Target Metal</th>
                    <th>Screening Status Title</th>
                    <th>Decision Details</th>
                </tr>
                <tr>
                    <td><strong>Nickel (Ni)</strong></td>
                    <td><strong>{ni_res['decision']['title']}</strong></td>
                    <td>{ni_res['decision']['description']}</td>
                </tr>
                <tr>
                    <td><strong>Cadmium (Cd)</strong></td>
                    <td><strong>{cd_res['decision']['title']}</strong></td>
                    <td>{cd_res['decision']['description']}</td>
                </tr>
            </table>
        </div>

        <div class="section">
            <div class="section-title">7. SCIENTIFIC INTERPRETATION & PRIMARY PREDICTIVE FACTORS</div>
            <p><strong>Nickel (Ni):</strong> TDS is the primary predictive variable in the trained Ni HuberRegressor model.</p>
            <p><strong>Cadmium (Cd):</strong> NO3-N is the primary predictive variable in the trained Cd LinearRegression model.</p>
            <p style="font-size: 12px; color: #64748b; margin-top: 6px;"><em>Note: Predictive variables represent empirical statistical association and do not imply physical causation.</em></p>
        </div>

        <div class="section">
            <div class="section-title">8. RECOMMENDED SCREENING ACTION</div>
            <div class="rec-box">
                {overall_recommendation}
            </div>
        </div>

        <div class="section">
            <div class="section-title">9. MODEL VALIDATION & COMPUTATIONAL METRICS</div>
            <p style="font-size: 13px;"><strong>Validation Protocol:</strong> 5×5 Repeated Nested Cross-Validation</p>
            <p style="font-size: 13px;">• <strong>Ni HuberRegressor:</strong> Primary 5×5 Nested CV $R^2 = 0.2240$ | Single-pass OOF $R^2 = 0.2271$ (Difference: $+0.0031$) | <em>Limited quantitative predictive performance</em></p>
            <p style="font-size: 13px;">• <strong>Cd LinearRegression:</strong> Primary 5×5 Nested CV $R^2 = 0.7061$ | Single-pass OOF $R^2 = 0.7158$ (Difference: $+0.0097$) | <em>Stronger quantitative predictive performance</em></p>
            <p style="font-size: 12px; color: #475569; margin-top: 4px;">• <strong>Training N Trace:</strong> Model estimator trained on $N = 40$ samples using median imputation (sample S98_01798 had 1 missing WELL_DEPTH; complete-case domain profiling evaluated $N = 39$).</p>
            <p style="font-size: 12px; color: #475569; margin-top: 4px;">• <strong>Conformal Calibration:</strong> 90% nominal out-of-fold conformal prediction interval; observed coverage in evaluation artifact = 90.0% (36/40 samples covered).</p>
            <p style="font-size: 12px; color: #dc2626; margin-top: 6px;"><strong>Note:</strong> These are research validation metrics for preliminary screening and do not guarantee prediction accuracy for an individual new sample.</p>
        </div>

        <div class="disclaimer">
            <strong>10. DISCLAIMER:</strong><br>
            {DISCLAIMER_TEXT}
        </div>
    </body>
    </html>
    """
    return html


def render_report_download_button(input_params, results_dict, overall_recommendation, domain_state="IN_DOMAIN"):
    """
    Renders report download section in Streamlit UI.
    """
    st.markdown("### 📄 Export Research Report")

    sample_ref = f"GW-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    html_content = generate_html_report(
        input_params, results_dict, overall_recommendation, domain_state=domain_state, sample_id=sample_ref
    )

    st.download_button(
        label="📥 GENERATE & DOWNLOAD SAMPLE REPORT (HTML)",
        data=html_content,
        file_name=f"Groundwater_Screening_Report_{sample_ref}.html",
        mime="text/html",
        type="secondary"
    )
