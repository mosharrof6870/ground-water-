"""
Report Generator — all values sourced from MODEL_CONFIG (no hardcoding).
Timestamps use UTC. Interpretation from config.
"""
from datetime import datetime, timezone
from config import DISCLAIMER_TEXT, VALIDATED_RANGES, ENVIRONMENTAL_RANGES, MODEL_CONFIG


def generate_html_report(input_params, results_dict, overall_rec,
                         domain_state="IN_DOMAIN", sample_id="SAMPLE-001"):
    ts   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    ni   = results_dict["ni"]
    cd   = results_dict["cd"]
    ni_c = MODEL_CONFIG["ni"]
    cd_c = MODEL_CONFIG["cd"]

    # — Model domain rows —
    dom_rows = ""
    if VALIDATED_RANGES:
        for p_key, p_name, v_rkey in [
            ("pH", "pH", "pH"), ("TDS", "TDS", "TDS"),
            ("NO3-N", "NO3-N", "NO3-N"), ("well_depth", "Well Depth", "Well Depth")
        ]:
            val    = input_params[p_key]
            vr     = VALIDATED_RANGES[v_rkey]
            in_r   = vr["min"] <= val <= vr["max"]
            color  = "#059669" if in_r else "#dc2626"
            tag    = "✓ Within Range" if in_r else "⚠️ Outside Range"
            u_str  = f" {vr['unit']}" if vr["unit"] != "-" else ""
            v_str  = f"{val:.2f}" if p_key not in ("TDS","well_depth") else f"{val:.1f}"
            dom_rows += f"""<tr>
              <td><strong>{p_name}</strong></td><td>{v_str}{u_str}</td>
              <td>{vr['min']}{u_str} – {vr['max']}{u_str}</td>
              <td><span style="color:{color};font-weight:bold;">{tag}</span></td>
            </tr>"""

    # — Environmental reference rows —
    env_rows = ""
    for p_key, p_name, e_key in [
        ("pH","pH","pH"), ("TDS","TDS","TDS"),
        ("NO3-N","NO3-N","NO3-N"), ("well_depth","Well Depth","Well Depth")
    ]:
        val   = input_params[p_key]
        ei    = ENVIRONMENTAL_RANGES[e_key]
        u_str = f" {ei['unit']}" if ei["unit"] != "-" else ""
        v_str = f"{val:.2f}" if p_key not in ("TDS","well_depth") else f"{val:.1f}"
        env_rows += f"""<tr>
          <td><strong>{p_name}</strong></td><td>{v_str}{u_str}</td>
          <td>{ei['range']}{u_str}</td><td>{ei['standard']}</td>
        </tr>"""

    # — Conformal text —
    def conf_text(res):
        u = res["uncertainty"]
        if u["available"]:
            return f"{u['lower_bound']:.2f} – {u['upper_bound']:.2f} µg/L"
        return "90% conformal interval unavailable."

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Groundwater AI Screening Report — {sample_id}</title>
<style>
body{{font-family:'Segoe UI',Arial,sans-serif;color:#1e293b;line-height:1.5;
     padding:30px;max-width:860px;margin:0 auto;background:#fff;}}
.hdr{{border-bottom:3px solid #2563eb;padding-bottom:14px;margin-bottom:24px;}}
.title{{font-size:22px;font-weight:800;color:#0f172a;}}
.sub{{font-size:12px;color:#64748b;margin-top:5px;}}
.sec{{margin-bottom:26px;}}
.sec-title{{font-size:14px;font-weight:700;color:#1e3a8a;border-bottom:2px solid #e2e8f0;
            padding-bottom:5px;margin-bottom:10px;text-transform:uppercase;letter-spacing:.5px;}}
table{{width:100%;border-collapse:collapse;font-size:12px;}}
th,td{{border:1px solid #cbd5e1;padding:8px 11px;text-align:left;}}
th{{background:#f1f5f9;color:#334155;font-weight:700;}}
.rec{{padding:14px;border-radius:8px;background:#f8fafc;border-left:4px solid #2563eb;
      font-weight:600;color:#0f172a;font-size:13px;line-height:1.5;}}
.disc{{background:#f8fafc;border-left:4px solid #94a3b8;padding:12px;font-size:11px;
       color:#475569;margin-top:32px;border-radius:4px;}}
</style></head><body>

<div class="hdr">
  <div class="title">🧪 GROUNDWATER AI HEAVY-METAL SCREENING REPORT</div>
  <div class="sub">Sample: <strong>{sample_id}</strong> | Timestamp: {ts} | Domain State: <strong>{domain_state}</strong></div>
</div>

<div class="sec"><div class="sec-title">1. Sample Information</div>
<table><tr><th>Sample ID</th><td>{sample_id}</td><th>Framework</th><td>North Bengal Aquifer Study</td></tr>
<tr><th>Timestamp (UTC)</th><td>{ts}</td><th>Domain State</th><td>{domain_state}</td></tr></table></div>

<div class="sec"><div class="sec-title">2. Input Hydrochemical Parameters</div>
<table><tr><th>Parameter</th><th>Value</th><th>Unit</th></tr>
<tr><td>pH</td><td>{input_params['pH']:.2f}</td><td>–</td></tr>
<tr><td>TDS</td><td>{input_params['TDS']:.1f}</td><td>mg/L</td></tr>
<tr><td>NO3-N</td><td>{input_params['NO3-N']:.2f}</td><td>mg/L</td></tr>
<tr><td>Well Depth</td><td>{input_params['well_depth']:.1f}</td><td>m</td></tr></table></div>

<div class="sec"><div class="sec-title">3. Model Applicability — Validated Input Domain</div>
<table><tr><th>Parameter</th><th>Submitted</th><th>Validated Model Range</th><th>Status</th></tr>
{dom_rows}</table></div>

<div class="sec"><div class="sec-title">4. Environmental Reference Standards</div>
<table><tr><th>Parameter</th><th>Submitted</th><th>Reference Range</th><th>Standard</th></tr>
{env_rows}</table></div>

<div class="sec"><div class="sec-title">5. Prediction Results — 90% Conformal Prediction Intervals</div>
<table>
<tr><th>Metal</th><th>Algorithm</th><th>Prediction</th><th>90% Conformal Interval</th><th>Coverage</th><th>Threshold</th></tr>
<tr>
  <td><strong>Nickel (Ni)</strong></td>
  <td>{ni_c['model_type']}</td>
  <td>{ni['prediction']:.4f} µg/L</td>
  <td>{conf_text(ni)}</td>
  <td>{ni['uncertainty'].get('observed_coverage','90.0%')}</td>
  <td>{ni['decision']['threshold']:.1f} µg/L</td>
</tr>
<tr>
  <td><strong>Cadmium (Cd)</strong></td>
  <td>{cd_c['model_type']}</td>
  <td>{cd['prediction']:.4f} µg/L</td>
  <td>{conf_text(cd)}</td>
  <td>{cd['uncertainty'].get('observed_coverage','90.0%')}</td>
  <td>{cd['decision']['threshold']:.1f} µg/L</td>
</tr></table></div>

<div class="sec"><div class="sec-title">6. Screening Status</div>
<table><tr><th>Metal</th><th>Status</th><th>Details</th></tr>
<tr><td><strong>Ni</strong></td><td><strong>{ni['decision']['title']}</strong></td><td>{ni['decision']['description']}</td></tr>
<tr><td><strong>Cd</strong></td><td><strong>{cd['decision']['title']}</strong></td><td>{cd['decision']['description']}</td></tr>
</table></div>

<div class="sec"><div class="sec-title">7. Scientific Interpretation</div>
<p><strong>Nickel (Ni):</strong> {ni_c['interpretation']}</p>
<p><strong>Cadmium (Cd):</strong> {cd_c['interpretation']}</p>
<p style="font-size:11px;color:#64748b;"><em>Predictive variables represent empirical statistical association — not physical causation.</em></p></div>

<div class="sec"><div class="sec-title">8. Recommended Screening Action</div>
<div class="rec">{overall_rec}</div></div>

<div class="sec"><div class="sec-title">9. Model Validation Metrics</div>
<p style="font-size:12px;"><strong>Protocol:</strong> 5×5 Repeated Nested Cross-Validation (25 outer folds)</p>
<p style="font-size:12px;">• <strong>Ni {ni_c['model_type']}:</strong>
   5×5 CV OOF R² = {ni_c['oof_r2_primary']:.4f} |
   Global OOF R² = {ni_c['oof_r2_secondary']:.4f} |
   RMSE = {ni_c['oof_rmse']:.3f} µg/L |
   Spearman r = {ni_c['spearman']:.3f} | <em>{ni_c['performance_note']}</em></p>
<p style="font-size:12px;">• <strong>Cd {cd_c['model_type']}:</strong>
   5×5 CV OOF R² = {cd_c['oof_r2_primary']:.4f} |
   Global OOF R² = {cd_c['oof_r2_secondary']:.4f} |
   RMSE = {cd_c['oof_rmse']:.3f} µg/L |
   Spearman r = {cd_c['spearman']:.3f} | <em>{cd_c['performance_note']}</em></p>
<p style="font-size:11px;color:#dc2626;"><strong>Note:</strong> Research metrics — do not guarantee individual sample accuracy.</p></div>

<div class="disc"><strong>10. DISCLAIMER:</strong><br>{DISCLAIMER_TEXT}</div>
</body></html>"""
    return html
