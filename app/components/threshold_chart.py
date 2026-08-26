"""
Threshold Chart Component.
Generates an intuitive horizontal threshold chart showing predicted concentration,
conformal prediction intervals, and the regulatory threshold benchmark.
"""
import plotly.graph_objects as go
import streamlit as st


def render_threshold_chart(metal_res):
    """
    Renders horizontal range chart for a single heavy metal prediction.
    
    Args:
        metal_res (dict): Result entry from PredictionService
    """
    cfg = metal_res["config"]
    pred = metal_res["prediction"]
    uncertainty = metal_res["uncertainty"]
    decision = metal_res["decision"]

    symbol = cfg["symbol"]
    name = cfg["name"]
    unit = cfg["unit"]
    threshold = decision["threshold"]

    # Determine axis max
    if uncertainty["available"]:
        upper_b = uncertainty["upper_bound"]
        lower_b = uncertainty["lower_bound"]
    else:
        upper_b = pred
        lower_b = pred

    max_val = max(threshold * 1.4, upper_b * 1.25, pred * 1.3)

    fig = go.Figure()

    # 1. Uncertainty interval bar
    if uncertainty["available"]:
        fig.add_trace(go.Scatter(
            x=[lower_b, upper_b],
            y=[0, 0],
            mode="lines",
            line=dict(color="#3b82f6", width=8),
            name="90% Conformal Interval",
            hoverinfo="text",
            hovertext=f"90% Interval: [{lower_b:.2f} – {upper_b:.2f}] {unit}"
        ))

    # 2. Predicted Point dot
    fig.add_trace(go.Scatter(
        x=[pred],
        y=[0],
        mode="markers+text",
        marker=dict(color="#1e40af", size=16, symbol="circle", line=dict(color="#ffffff", width=2)),
        text=[f"Prediction: {pred:.2f}"],
        textposition="top center",
        name="Predicted Value",
        hoverinfo="text",
        hovertext=f"Predicted {symbol}: {pred:.2f} {unit}"
    ))

    # 3. Threshold line
    fig.add_shape(
        type="line",
        x0=threshold,
        y0=-0.3,
        x1=threshold,
        y1=0.3,
        line=dict(color="#dc2626", width=3, dash="dash")
    )

    fig.add_annotation(
        x=threshold,
        y=0.35,
        text=f"THRESHOLD: {threshold} {unit}",
        showarrow=False,
        font=dict(color="#dc2626", size=12, family="sans-serif"),
        bgcolor="#fef2f2",
        bordercolor="#f87171",
        borderwidth=1,
        borderpad=4
    )

    fig.update_layout(
        font=dict(family="Inter, Outfit, sans-serif", color="#e2e8f0"),
        title=dict(
            text=f"Horizontal Threshold Comparison ({name})",
            font=dict(size=14, color="#f8fafc", family="Outfit, Inter, sans-serif")
        ),
        xaxis=dict(
            title=dict(text=f"Concentration ({unit})", font=dict(color="#cbd5e1")),
            tickfont=dict(color="#94a3b8"),
            range=[0, max_val],
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.15)",
            zeroline=True,
            zerolinecolor="rgba(148, 163, 184, 0.3)"
        ),
        yaxis=dict(
            showticklabels=False,
            range=[-0.5, 0.5],
            showgrid=False
        ),
        height=185,
        margin=dict(l=20, r=20, t=40, b=35),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig)
