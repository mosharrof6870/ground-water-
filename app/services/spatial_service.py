"""
Spatial Service — Geospatial Intelligence for North Bengal Groundwater.
Light-themed interactive GIS layer for 40 North Bengal monitoring well stations.
"""

import json
import os
import math
from typing import List, Dict, Optional
import pandas as pd
import plotly.graph_objects as go

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_PATH = os.path.join(APP_DIR, "models", "03_northbengal_spatial_reference.json")

_STATIONS_CACHE: Optional[List[Dict]] = None

def get_all_stations() -> List[Dict]:
    global _STATIONS_CACHE
    if _STATIONS_CACHE is None:
        if os.path.exists(REF_PATH):
            with open(REF_PATH, "r") as f:
                _STATIONS_CACHE = json.load(f)
        else:
            _STATIONS_CACHE = []
    return _STATIONS_CACHE

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points on the earth in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def find_nearest_stations(lat: float, lon: float, top_k: int = 3) -> List[Dict]:
    """Find the top-k nearest monitoring stations and calculate distances."""
    stations = get_all_stations()
    if not stations:
        return []
    
    results = []
    for s in stations:
        dist = haversine_distance_km(lat, lon, s["lat"], s["lon"])
        res = dict(s)
        res["distance_km"] = round(dist, 1)
        results.append(res)
    
    results.sort(key=lambda x: x["distance_km"])
    return results[:top_k]

def interpolate_spatial_risk(lat: float, lon: float, power: float = 2.0) -> Dict:
    """
    Inverse Distance Weighting (IDW) interpolation of Multi-Metal Hazard Index.
    """
    stations = get_all_stations()
    if not stations:
        return {"interpolated_mhi": 0.35, "confidence": "LOW", "nearest_distance_km": 0.0}

    nearest = find_nearest_stations(lat, lon, top_k=5)
    if not nearest:
        return {"interpolated_mhi": 0.35, "confidence": "LOW", "nearest_distance_km": 0.0}

    if nearest[0]["distance_km"] < 0.1:
        return {
            "interpolated_mhi": nearest[0]["mhi_score"],
            "confidence": "VERY_HIGH",
            "nearest_distance_km": nearest[0]["distance_km"],
            "nearest_station": nearest[0]["thana"]
        }

    weights = []
    values = []
    for s in nearest:
        w = 1.0 / (s["distance_km"] ** power)
        weights.append(w)
        values.append(s["mhi_score"] * w)

    interp_val = sum(values) / sum(weights)
    min_dist = nearest[0]["distance_km"]

    if min_dist < 15.0:
        conf = "HIGH"
    elif min_dist < 35.0:
        conf = "MODERATE"
    else:
        conf = "EXTRAPOLATED"

    return {
        "interpolated_mhi": round(float(interp_val), 3),
        "confidence": conf,
        "nearest_distance_km": min_dist,
        "nearest_station": f"{nearest[0]['thana']}, {nearest[0]['district']}"
    }

def build_plotly_spatial_map(user_lat: Optional[float] = None,
                             user_lon: Optional[float] = None,
                             user_risk_label: Optional[str] = None) -> go.Figure:
    """
    Build a clean light-themed OpenStreetMap scatter plot for North Bengal.
    """
    stations = get_all_stations()
    df_st = pd.DataFrame(stations)

    color_map = {
        "LOW_RISK": "#059669",       # Rich Emerald Green
        "MODERATE_RISK": "#d97706",  # Clean Amber/Orange
        "ELEVATED_RISK": "#dc2626",  # Clear Crimson Red
    }

    fig = go.Figure()

    # Add reference monitoring stations grouped by risk category
    for cat, display_name in [
        ("LOW_RISK", "🟢 নিরাপদ টিউবওয়েল (Safe Baseline)"),
        ("MODERATE_RISK", "🟡 মাঝারি সতর্কতা (Moderate Risk)"),
        ("ELEVATED_RISK", "🔴 উচ্চ দূষণ ঝুঁকি (High Risk)")
    ]:
        sub = df_st[df_st["risk_category"] == cat]
        if sub.empty:
            continue

        hover_texts = [
            f"<b>স্টেশন:</b> {r['thana']}, {r['district']}<br>"
            f"<b>গভীরতা:</b> {r['depth_m']} মিটার<br>"
            f"<b>pH:</b> {r['ph']} | <b>TDS:</b> {r['tds_mg_l']} mg/L<br>"
            f"<b>ক্যাডমিয়াম (Cd):</b> {r['cd_ug_l']} µg/L<br>"
            f"<b>আর্সেনিক (As):</b> {r['as_ug_l']} µg/L | <b>সীসা (Pb):</b> {r['pb_ug_l']} µg/L<br>"
            f"<b>হ্যাজার্ড স্কোর:</b> {r['mhi_score']}<br>"
            f"<b>স্ট্যাটাস:</b> {display_name}"
            for _, r in sub.iterrows()
        ]

        fig.add_trace(go.Scattermapbox(
            lat=sub["lat"],
            lon=sub["lon"],
            mode="markers",
            marker=dict(
                size=13,
                color=color_map.get(cat, "#64748b"),
                opacity=0.9
            ),
            name=display_name,
            text=hover_texts,
            hoverinfo="text"
        ))

    # Add user current sample location if provided
    if user_lat is not None and user_lon is not None:
        user_color = color_map.get(user_risk_label, "#0284c7")
        fig.add_trace(go.Scattermapbox(
            lat=[user_lat],
            lon=[user_lon],
            mode="markers+text",
            marker=dict(
                size=22,
                color=user_color,
                symbol="circle"
            ),
            name="📍 আপনার বর্তমান টিউবওয়েল",
            text=["📍 আপনার টিউবওয়েল"],
            textposition="top right",
            hovertext=f"<b>আপনার নির্বাচিত টিউবওয়েল</b><br>অক্ষাংশ: {user_lat:.4f}, দ্রাঘিমাংশ: {user_lon:.4f}<br>স্ট্যাটাস: {user_risk_label}",
            hoverinfo="text"
        ))

    center_lat = user_lat if user_lat is not None else 25.2
    center_lon = user_lon if user_lon is not None else 88.9

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=7.3
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=480,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="#e2e8f0",
            borderwidth=1,
            font=dict(size=12, color="#0f172a", family="Inter, sans-serif")
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff"
    )

    return fig
