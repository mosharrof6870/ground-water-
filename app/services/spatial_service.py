"""
Spatial Service — Geospatial Intelligence for North Bengal Groundwater.
Provides robust interactive mapping via Folium & Plotly for 40 North Bengal stations.
"""

import json
import os
import math
from typing import List, Dict, Optional
import pandas as pd
import folium

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

def build_folium_spatial_map(user_lat: Optional[float] = None,
                             user_lon: Optional[float] = None,
                             user_risk_label: Optional[str] = None) -> str:
    """
    Build a zero-crash, highly resilient interactive Folium HTML map for North Bengal.
    Uses CartoDB positron clean light tiles.
    """
    stations = get_all_stations()
    
    center_lat = user_lat if user_lat is not None else 25.3
    center_lon = user_lon if user_lon is not None else 88.9

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles="CartoDB positron"
    )

    color_map = {
        "LOW_RISK": "#059669",       # Emerald Green
        "MODERATE_RISK": "#d97706",  # Amber/Orange
        "ELEVATED_RISK": "#dc2626",  # Crimson Red
    }

    # Add 40 monitoring station markers
    for r in stations:
        cat = r.get("risk_category", "LOW_RISK")
        c = color_map.get(cat, "#64748b")
        label_text = cat.replace("_", " ").title()
        
        popup_html = (
            f"<div style='font-family:Inter,sans-serif;font-size:12px;color:#0f172a;min-width:180px;'>"
            f"<b>Station:</b> {r['sample_id']}<br>"
            f"<b>Location:</b> {r['thana']}, {r['district']}<br>"
            f"<b>Well Depth:</b> {r['depth_m']} m<br>"
            f"<b>pH:</b> {r['ph']} | <b>TDS:</b> {r['tds_mg_l']} mg/L<br>"
            f"<b>Cadmium (Cd):</b> {r['cd_ug_l']} µg/L<br>"
            f"<b>Arsenic (As):</b> {r['as_ug_l']} µg/L<br>"
            f"<b>Hazard Score (MHI):</b> {r['mhi_score']}<br>"
            f"<b>Status:</b> <span style='color:{c};font-weight:700;'>{label_text}</span>"
            f"</div>"
        )
        
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=7,
            color=c,
            fill=True,
            fill_color=c,
            fill_opacity=0.85,
            weight=1.5,
            tooltip=f"{r['thana']}, {r['district']} ({label_text})",
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(m)

    # Add current user well location marker
    if user_lat is not None and user_lon is not None:
        user_color = color_map.get(user_risk_label, "#0284c7")
        user_popup = (
            f"<div style='font-family:Inter,sans-serif;font-size:12px;color:#0f172a;'>"
            f"<b>📍 Current Evaluated Well</b><br>"
            f"Lat: {user_lat:.4f}, Lon: {user_lon:.4f}<br>"
            f"Screening Result: <b>{user_risk_label}</b>"
            f"</div>"
        )
        folium.Marker(
            location=[user_lat, user_lon],
            icon=folium.Icon(color="blue", icon="tint", prefix="fa"),
            tooltip="📍 Current Tubewell Location",
            popup=folium.Popup(user_popup, max_width=200)
        ).add_to(m)

    return m._repr_html_()
