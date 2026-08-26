"""
North Bengal Groundwater Research Project — Phase 2 Risk Indices Computation & Phase 2.1 Spatial Hotspot Mapping
Target Journal Standards: Q1 (Journal of Hydrology / Water Research / Environmental Pollution)

Author: Research Team
Date: August 2026

Description:
This script computes comprehensive water quality and heavy metal risk indices with EXPLICIT UNIT MATCHING:
1. Fe and Mn are stored as mg/L in raw data; explicitly converted to Fe_ugL and Mn_ugL (x 1000) for comparison against WHO ug/L standards.
2. Original raw variables (Fe_num, Mn_num, As_num, etc.) are strictly preserved.
3. Water Quality Index (WQI - Weighted Arithmetic Index Method across 16 major/minor parameters).
4. Heavy Metal Pollution Index (HPI - WHO threshold weighted composite across As, Fe, Mn, Pb, Ni, Zn).
5. Heavy Metal Evaluation Index (HEI - Relative toxicity burden across heavy metals).
6. Degree of Contamination (Cd - Contamination factor summation).
7. Thanawise GIS Geocoding (Latitude & Longitude for all 40 monitoring wells, explicitly documented as Thana Centroids).
8. Generates interactive HTML map (north_bengal_groundwater_hotspots.html) and saves CSV to data/processed/phase2_risk_indices_results.csv.
"""

import pandas as pd
import numpy as np
import os
import folium

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'processed', 'phase1_5_hydrochemistry_results.csv'))
OUTPUT_CSV = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'processed', 'phase2_risk_indices_results.csv'))
OUTPUT_MAP = os.path.abspath(os.path.join(BASE_DIR, '..', 'reports', 'north_bengal_groundwater_hotspots.html'))

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Input hydrochemistry dataset not found at: {INPUT_CSV}")

# GIS Coordinates dictionary for all 40 monitoring wells in North Bengal Thanas (Thana Centroids)
THANA_COORDS = {
    'Tetulia': (26.5851, 88.3540),
    'Boda': (26.2104, 88.5601),
    'Baliadangi': (26.0984, 88.2778),
    'Thakurgaon Sadar': (26.0337, 88.4617),
    'Birganj': (25.8542, 88.6606),
    'Ranisankail': (25.8711, 88.2514),
    'Dinajpur Sadar': (25.6279, 88.6332),
    'Parbatipur': (25.6565, 88.9168),
    'Hakimpur': (25.2818, 89.0271),
    'Domar': (26.1037, 88.8378),
    'Nilphamari Sadar': (25.9406, 88.8475),
    'Hatibandha': (26.1147, 89.1353),
    'Jaldhaka': (26.0125, 88.9482),
    'Rangpur Sadar': (25.7558, 89.2444),
    'Lalmonirhat Sadar': (25.9181, 89.4475),
    'Nageshwari': (25.9792, 89.7042),
    'Ulipur': (25.6606, 89.6253),
    'Sundarganj': (25.5601, 89.5168),
    'Gaibandha Sadar': (25.3292, 89.5417),
    'Gobindaganj': (25.1311, 89.4002),
    'Jaipurhat Sadar': (25.0968, 89.0225),
    'Adamdighi': (24.8168, 89.0417),
    'Bogra Sadar': (24.8481, 89.3730),
    'Patnitala': (25.0501, 88.7514),
    'Manda': (24.7834, 88.6668),
    'Gomastapur': (24.7765, 88.2834),
    'Shibganj (N)': (24.6851, 88.1617),
    'Godagari': (24.4668, 88.3302),
    'Rajpara': (24.3752, 88.5834),
    'Charghat': (24.2834, 88.7302),
    'Natore Sadar': (24.4102, 88.9834),
    'Singra': (24.5002, 89.1502),
    'Lalpur': (24.1834, 88.9751),
    'Raiganj': (24.5334, 89.5334),
    'Sirajganj Sadar': (24.4584, 89.7084),
    'Chatmohar': (24.2251, 89.2918),
    'Pabna Sadar': (24.0084, 89.2418),
    'Sujanagar': (24.0125, 89.4334),
    'Bera': (24.0668, 89.6251),
    'Sherpur': (24.6668, 89.4168)
}

# Drinking Water Standards (WHO 2011 Guidelines)
# Concentration units strictly matched: mg/L for major parameters, ug/L for trace/heavy metals
STANDARDS_WHO = {
    'pH': 8.5,          # Ideal = 7.0 (unitless)
    'TDS_calc': 500.0,  # mg/L
    'Ca_num': 75.0,     # mg/L
    'Mg_num': 50.0,     # mg/L
    'Na_num': 200.0,    # mg/L
    'K_num': 12.0,      # mg/L
    'Cl_num': 250.0,    # mg/L
    'HCO3_num': 300.0,  # mg/L
    'SO4_num': 250.0,   # mg/L
    'NO3-N_num': 10.0,  # mg/L
    'Fe_ugL': 300.0,    # ug/L
    'Mn_ugL': 100.0,    # ug/L
    'As_ugL': 10.0,     # ug/L
    'Pb_ugL': 10.0,     # ug/L
    'Ni_ugL': 70.0,     # ug/L
    'Zn_ugL': 3000.0    # ug/L
}

# Heavy Metal Specific Standards (ug/L)
HEAVY_METAL_STANDARDS = {
    'As_ugL': 10.0,     # ug/L
    'Fe_ugL': 300.0,    # ug/L
    'Mn_ugL': 100.0,    # ug/L
    'Pb_ugL': 10.0,     # ug/L
    'Ni_ugL': 70.0,     # ug/L
    'Zn_ugL': 3000.0    # ug/L
}

def create_explicit_unit_variables(df):
    """Preserves raw variables and adds explicit unit-labeled variables."""
    df_out = df.copy()
    # Preserve original mg/L and ug/L designations
    df_out['Fe_mgL'] = df_out['Fe_num']
    df_out['Mn_mgL'] = df_out['Mn_num']
    
    # Create explicit ug/L variables for Fe and Mn (mg/L * 1000 = ug/L)
    df_out['Fe_ugL'] = df_out['Fe_num'] * 1000.0
    df_out['Mn_ugL'] = df_out['Mn_num'] * 1000.0
    
    # Explicit unit labels for other heavy metals (already in ug/L)
    df_out['As_ugL'] = df_out['As_num']
    df_out['Pb_ugL'] = df_out['Pb_num']
    df_out['Ni_ugL'] = df_out['Ni_num']
    df_out['Zn_ugL'] = df_out['Zn_num']
    df_out['Cd_ugL'] = df_out['Cd_num'] if 'Cd_num' in df_out else df_out['Cd'] if 'Cd' in df_out and df_out['Cd'].dtype != object else 0.05
    return df_out

def calculate_wqi(df):
    """Calculates Water Quality Index (WQI) via Weighted Arithmetic Index Method."""
    df_wqi = df.copy()
    
    # Calculate Unit Weights (Wi = K / Si)
    inv_s_sum = sum(1.0 / std for std in STANDARDS_WHO.values())
    K = 1.0 / inv_s_sum
    weights = {param: K / std for param, std in STANDARDS_WHO.items()}
    
    wqi_list = []
    for idx, row in df_wqi.iterrows():
        sub_indices = []
        for param, std in STANDARDS_WHO.items():
            val = row[param] if param in row and pd.notna(row[param]) else 0.0
            if param == 'pH':
                # Measured pH defaults to 7.0 if missing, or use pH_proxy if available
                pH_val = row['pH'] if 'pH' in row and pd.notna(row['pH']) else 7.0
                qi = (abs(pH_val - 7.0) / (8.5 - 7.0)) * 100.0
            else:
                qi = (val / std) * 100.0
            si = weights[param] * qi
            sub_indices.append(si)
        wqi = sum(sub_indices)
        wqi_list.append(wqi)
        
    df_wqi['WQI'] = wqi_list
    
    def classify_wqi(wqi):
        if wqi < 50:
            return "Excellent Water (WQI < 50)"
        elif wqi < 100:
            return "Good Water (50 <= WQI < 100)"
        elif wqi < 200:
            return "Poor Water (100 <= WQI < 200)"
        elif wqi < 300:
            return "Very Poor Water (200 <= WQI < 300)"
        else:
            return "Unsuitable for Drinking (WQI >= 300)"
            
    df_wqi['WQI_Category'] = df_wqi['WQI'].apply(classify_wqi)
    return df_wqi

def calculate_heavy_metal_indices(df):
    """Calculates HPI, HEI, and Cd across heavy metals (As, Fe, Mn, Pb, Ni, Zn) with unit-matched standards."""
    df_hm = df.copy()
    
    inv_std_sum = sum(1.0 / std for std in HEAVY_METAL_STANDARDS.values())
    K = 1.0 / inv_std_sum
    weights = {param: K / std for param, std in HEAVY_METAL_STANDARDS.items()}
    
    hpi_list = []
    hei_list = []
    cd_list = []
    
    for idx, row in df_hm.iterrows():
        # 1. HPI Calculation
        hpi_numerator = 0.0
        hpi_denominator = sum(weights.values())
        for param, std in HEAVY_METAL_STANDARDS.items():
            val = row[param] if param in row and pd.notna(row[param]) else 0.0
            qi = (val / std) * 100.0
            hpi_numerator += weights[param] * qi
        hpi = hpi_numerator / hpi_denominator
        hpi_list.append(hpi)
        
        # 2. HEI Calculation (HEI = Sum(Mi / Si))
        hei = sum((row[param] if param in row and pd.notna(row[param]) else 0.0) / std 
                  for param, std in HEAVY_METAL_STANDARDS.items())
        hei_list.append(hei)
        
        # 3. Degree of Contamination Cd (Cd = Sum((Mi / Si) - 1))
        cd = sum(((row[param] if param in row and pd.notna(row[param]) else 0.0) / std) - 1.0 
                 for param, std in HEAVY_METAL_STANDARDS.items())
        cd_list.append(cd)
        
    df_hm['HPI'] = hpi_list
    df_hm['HEI'] = hei_list
    df_hm['Cd'] = cd_list
    
    df_hm['HPI_Polluted'] = df_hm['HPI'] > 100.0  # Critical threshold HPI > 100
    
    def classify_hei(hei):
        if hei < 10:
            return "Low Contamination (HEI < 10)"
        elif hei <= 20:
            return "Moderate Contamination (10 <= HEI <= 20)"
        else:
            return "High Contamination (HEI > 20)"
            
    df_hm['HEI_Category'] = df_hm['HEI'].apply(classify_hei)
    return df_hm

def geocode_samples(df):
    """Assigns Thana-level administrative centroid Latitude and Longitude coordinates."""
    df_geo = df.copy()
    lats = []
    lons = []
    for idx, row in df_geo.iterrows():
        thana = str(row['THANA']).strip() if 'THANA' in row and pd.notna(row['THANA']) else ''
        if thana in THANA_COORDS:
            lat, lon = THANA_COORDS[thana]
        else:
            lat, lon = (25.0, 89.0)  # Default fallback centroid
        lats.append(lat)
        lons.append(lon)
    df_geo['Latitude'] = lats
    df_geo['Longitude'] = lons
    return df_geo

def generate_spatial_hotspot_map(df, map_path):
    """Generates an interactive Folium GIS map visualizing WQI and HPI Hotspots."""
    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles='cartodbpositron')
    
    for idx, row in df.iterrows():
        sample_id = row['SAMPLE_ID']
        district = row['DISTRICT']
        thana = row['THANA']
        wqi = row['WQI']
        hpi = row['HPI']
        hei = row['HEI']
        facies = row['Facies']
        as_val = row['As_ugL']
        fe_val = row['Fe_mgL']
        
        # Color coding based on WQI
        if wqi < 50:
            color = 'green'
        elif wqi < 100:
            color = 'blue'
        elif wqi < 200:
            color = 'orange'
        elif wqi < 300:
            color = 'purple'
        else:
            color = 'red'
            
        popup_html = f"""
        <div style="font-family: Arial; width: 230px;">
            <b>Sample ID:</b> {sample_id}<br>
            <b>Location (Thana Centroid):</b> {thana}, {district}<br>
            <b>Facies:</b> {facies}<br>
            <hr style="margin:4px 0;">
            <b>WQI:</b> {wqi:.2f} ({row['WQI_Category']})<br>
            <b>HPI:</b> {hpi:.2f} {'<span style="color:red;font-weight:bold;">[CRITICAL POLLUTED]</span>' if row['HPI_Polluted'] else '[Normal]'}<br>
            <b>HEI:</b> {hei:.2f}<br>
            <b>As Conc:</b> {as_val:.1f} &mu;g/L<br>
            <b>Fe Conc:</b> {fe_val:.2f} mg/L ({row['Fe_ugL']:.0f} &mu;g/L)
        </div>
        """
        
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=7 if not row['HPI_Polluted'] else 11,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{sample_id} ({district}) - WQI: {wqi:.1f}, HPI: {hpi:.1f}"
        ).add_to(m)
        
    m.save(map_path)
    print(f"Interactive Spatial Hotspot Map saved to: {map_path}")

def main():
    print(f"Loading hydrochemistry dataset from: {INPUT_CSV}")
    df_clean = pd.read_csv(INPUT_CSV)
    
    # 1. Create Explicit Unit Variables
    df_units = create_explicit_unit_variables(df_clean)
    
    # 2. Calculate WQI
    df_wqi = calculate_wqi(df_units)
    
    # 3. Calculate Heavy Metal Risk Indices (HPI, HEI, Cd)
    df_risk = calculate_heavy_metal_indices(df_wqi)
    
    # 4. Geocode GIS Coordinates (Thana Centroids)
    df_final = geocode_samples(df_risk)
    
    print("\n=== PHASE 2 REPAIRED RISK INDICES SUMMARY (N=40, Unit-Matched) ===")
    print(f"  WQI Mean = {df_final['WQI'].mean():.2f}, Median = {df_final['WQI'].median():.2f}, Range = {df_final['WQI'].min():.2f} - {df_final['WQI'].max():.2f}")
    print("\n  WQI Category Counts:")
    for cat, cnt in df_final['WQI_Category'].value_counts().items():
        print(f"    {cat}: {cnt} samples ({cnt/len(df_final)*100:.1f}%)")
        
    print(f"\n  HPI Mean = {df_final['HPI'].mean():.2f}, Median = {df_final['HPI'].median():.2f}, Range = {df_final['HPI'].min():.2f} - {df_final['HPI'].max():.2f}")
    polluted_cnt = df_final['HPI_Polluted'].sum()
    print(f"    Critical HPI (>100) Polluted Hotspots: {polluted_cnt} samples ({polluted_cnt/len(df_final)*100:.1f}%)")

    print(f"\n  HEI Mean = {df_final['HEI'].mean():.2f}, Median = {df_final['HEI'].median():.2f}, Range = {df_final['HEI'].min():.2f} - {df_final['HEI'].max():.2f}")
    print("\n  HEI Category Counts:")
    for cat, cnt in df_final['HEI_Category'].value_counts().items():
        print(f"    {cat}: {cnt} samples ({cnt/len(df_final)*100:.1f}%)")

    # 5. Save Processed Output Dataset
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"\nRepaired Phase 2 Risk Indices dataset saved to: {OUTPUT_CSV}")
    
    # 6. Generate Interactive Folium GIS Map
    os.makedirs(os.path.dirname(OUTPUT_MAP), exist_ok=True)
    generate_spatial_hotspot_map(df_final, OUTPUT_MAP)

if __name__ == '__main__':
    main()
