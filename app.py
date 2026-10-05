import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# MATPLOTLIB CONFIGURATION (Prevents Math/LaTeX Parsing Crashes)
# ---------------------------------------------------------
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.default'] = 'regular'

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="IndusMind AI | Predictive Maintenance",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark UI Enhancement
st.markdown("""
<style>
    /* Global Styling */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Header Container */
    .header-box {
        background: linear-gradient(90deg, #1f2937 0%, #111827 100%);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #374151;
        margin-bottom: 25px;
    }
    
    /* Custom Alert Badges */
    .status-badge {
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        text-align: center;
        display: inline-block;
        width: 100%;
        font-size: 16px;
    }
    .badge-critical { background-color: #7f1d1d; color: #fca5a5; border: 1px solid #ef4444; }
    .badge-warning { background-color: #78350f; color: #fcd34d; border: 1px solid #f59e0b; }
    .badge-normal { background-color: #064e3b; color: #6ee7b7; border: 1px solid #10b981; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 1. LOAD MODEL ARTIFACTS
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    xgb_model = joblib.load('models/xgb_failure_model.pkl')
    iso_forest = joblib.load('models/iso_forest.pkl')
    feature_cols = joblib.load('models/feature_cols.pkl')
    explainer = shap.TreeExplainer(xgb_model)
    return xgb_model, iso_forest, feature_cols, explainer

try:
    xgb_model, iso_forest, feature_cols, explainer = load_artifacts()
except Exception as e:
    st.error("⚠️ Error loading trained models. Run 'python train.py' first.")
    st.stop()

# ---------------------------------------------------------
# 2. HEADER SECTION
# ---------------------------------------------------------
st.markdown("""
<div class="header-box">
    <h1 style="margin:0; font-size: 32px; color: #60a5fa;">⚡ IndusMind AI</h1>
    <p style="margin-top: 5px; color: #9ca3af; font-size: 15px;">
        Intelligent Predictive Maintenance & Real-time Anomaly Diagnostics for Industrial CNC Spindles
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. MAIN DASHBOARD TABS
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["🎛️ Live Machine Diagnostics", "📊 Fleet Health Monitor"])

with tab1:
    col_ctrl, col_diag = st.columns([1, 1.4], gap="large")
    
    # ---------------------------------------------------------
    # LEFT COLUMN: SENSOR INPUT CONTROLS
    # ---------------------------------------------------------
    with col_ctrl:
        st.markdown("### 🎛️ Sensor Input Control Panel")
        st.caption("Adjust parameters manually or load operational scenario presets.")
        
        preset = st.selectbox(
            "📋 Operational Presets",
            ["Normal Operation", "High Thermal Load", "Excessive Tool Strain & Wear"]
        )
        
        if preset == "High Thermal Load":
            def_air, def_proc, def_speed, def_torque, def_wear = 300.0, 314.0, 1500, 40.0, 50
        elif preset == "Excessive Tool Strain & Wear":
            def_air, def_proc, def_speed, def_torque, def_wear = 298.0, 308.0, 1200, 68.0, 220
        else:
            def_air, def_proc, def_speed, def_torque, def_wear = 298.0, 308.0, 1500, 35.0, 30

        st.divider()

        air_temp = st.slider("🌡️ Air Temperature [K]", 290.0, 310.0, def_air, 0.5)
        proc_temp = st.slider("🔥 Process Temperature [K]", 295.0, 320.0, def_proc, 0.5)
        speed = st.slider("⚙️ Rotational Speed [RPM]", 1100, 2900, def_speed, 50)
        torque = st.slider("💪 Torque [Nm]", 10.0, 80.0, def_torque, 1.0)
        wear = st.slider("⏳ Cumulative Tool Wear [min]", 0, 250, def_wear, 5)

        # Feature Computations
        temp_diff = proc_temp - air_temp
        power_w = torque * (speed * (2 * np.pi / 60))
        overstrain = wear * torque

        input_data = pd.DataFrame([[
            air_temp, proc_temp, speed, torque, wear,
            temp_diff, power_w, overstrain
        ]], columns=feature_cols)

    # ---------------------------------------------------------
    # RIGHT COLUMN: REAL-TIME DIAGNOSTICS & EXPLAINABILITY
    # ---------------------------------------------------------
    with col_diag:
        st.markdown("### 📊 Model Inference & Health Status")
        
        failure_prob = xgb_model.predict_proba(input_data)[0][1]
        anomaly_flag = iso_forest.predict(input_data)[0]
        health_score = int(max(0, 100 - (failure_prob * 100)))

        if failure_prob > 0.70:
            badge_html = '<div class="status-badge badge-critical">🚨 CRITICAL RISK DETECTED</div>'
            rec_text = "🛑 **Immediate Action Required:** Shut down spindle. Replace worn tool bit and inspect cooling subsystem."
        elif failure_prob > 0.35:
            badge_html = '<div class="status-badge badge-warning">⚠️ ELEVATED FAILURE RISK</div>'
            rec_text = "⚠️ **Warning:** Schedule technical inspection within 4 operational hours. Check lubricant and balance."
        else:
            badge_html = '<div class="status-badge badge-normal">✅ OPTIMAL OPERATIONAL STATE</div>'
            rec_text = "✅ **System Healthy:** Continuous monitoring active. Operational metrics within safe bounds."

        st.markdown(badge_html, unsafe_allow_html=True)
        st.write("")

        m1, m2, m3 = st.columns(3)
        m1.metric("Health Index", f"{health_score}/100")
        m2.metric("Failure Risk", f"{failure_prob * 100:.1f}%")
        m3.metric("Anomaly Flag", "NORMAL" if anomaly_flag == 1 else "⚠️️ ANOMALY")

        st.divider()

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_score,
            number={'suffix': "%", 'font': {'color': "white", 'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#60a5fa", 'thickness': 0.3},
                'bgcolor': "#1f2937",
                'borderwidth': 1,
                'bordercolor': "#374151",
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(239, 68, 68, 0.4)'},
                    {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.4)'},
                    {'range': [70, 100], 'color': 'rgba(16, 185, 129, 0.4)'}
                ]
            }
        ))
        fig_gauge.update_layout(
            height=180,
            margin=dict(l=30, r=30, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"}
        )
        st.plotly_chart(fig_gauge, width="stretch")

        st.info(rec_text)

        # ---------------------------------------------------------
        # SHAP EXPLAINABILITY SECTION
        # ---------------------------------------------------------
        st.markdown("### 🔍 Root-Cause Analysis (SHAP Explanation)")
        st.caption("Quantifying feature impacts pushing health risk up or down.")

        plt.close('all')
        fig_shap, ax = plt.subplots(figsize=(8, 3.5))
        fig_shap.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        
        shap_values = explainer(input_data)
        shap.plots.waterfall(shap_values[0], show=False)
        
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(colors='white', which='both', labelsize=9)
        
        for text in fig_shap.texts:
            text.set_color('white')
            
        for spine in ax.spines.values():
            spine.set_color('#374151')

        st.pyplot(fig_shap)

# ---------------------------------------------------------
# TAB 2: MULTI-MACHINE FLEET MONITOR
# ---------------------------------------------------------
with tab2:
    st.markdown("### 🏭 Factory Floor CNC Matrix")
    st.caption("Real-time telemetry and predictive status across operational CNC spindles.")

    fleet_data = pd.DataFrame([
        {"Machine ID": "CNC-001 (Main Assembly)", "Speed [RPM]": 1500, "Torque [Nm]": 35.0, "Tool Wear": "30 min", "Health Score": "98%", "Status": "Normal"},
        {"Machine ID": "CNC-002 (Heavy Milling)", "Speed [RPM]": 2800, "Torque [Nm]": 68.0, "Tool Wear": "220 min", "Health Score": "14%", "Status": "Critical"},
        {"Machine ID": "CNC-003 (Precision Lathe)", "Speed [RPM]": 1200, "Torque [Nm]": 42.0, "Tool Wear": "85 min", "Health Score": "88%", "Status": "Normal"},
        {"Machine ID": "CNC-004 (Secondary Line)", "Speed [RPM]": 1800, "Torque [Nm]": 55.0, "Tool Wear": "140 min", "Health Score": "62%", "Status": "Warning"}
    ])

    st.dataframe(fleet_data, width="stretch", hide_index=True)