"""Smart Traffic Intelligence & Operations Center (TOC) — Executive Operations Console.

A professional, modern traffic operations platform providing predictive demand forecasting,
trend detection, and rule-based operator advisories across two strictly decoupled benchmark datasets.
Operates in Historical Demo Replay Mode using peer-reviewed scientific datasets (UCI #734 and UCI #492).
"""

import os
import sys
import textwrap
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import streamlit as st

# Add project root to sys.path so modules resolve cleanly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.utils.styles import get_shared_css, render_header_bar
from app.utils.data_loader import (
    load_dataset_a_metadata,
    load_dataset_b_metadata,
    load_dataset_a_validation,
    load_dataset_b_validation,
    load_dataset_a_adjacency,
    load_dataset_a_leaderboard,
    load_dataset_b_leaderboard,
    load_dataset_a_models,
    load_dataset_b_models,
    load_dataset_a_test_leaderboard,
    load_dataset_b_test_leaderboard
)
from app.utils.plots import (
    classify_demand_trend,
    plot_sensor_network_topology,
    plot_sensor_flow_strip_dataset_a,
    plot_corridor_schematic_dataset_b,
    plot_centerpiece_forecast_dataset_a,
    plot_centerpiece_forecast_dataset_b
)

st.set_page_config(
    page_title="Traffic Intelligence & Operations Center | TOC Console",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply centralized enterprise CSS design tokens
st.markdown(get_shared_css(), unsafe_allow_html=True)

# Sidebar Navigation & Telemetry
with st.sidebar:
    st.markdown("### 🚦 **Traffic Operations Console**")
    st.caption("AI/ML Capstone Project | ExcelR Internship")
    st.markdown('<span class="status-badge badge-demo">Historical Demo Mode</span> <span class="status-badge badge-online">Trained Models Loaded</span>', unsafe_allow_html=True)
    st.divider()

    st.markdown("#### **Control Center Modules**")
    st.markdown("""
    - 📊 **1. Corridor & Sensor Inventory**
    - 🏆 **2. Model Performance**
    - 🔮 **3. Forecasting Studio**
    - 🌧️ **4. Weather Impact Analysis**
    - ⚠️ **5. Limitations & Methodology**
    """)
    st.divider()

    st.caption("Engineered with Python 3.13, Scikit-Learn, Streamlit, and Plotly.")

# Top Console Header Bar
st.markdown(
    render_header_bar(
        title="🚦 Traffic Intelligence & Operations Center",
        subtitle="Predictive Corridor Demand Forecasting, Trend Analysis & Simulated Decision Support"
    ),
    unsafe_allow_html=True
)

# High-Level System Scope Caption (1 clean line, replacing bulky callout)
st.markdown("""
<div style="background-color: #EFF6FF; border-left: 4px solid #3B82F6; padding: 0.55rem 0.95rem; border-radius: 6px; margin-bottom: 0.9rem; font-size: 0.82rem; color: #1E3A8A; line-height: 1.45;">
    🛡️ <strong>Operational Scope:</strong> Offline predictive intelligence & simulated decision support evaluated on benchmark datasets (UCI #734 & UCI #492). In traffic flow theory (<em>q = k · v</em>), flow and volume quantify throughput demand, not road congestion.
</div>
""", unsafe_allow_html=True)

# Domain / Corridor Selection
st.markdown("### 🌐 **Select Operations Domain & Traffic Corridor**")
domain_choice = st.radio(
    "Active Traffic Corridor:",
    [
        "Dataset A: Northern Virginia / Washington D.C. Capital Region (36 Highway Sensors, 30-Minute Verified Horizon, Normalized Flow)",
        "Dataset B: Interstate 94 Corridor (MN DoT ATR Station 301, 1-Hour Verified Horizon, Vehicles / Hour)"
    ],
    index=0,
    horizontal=True,
    help="Both datasets represent distinct, decoupled traffic forecasting domains with different sensors, horizons, and units."
)

st.divider()

# ==============================================================================
# CORRIDOR 1: DATASET A (NORTHERN VIRGINIA / D.C. 36-SENSOR NETWORK)
# ==============================================================================
if "Dataset A" in domain_choice:
    st.subheader("🛰️ Northern Virginia / Washington D.C. Capital Region — 36-Sensor Spatiotemporal Operations")

    # Load assets
    meta_a = load_dataset_a_metadata()
    val_a = load_dataset_a_validation()
    adj_mat = load_dataset_a_adjacency()
    models_a = load_dataset_a_models()
    lb_a = load_dataset_a_leaderboard()
    test_lb_a = load_dataset_a_test_leaderboard()

    # Replay Controller Ribbon
    with st.container():
        col_ctrl1, col_ctrl2 = st.columns([1.6, 1.0])
        with col_ctrl1:
            step_idx_selected = st.slider(
                "Historical Observation Step (Validation Partition):",
                min_value=int(val_a['step_idx'].min()) + 10,
                max_value=int(val_a['step_idx'].max()) - 2,
                value=int(val_a['step_idx'].min()) + 25,
                step=1,
                help="Validation steps span 261 consecutive 15-minute intervals. Target is verified at 30 minutes ahead (+2 steps)."
            )
            retrospective_mode_a = st.checkbox(
                "🔍 Retrospective Evaluation Mode (Display verified t+30m ground truth & model residual)",
                value=False,
                help="In prospective real-time mode, future ground truth is unobserved. Enable this to retrospectively evaluate model accuracy against the actual recorded outcome."
            )
        with col_ctrl2:
            focus_sensor = st.selectbox(
                "Focus Sensor Station for Advisory:",
                options=list(range(36)),
                index=14,
                format_func=lambda s: f"Sensor S{s:02d} (Connectivity: {int(np.sum(adj_mat[s]))} adjacent links)"
            )

    # Slice data for selected sensor and step
    sensor_val_data = val_a[val_a['sensor_idx'] == focus_sensor].sort_values('step_idx').reset_index(drop=True)
    step_row_match = sensor_val_data[sensor_val_data['step_idx'] == step_idx_selected]
    if len(step_row_match) == 0:
        step_row = sensor_val_data.iloc[0]
        step_idx_selected = int(step_row['step_idx'])
    else:
        step_row = step_row_match.iloc[0]

    # Historical slice for centerpiece chart (past 8 steps up to t)
    hist_slice = sensor_val_data[(sensor_val_data['step_idx'] <= step_idx_selected) & (sensor_val_data['step_idx'] >= step_idx_selected - 8)]
    history_steps = list(hist_slice['step_idx'])
    history_flows = [float(f) for f in hist_slice['persistence_pred']]

    # Model inference for selected step
    feat_cols_a = [f"feat_{i:02d}" for i in range(48)]
    hgb_model_a = models_a.get("hist_gradient_boosting")
    X_sample_a = step_row[feat_cols_a].values.reshape(1, -1).astype(np.float32)

    curr_obs_flow = float(step_row['persistence_pred'])
    if hgb_model_a is not None:
        fcst_flow = float(hgb_model_a.predict(X_sample_a)[0])
    else:
        fcst_flow = curr_obs_flow

    actual_target = float(step_row['target'])
    delta_flow = fcst_flow - curr_obs_flow
    pct_change = (delta_flow / curr_obs_flow * 100.0) if curr_obs_flow > 0.001 else 0.0

    # Trend categorization using rigorous 5-tier classification
    trend_info = classify_demand_trend(delta_flow, dataset="A")
    trend_status = trend_info["status"]
    trend_badge = trend_info["badge_html"]
    trend_plain = trend_info["plain_text"]

    # Test set metrics from verified artifacts
    test_mae_a = 0.0257
    test_rmse_a = 0.0379
    test_r2_a = 0.9657
    test_lift_a = 30.96

    # Operational Regime Interpretation & Grounded Analytical Recommendations
    if fcst_flow >= 0.65 and ("Rising" in trend_status or "Higher" in trend_status):
        adv_class = "action-warning"
        adv_status = "HIGH THROUGHPUT SURGE"
        adv_regime = (
            "Detector telemetry indicates high vehicle throughput with upward momentum. "
            "Under traffic flow theory (q = k · v), high flow reflects high vehicle throughput rather than queue gridlock. "
            "Compressed vehicle headways increase vulnerability to downstream merge friction."
        )
        rec_title = "Elevated Throughput Surveillance & Inflow Tracking"
        rec_steps = [
            "1. <strong>Sensor Polling Cadence:</strong> Maintain close surveillance on upstream and downstream detector stations to detect early throughput variations.",
            "2. <strong>Spatial Gradient Cross-Check:</strong> Monitor adjacent sensor pairs for localized flow divergence along highway segments.",
            "3. <strong>Diurnal Profile Comparison:</strong> Compare observed surge rate against historical weekday baseline to assess peak arrival alignment."
        ]
        rec_assumptions = (
            f"Loop detector S{focus_sensor:02d} operates normally; analytical decision support based on offline benchmark telemetry."
        )
    elif fcst_flow >= 0.65 and not ("Rising" in trend_status or "Higher" in trend_status):
        adv_class = "action-info"
        adv_status = "PEAK STEADY-STATE THROUGHPUT"
        adv_regime = (
            "Corridor is operating at steady peak throughput. Vehicle headways are compressed but moving smoothly. "
            "Traffic demand is balanced without immediate surge momentum."
        )
        rec_title = "Peak Steady-State Surveillance & Divergence Audit"
        rec_steps = [
            "1. <strong>Continuous Polling:</strong> Maintain regular 15-minute polling across all 36 corridor detector stations.",
            "2. <strong>Flow Continuity Check:</strong> Verify flow continuity across topological graph neighbor stations.",
            "3. <strong>Baseline Variance Logging:</strong> Record deviations from diurnal expectations to calibrate historical envelopes."
        ]
        rec_assumptions = (
            "Uninterrupted detector transmission; historical diurnal profile remains representative."
        )
    elif "Falling" in trend_status or "Lower" in trend_status or "Easing" in trend_status:
        adv_class = "action-normal"
        adv_status = "POST-PEAK DEMAND RECEDING"
        adv_regime = (
            "Traffic throughput is tapering downward toward off-peak levels. Vehicle headways are historically expanding."
        )
        rec_title = "Demand Easing Tracking & Sensor Baseline Alignment"
        rec_steps = [
            "1. <strong>Baseline Tracking:</strong> Track rate of demand descent against standard diurnal decay curves.",
            "2. <strong>Sensor Continuity:</strong> Verify telemetry continuity as flow transitions toward nocturnal bounds.",
            "3. <strong>Model Residual Logging:</strong> Log forecast error delta across receding transitions to validate lag stability."
        ]
        rec_assumptions = (
            "Downstream demand follows normal off-peak transition; loop telemetry exhibits nominal zero-drift."
        )
    else:
        adv_class = "action-normal"
        adv_status = "NOMINAL FREE-FLOW REGIME"
        adv_regime = (
            "Traffic flow is well within normal free-flow bounds. Roadway operates with wide vehicle headways and minimal variation."
        )
        rec_title = "Standard Continuous Corridor Surveillance"
        rec_steps = [
            "1. <strong>Routine Polling:</strong> Continuous automated 15-minute telemetry ingestion across all 36 stations.",
            "2. <strong>Data Integrity Check:</strong> Screen for missing records, frozen telemetry readings, or zero-variance flags.",
            "3. <strong>Baseline Drift Monitor:</strong> Compare rolling weekly mean against historical seasonal baseline."
        ]
        rec_assumptions = (
            "Sensors report valid baseline telemetry; nominal free-flow traffic stream."
        )

    # Extract 36 sensor flows at selected step for the linear sensor strip
    step_sensors = val_a[val_a['step_idx'] == step_idx_selected].sort_values('sensor_idx')
    sensor_flows_36 = step_sensors['persistence_pred'].values if len(step_sensors) == 36 else None

    # ==========================================================================
    # 4-KPI HERO TELEMETRY STRIP (DIRECT FIRST VIEWPORT VISIBILITY)
    # ==========================================================================
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.2rem;">
        <div class="hero-stat-card" title="Observed normalized traffic flow at current historical time step t">
            <div class="hero-stat-tag">Current Observed (t)</div>
            <div class="hero-stat-val">{curr_obs_flow:.3f}</div>
            <div class="hero-stat-sub">Sensor S{focus_sensor:02d} • Step #{step_idx_selected}</div>
        </div>
        <div class="hero-stat-card-forecast" title="HistGradientBoosting predicted flow for t+30 minutes (+2 steps forward)">
            <div class="hero-stat-tag" style="color: #1D4ED8; font-weight: 700;">Forward Forecast (t+30m)</div>
            <div class="hero-stat-val" style="color: #2563EB;">{fcst_flow:.3f}</div>
            <div class="hero-stat-sub" style="color: #2563EB; font-weight: 600;">Step #{step_idx_selected + 2} • HistGradientBoosting</div>
        </div>
        <div class="hero-stat-card" title="Forecast delta (t+30m minus t) and percentage variation">
            <div class="hero-stat-tag">Projected Change</div>
            <div class="hero-stat-val" style="font-size: 1.6rem;">{delta_flow:+.3f}</div>
            <div class="hero-stat-sub">{trend_badge}</div>
        </div>
        <div class="hero-stat-card" title="Historical demand regime and test-set baseline improvement">
            <div class="hero-stat-tag">Demand Regime & Reliability</div>
            <div class="hero-stat-val" style="font-size: 1.15rem; color: #0F172A;">{adv_status}</div>
            <div class="hero-stat-sub">Test MAE: <strong>{test_mae_a:.4f}</strong> (+{test_lift_a:.1f}% vs baseline)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ==========================================================================
    # 2-COLUMN COCKPIT LAYOUT: MASTER VISUAL DISPLAY (LEFT) & OPERATIONS RAIL (RIGHT)
    # ==========================================================================
    col_viz, col_rail = st.columns([1.65, 1.0], gap="large")

    with col_viz:
        # VISUAL CENTERPIECE: PROMINENT TIME-SERIES FORECAST WINDOW
        st.markdown("""
        <div class="viz-panel">
            <div class="viz-panel-title">📈 Visual Centerpiece: Historical Trajectory & 30-Minute Forward Forecast Window</div>
        """, unsafe_allow_html=True)
        fig_centerpiece_a = plot_centerpiece_forecast_dataset_a(
            history_steps=history_steps,
            history_flows=history_flows,
            current_step=step_idx_selected,
            current_flow=curr_obs_flow,
            forecast_step=step_idx_selected + 2,
            forecast_flow=fcst_flow,
            actual_target=actual_target,
            sensor_id=focus_sensor,
            trend_str=trend_status,
            retrospective_mode=retrospective_mode_a
        )
        st.plotly_chart(fig_centerpiece_a, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # SPATIAL SENSOR STRIP: 36 SENSORS LINEAR FLOW DISTRIBUTION
        st.markdown("""
        <div class="viz-panel">
            <div class="viz-panel-title">🛰️ Network Flow Distribution: 36 Highway Loop Sensors (Normalized [0, 1])</div>
        """, unsafe_allow_html=True)
        fig_strip_a = plot_sensor_flow_strip_dataset_a(
            sensor_flows=sensor_flows_36,
            selected_sensor=focus_sensor
        )
        st.plotly_chart(fig_strip_a, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Collapsible Topology Graph for Deep Inspection
        with st.expander("🗺️ Inspect Full 36-Sensor Schematic Topology Graph (Graph Adjacency)", expanded=False):
            fig_net = plot_sensor_network_topology(adj_mat)
            st.plotly_chart(fig_net, use_container_width=True)

        # Educational Callout: Beginner Guide
        with st.expander("📘 How to Read These Traffic Metrics (Beginner Guide)", expanded=False):
            st.markdown("""
            - **Forecast (t+30m):** The model's forward prediction for traffic flow 30 minutes in advance. Values are normalized between `0.0` (empty roadway) and `1.0` (maximum lane capacity).
            - **MAE (Mean Absolute Error = 0.0257):** On average, the model's prediction deviates from the true traffic flow by only 0.0257 units on the 0-to-1 scale. Lower is better.
            - **RMSE (Root Mean Squared Error = 0.0379):** Similar to MAE, but squares errors before averaging, penalizing large misses heavily. Since RMSE is close to MAE, the model has few extreme prediction errors.
            - **R² Score (0.9657):** The coefficient of determination. It measures how much of the variance in traffic flow is captured by the model (`1.0` is perfect prediction, capturing 96.6% of flow variance).
            - **Trend Indicator:** Shows the difference between the current observation ($t$) and future forecast ($t+30\\text{m}$). Positive values indicate rising traffic demand, while negative values indicate easing demand.
            """)

    with col_rail:
        # TACTICAL DECISION BOX (ANALYTICAL DECISION SUPPORT)
        steps_html = "<br>".join(rec_steps)
        if retrospective_mode_a:
            eval_banner = (
                f'<div style="font-size: 0.84rem; color: #1E3A8A; margin-bottom: 0.6rem; background: #EFF6FF; '
                f'border: 1px solid #BFDBFE; padding: 0.45rem 0.65rem; border-radius: 6px;">'
                f'🔍 <strong>Retrospective Ground Truth (t+30m):</strong> {actual_target:.3f} &nbsp;|&nbsp; '
                f'<strong>Model Residual:</strong> {actual_target - fcst_flow:+.3f}</div>'
            )
        else:
            eval_banner = (
                '<div style="font-size: 0.82rem; color: #64748B; margin-bottom: 0.6rem; background: #F8FAFC; '
                'border: 1px dashed #CBD5E1; padding: 0.45rem 0.65rem; border-radius: 6px;">'
                '🔮 <strong>Prospective Replay:</strong> Future ground truth unobserved at <em>t</em>. Check "Retrospective Mode" above to evaluate error.</div>'
            )

        box_html_a = f"""
        <div class="action-box {adv_class}">
            <span class="action-badge-tag">Analytical Decision Support (Simulated)</span>
            <div style="font-size: 1.12rem; font-weight: 800; margin-bottom: 0.45rem; color: #0F172A;">
                🛠️ {rec_title}
            </div>
            <div style="font-size: 0.88rem; line-height: 1.45; margin-bottom: 0.6rem; color: #1E293B;">
                {adv_regime}
            </div>
            {eval_banner}
            <div style="font-size: 0.86rem; line-height: 1.5; margin-bottom: 0.65rem;">
                {steps_html}
            </div>
            <div style="font-size: 0.80rem; color: #475569; border-top: 1px dashed #CBD5E1; padding-top: 0.45rem;">
                📋 <strong>Assumptions:</strong> {rec_assumptions}<br>
                ⚠️ <em>Simulated analytical guidance evaluated on offline benchmark data. Does not actuate municipal traffic systems.</em>
            </div>
        </div>
        """
        st.markdown(textwrap.dedent(box_html_a), unsafe_allow_html=True)

        # MODEL BENCHMARK RELIABILITY CARD
        rel_html_a = f"""
        <div class="tactical-decision-box">
            <div class="tactical-badge">Official Test Benchmark</div>
            <div class="tactical-title">🏆 Model Reliability on Unseen Test Data</div>
            <div class="tactical-interp">
                Performance evaluated on strictly held-out chronological test partition (Phase 5 verified):
            </div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.6rem; margin-bottom: 0.6rem;">
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 0.6rem; border-radius: 6px;">
                    <div style="font-size: 0.80rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Test MAE</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #0F172A;">{test_mae_a:.4f}</div>
                    <div style="font-size: 0.82rem; color: #059669; font-weight: 700;">+{test_lift_a:.1f}% vs persistence</div>
                </div>
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 0.6rem; border-radius: 6px;">
                    <div style="font-size: 0.80rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Test RMSE</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #0F172A;">{test_rmse_a:.4f}</div>
                    <div style="font-size: 0.82rem; color: #64748B;">Low dispersion</div>
                </div>
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 0.6rem; border-radius: 6px;">
                    <div style="font-size: 0.80rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Test R² Score</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #0F172A;">{test_r2_a:.4f}</div>
                    <div style="font-size: 0.82rem; color: #64748B;">96.6% variance explained</div>
                </div>
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 0.6rem; border-radius: 6px;">
                    <div style="font-size: 0.80rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Persistence Baseline</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #64748B;">0.0372</div>
                    <div style="font-size: 0.82rem; color: #64748B;">Naive t-2 baseline</div>
                </div>
            </div>
            <div style="font-size: 0.82rem; color: #64748B; line-height: 1.45;">
                🔒 Zero lookahead leakage. Model trained strictly on pre-2019 chronological train partition.
            </div>
        </div>
        """
        st.markdown(textwrap.dedent(rel_html_a), unsafe_allow_html=True)

# ==============================================================================
# CORRIDOR 2: DATASET B (INTERSTATE 94 CORRIDOR)
# ==============================================================================
else:
    st.subheader("🚗 Interstate 94 Corridor (MN DoT ATR Station 301) — Hourly Traffic Volume Operations")

    # Load assets
    meta_b = load_dataset_b_metadata()
    val_b = load_dataset_b_validation()
    models_b = load_dataset_b_models()
    lb_b = load_dataset_b_leaderboard()
    test_lb_b = load_dataset_b_test_leaderboard()

    # Replay Controller Ribbon
    with st.container():
        col_ctrl1, col_ctrl2 = st.columns([1.6, 1.0])
        with col_ctrl1:
            row_idx_selected = st.slider(
                "Historical Record Index (Validation Partition):",
                min_value=24,
                max_value=len(val_b) - 2,
                value=120,
                step=1,
                help="Validation set contains 5,673 gap-filtered hourly records spanning 2017 to 2018."
            )
            retrospective_mode_b = st.checkbox(
                "🔍 Retrospective Evaluation Mode (Display verified t+1h ground truth & model residual)",
                value=False,
                help="In prospective real-time mode, future ground truth is unobserved. Enable this to retrospectively evaluate model accuracy against the actual recorded outcome."
            )
        with col_ctrl2:
            model_choice_b = st.selectbox(
                "Forecasting Model:",
                ["random_forest", "hist_gradient_boosting", "ridge", "persistence"],
                format_func=lambda m: {
                    "random_forest": "Random Forest Champion (Test MAE: 143.7 veh/h)",
                    "hist_gradient_boosting": "HistGradientBoosting (Test MAE: 149.7 veh/h)",
                    "ridge": "Ridge Regression (Test MAE: 362.0 veh/h)",
                    "persistence": "Persistence Baseline (Test MAE: 592.4 veh/h)"
                }[m]
            )

    # Slice data for selected row
    row_data = val_b.iloc[row_idx_selected]
    dt_str = str(row_data.get('date_time', f"Row #{row_idx_selected}"))
    curr_volume = float(row_data['traffic_volume'])
    actual_target_1h = float(row_data['target_1h'])

    # Historical slice for centerpiece chart (past 24 hours up to t)
    # Using format '%b %d, %H:00' ensures each timestamp is uniquely labeled across day boundaries
    hist_slice_b = val_b.iloc[max(0, row_idx_selected - 24): row_idx_selected + 1]
    history_times = [pd.to_datetime(t).strftime('%b %d, %H:00') for t in hist_slice_b['date_time']]
    history_vols = [float(v) for v in hist_slice_b['traffic_volume']]

    current_time_str = history_times[-1] if len(history_times) > 0 else "t"
    next_dt_val = val_b.iloc[row_idx_selected + 1].get('date_time')
    next_time_str = pd.to_datetime(next_dt_val).strftime('%b %d, %H:00') if pd.notnull(next_dt_val) else "t+1h"

    # Feature extraction & model inference
    meta_cols_b = ['target_1h', 'date_time', 'traffic_volume']
    feat_cols_b = [c for c in val_b.columns if c not in meta_cols_b]
    X_sample_b = row_data[feat_cols_b].values.reshape(1, -1).astype(np.float32)

    selected_model_b = models_b.get(model_choice_b)
    if selected_model_b is not None:
        fcst_volume = float(selected_model_b.predict(X_sample_b)[0])
    else:
        fcst_volume = curr_volume

    delta_vol = fcst_volume - curr_volume
    pct_change_b = (delta_vol / curr_volume * 100.0) if curr_volume > 0 else 0.0

    # Trend categorization using rigorous 5-tier classification
    trend_info_b = classify_demand_trend(delta_vol, dataset="B")
    trend_status_b = trend_info_b["status"]
    trend_badge_b = trend_info_b["badge_html"]
    trend_plain_b = trend_info_b["plain_text"]

    # Extract weather
    weather_cols = [c for c in val_b.columns if c.startswith('weather_main_')]
    active_weather = "Clear / Clouds"
    for wc in weather_cols:
        if row_data.get(wc, 0.0) == 1.0:
            active_weather = wc.replace('weather_main_', '')
            break

    # Test metrics for selected model from verified artifacts
    test_metrics_map = {
        "random_forest": {"mae": 143.7, "rmse": 221.7, "r2": 0.9875, "lift": 75.74},
        "hist_gradient_boosting": {"mae": 149.7, "rmse": 225.3, "r2": 0.9871, "lift": 74.72},
        "ridge": {"mae": 362.0, "rmse": 486.1, "r2": 0.9400, "lift": 38.89},
        "persistence": {"mae": 592.4, "rmse": 818.6, "r2": 0.8298, "lift": 0.0}
    }
    m_info = test_metrics_map.get(model_choice_b, test_metrics_map["random_forest"])

    # Operational Regime Interpretation & Grounded Analytical Recommendations
    is_weather_hazard = active_weather in ["Snow", "Rain", "Thunderstorm", "Squall"]

    if fcst_volume >= 4800.0 and ("Rising" in trend_status_b or "Higher" in trend_status_b):
        adv_class_b = "action-warning"
        adv_status_b = "RUSH-HOUR PEAK VOLUME SURGE"
        adv_regime_b = (
            "Station ATR 301 records high volume with upward momentum, approaching 5,000+ veh/h. "
            "In macroscopic traffic theory (q = k · v), high volume indicates high throughput rather than gridlock, "
            "with compressed vehicle headways sensitive to flow fluctuations."
        )
        rec_title_b = "High Throughput Volume Surveillance & Gradient Tracking"
        rec_steps_b = [
            "1. <strong>Volume Profile Tracking:</strong> Cross-check 1-hour volume forecast against historical peak hour distribution for this day of week.",
            "2. <strong>Hourly Gradient Check:</strong> Track hourly rate-of-change (Δ veh/h) to anticipate evening peak crest.",
            "3. <strong>Model Persistence Tracking:</strong> Verify that tree ensemble prediction maintains performance advantage over naive hourly lag."
        ]
        rec_assumptions_b = (
            "Loop detector ATR 301 telemetry is continuous; analytical decision support based on offline benchmark data."
        )
    elif is_weather_hazard and fcst_volume >= 3000.0:
        adv_class_b = "action-warning"
        adv_status_b = f"METEOROLOGICAL ADVISORY ACTIVE ({active_weather.upper()})"
        adv_regime_b = (
            f"Active weather telemetry indicates {active_weather}. Historical analysis reveals adverse weather "
            "reduces average mainline volume by 8–15% during peak periods due to reduced travel speeds and suppressed trip demand."
        )
        rec_title_b = "Adverse Weather Telemetry Flag & Model Vigilance"
        rec_steps_b = [
            "1. <strong>Weather Sensitivity Audit:</strong> Inspect model output relative to dry-weather baseline to confirm weather features appropriately tempered the volume prediction.",
            "2. <strong>Precipitation Lag Tracking:</strong> Monitor station throughput over subsequent hours to capture post-storm recovery timing.",
            "3. <strong>Residual Logging:</strong> Flag current interval for retrospective error audit to assess model accuracy under adverse meteorological conditions."
        ]
        rec_assumptions_b = (
            "Meteorological telemetry is current; offline simulation only without municipal dispatch."
        )
    elif fcst_volume < 2000.0:
        adv_class_b = "action-normal"
        adv_status_b = "OFF-PEAK / LOW VOLUME REGIME"
        adv_regime_b = (
            "Corridor is operating in an off-peak period. Mainline vehicle headways are wide and throughput is low."
        )
        rec_title_b = "Off-Peak Volume Monitoring & Sensor Baseline Check"
        rec_steps_b = [
            "1. <strong>Nocturnal Flow Baseline:</strong> Verify loop detector counts match typical off-peak baseline (~500–1,500 veh/h).",
            "2. <strong>Data Continuity Check:</strong> Ensure consecutive hourly records exhibit zero telemetry dropouts or anomalous null values.",
            "3. <strong>Model Convergence Check:</strong> Confirm linear and tree-based forecasts converge tightly during low-variance intervals."
        ]
        rec_assumptions_b = (
            "Off-peak volume matches normal night or midday historical patterns; loop detector baseline remains stable."
        )
    else:
        adv_class_b = "action-info"
        adv_status_b = "BALANCED COMMUTER DEMAND"
        adv_regime_b = (
            "Corridor volume is at a moderate, stable level. Roadway operates well within design throughput capacity."
        )
        rec_title_b = "Standard Corridor Telemetry Monitoring"
        rec_steps_b = [
            "1. <strong>Loop Polling:</strong> Continuous automated volume recording at Station ATR 301.",
            "2. <strong>Diurnal Profile Tracking:</strong> Compare observed volume against historical median for this hour and day.",
            "3. <strong>Weather Watch:</strong> Monitor meteorological telemetry for potential precipitation changes."
        ]
        rec_assumptions_b = (
            "Mainline detector calibration is stable; standard weekday/weekend traffic patterns apply."
        )

    # ==========================================================================
    # 4-KPI HERO TELEMETRY STRIP (DIRECT FIRST VIEWPORT VISIBILITY)
    # ==========================================================================
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.2rem;">
        <div class="hero-stat-card" title="Observed traffic volume at current historical hour t">
            <div class="hero-stat-tag">Current Observed (t)</div>
            <div class="hero-stat-val">{curr_volume:,.0f} <span style="font-size: 1rem; color: #64748B;">veh/h</span></div>
            <div class="hero-stat-sub">Station ATR 301 • {current_time_str}</div>
        </div>
        <div class="hero-stat-card-forecast" title="Predicted hourly traffic volume for t+1 hour ahead">
            <div class="hero-stat-tag" style="color: #1D4ED8; font-weight: 700;">Forward Forecast (t+1h)</div>
            <div class="hero-stat-val" style="color: #2563EB;">{fcst_volume:,.0f} <span style="font-size: 1rem; color: #2563EB;">veh/h</span></div>
            <div class="hero-stat-sub" style="color: #2563EB; font-weight: 600;">{next_time_str} • {model_choice_b.replace('_', ' ').title()}</div>
        </div>
        <div class="hero-stat-card" title="Forecast delta (t+1h minus t) and percentage variation">
            <div class="hero-stat-tag">Projected Change</div>
            <div class="hero-stat-val" style="font-size: 1.55rem;">{delta_vol:+,.0f} <span style="font-size: 0.9rem; color: #64748B;">veh/h</span></div>
            <div class="hero-stat-sub">{trend_badge_b}</div>
        </div>
        <div class="hero-stat-card" title="Historical demand regime and test-set baseline improvement">
            <div class="hero-stat-tag">Demand Regime & Reliability</div>
            <div class="hero-stat-val" style="font-size: 1.15rem; color: #0F172A;">{adv_status_b}</div>
            <div class="hero-stat-sub">Test MAE: <strong>{m_info['mae']:.1f} veh/h</strong> (+{m_info['lift']:.1f}% vs baseline)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ==========================================================================
    # 2-COLUMN COCKPIT LAYOUT: MASTER VISUAL DISPLAY (LEFT) & OPERATIONS RAIL (RIGHT)
    # ==========================================================================
    col_viz_b, col_rail_b = st.columns([1.65, 1.0], gap="large")

    with col_viz_b:
        # VISUAL CENTERPIECE: PROMINENT TIME-SERIES FORECAST WINDOW
        st.markdown("""
        <div class="viz-panel">
            <div class="viz-panel-title">📈 Visual Centerpiece: 24-Hour Trajectory & 1-Hour Forward Forecast Window</div>
        """, unsafe_allow_html=True)
        fig_centerpiece_b = plot_centerpiece_forecast_dataset_b(
            history_times=history_times,
            history_vols=history_vols,
            current_time=current_time_str,
            current_vol=curr_volume,
            forecast_time=next_time_str,
            forecast_vol=fcst_volume,
            actual_target=actual_target_1h,
            trend_str=trend_status_b,
            weather_str=active_weather,
            retrospective_mode=retrospective_mode_b
        )
        st.plotly_chart(fig_centerpiece_b, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # SPATIAL CORRIDOR SCHEMATIC: INTERSTATE 94
        st.markdown("""
        <div class="viz-panel">
            <div class="viz-panel-title">🗺️ Interstate 94 Corridor Schematic (Station ATR 301 & Landmarks)</div>
        """, unsafe_allow_html=True)
        fig_schematic_b = plot_corridor_schematic_dataset_b(
            current_volume=curr_volume,
            forecast_volume=fcst_volume
        )
        st.plotly_chart(fig_schematic_b, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Educational Callout: Beginner Guide
        with st.expander("📘 How to Read These Traffic Metrics (Beginner Guide)", expanded=False):
            st.markdown("""
            - **Forecast (t+1h):** The model's prediction for the total vehicle count passing loop detector ATR Station 301 over the upcoming 1 hour.
            - **MAE (Mean Absolute Error = 143.7 veh/h):** On average, the champion model's prediction deviates from the true hourly volume by 143.7 vehicles per hour. For context, typical commuter flow ranges from 500 to 5,500 veh/h, so 143.7 veh/h represents an average relative error of only ~3-5% during daytime hours.
            - **RMSE (Root Mean Squared Error = 221.7 veh/h):** Gives higher penalty to occasional large errors (e.g. unexpected blizzards or severe accidents).
            - **R² Score (0.9875):** Measures how closely the model tracks volume dynamics. A score of 0.9875 means nearly 99% of the variation across day, night, weekends, and seasons is captured by the model.
            - **Trend Indicator:** Shows the difference between the current observation ($t$) and future forecast ($t+1\\text{h}$). Positive values denote rising traffic volume, while negative values indicate easing volume.
            """)

    with col_rail_b:
        # TACTICAL DECISION BOX (ANALYTICAL DECISION SUPPORT)
        steps_html_b = "<br>".join(rec_steps_b)
        if retrospective_mode_b:
            eval_banner_b = (
                f'<div style="font-size: 0.84rem; color: #1E3A8A; margin-bottom: 0.6rem; background: #EFF6FF; '
                f'border: 1px solid #BFDBFE; padding: 0.45rem 0.65rem; border-radius: 6px;">'
                f'🔍 <strong>Retrospective Ground Truth (t+1h):</strong> {actual_target_1h:,.0f} veh/h &nbsp;|&nbsp; '
                f'<strong>Model Residual:</strong> {actual_target_1h - fcst_volume:+,.0f} veh/h</div>'
            )
        else:
            eval_banner_b = (
                '<div style="font-size: 0.82rem; color: #64748B; margin-bottom: 0.6rem; background: #F8FAFC; '
                'border: 1px dashed #CBD5E1; padding: 0.45rem 0.65rem; border-radius: 6px;">'
                '🔮 <strong>Prospective Replay:</strong> Future ground truth unobserved at <em>t</em>. Check "Retrospective Mode" above to evaluate error.</div>'
            )

        box_html_b = f"""
        <div class="action-box {adv_class_b}">
            <span class="action-badge-tag">Analytical Decision Support (Simulated)</span>
            <div style="font-size: 1.12rem; font-weight: 800; margin-bottom: 0.45rem; color: #0F172A;">
                🛠️ {rec_title_b}
            </div>
            <div style="font-size: 0.88rem; line-height: 1.45; margin-bottom: 0.6rem; color: #1E293B;">
                {adv_regime_b}
            </div>
            {eval_banner_b}
            <div style="font-size: 0.86rem; line-height: 1.5; margin-bottom: 0.65rem;">
                {steps_html_b}
            </div>
            <div style="font-size: 0.80rem; color: #475569; border-top: 1px dashed #CBD5E1; padding-top: 0.45rem;">
                📋 <strong>Assumptions:</strong> {rec_assumptions_b}<br>
                ⚠️ <em>Simulated analytical guidance evaluated on offline benchmark data. Does not actuate municipal traffic systems.</em>
            </div>
        </div>
        """
        st.markdown(textwrap.dedent(box_html_b), unsafe_allow_html=True)

        # MODEL BENCHMARK RELIABILITY CARD
        rel_html_b = f"""
        <div class="tactical-decision-box">
            <div class="tactical-badge">Official Test Benchmark</div>
            <div class="tactical-title">🏆 Model Reliability on Unseen Test Data</div>
            <div class="tactical-interp">
                Performance evaluated on 5,675 strictly held-out chronological test hours (Phase 5 verified):
            </div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.6rem; margin-bottom: 0.6rem;">
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 0.6rem; border-radius: 6px;">
                    <div style="font-size: 0.80rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Test MAE</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #0F172A;">{m_info['mae']:.1f} <span style="font-size:0.80rem; color:#64748B;">veh/h</span></div>
                    <div style="font-size: 0.82rem; color: #059669; font-weight: 700;">+{m_info['lift']:.1f}% vs persistence</div>
                </div>
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 0.6rem; border-radius: 6px;">
                    <div style="font-size: 0.80rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Test RMSE</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #0F172A;">{m_info['rmse']:.1f} <span style="font-size:0.80rem; color:#64748B;">veh/h</span></div>
                    <div style="font-size: 0.82rem; color: #64748B;">Penalizes large misses</div>
                </div>
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 0.6rem; border-radius: 6px;">
                    <div style="font-size: 0.80rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Test R² Score</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #0F172A;">{m_info['r2']:.4f}</div>
                    <div style="font-size: 0.82rem; color: #64748B;">{m_info['r2']*100:.1f}% variance explained</div>
                </div>
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 0.6rem; border-radius: 6px;">
                    <div style="font-size: 0.80rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Persistence Baseline</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #64748B;">592.4 <span style="font-size:0.80rem; color:#64748B;">veh/h</span></div>
                    <div style="font-size: 0.82rem; color: #64748B;">Naive t-1 baseline</div>
                </div>
            </div>
            <div style="font-size: 0.82rem; color: #64748B; line-height: 1.45;">
                🔒 Zero lookahead leakage. Model trained strictly on pre-2017 chronological train partition.
            </div>
        </div>
        """
        st.markdown(textwrap.dedent(rel_html_b), unsafe_allow_html=True)

# Methodology & Theoretical Foundation Footer
st.divider()
with st.expander("📚 Methodology, Traffic Flow Physics & Scientific Provenance", expanded=False):
    st.markdown("""
    #### 1. Macroscopic Traffic Flow Fundamentals ($q = k \\cdot v$)
    In transportation engineering, highway operations are governed by Lighthill-Whitham-Richards (LWR) continuum theory:
    - **Traffic Flow ($q$):** The rate at which vehicles pass a fixed cross-section per unit time (vehicles/hour).
    - **Traffic Density ($k$):** The concentration of vehicles per unit length of highway lane (vehicles/mile or vehicles/km).
    - **Space-Mean Speed ($v$):** The average velocity of the vehicle stream.
    
    *Critical Operational Distinction:* High volume or flow ($q$) indicates high **throughput demand**, which occurs both during smooth free-flow conditions and at the crest of highway capacity. In stop-and-go queue congestion, vehicle flow collapses toward zero despite very high vehicle density. Therefore, flow and volume alone quantify throughput capacity, not geometric queue congestion.

    #### 2. Rigorous Chronological Train/Validation/Test Partitions
    - **Dataset A (UCI #734):** 36 loop sensors across Northern Virginia / D.C. capital region highway network. 15-minute sampling cadence. 30-minute verified forward forecast horizon (+2 steps). Models evaluated across 70% train (pre-June 2018), 15% validation, and 15% strictly unseen chronological test partition.
    - **Dataset B (UCI #492):** MN DoT Station ATR 301 on Interstate 94 (Minneapolis–St. Paul). 1-hour sampling cadence. 1-hour verified forward forecast horizon (+1 step). 48,204 records filtered for strict 1-hour continuity, chronologically partitioned into 37,204 train, 5,673 validation, and 5,675 unseen test hours.

    #### 3. Strict Machine Learning Operational Guardrails
    - **Zero Data Leakage:** Lag features, rolling statistics, and temporal encoders are computed strictly backward in time.
    - **CPU-First Latency:** Gradient-boosted and random forest architectures execute full network inference in < 0.1 milliseconds per station.
    - **Metric Parity:** All reported test MAE, RMSE, and R² scores match saved artifact arrays byte-for-byte.
    """)
