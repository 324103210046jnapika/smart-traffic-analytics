"""Streamlit Module 3: Forecasting & Scenario Replay Studio.

Interactive corridor simulation, dynamic inference replay, residual error analysis,
and grounded operational advisories evaluated across arbitrary time horizons.
"""

import os
import sys
import textwrap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add project root to sys.path so modules resolve cleanly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.utils.styles import get_shared_css, render_header_bar
from app.utils.data_loader import (
    load_dataset_a_validation,
    load_dataset_b_validation,
    load_dataset_a_test,
    load_dataset_b_test,
    load_dataset_a_models,
    load_dataset_b_models,
    load_dataset_a_adjacency
)
from app.utils.plots import (
    classify_demand_trend,
    plot_forecasting_studio_timeseries,
    plot_residual_distribution_histogram
)

st.set_page_config(
    page_title="Forecasting & Scenario Replay Studio | Smart Traffic TOC",
    page_icon="🔮",
    layout="wide"
)

# Apply shared design tokens and enterprise CSS
st.markdown(get_shared_css(), unsafe_allow_html=True)

# Top Console Header Bar
st.markdown(
    render_header_bar(
        title="🔮 Forecasting & Scenario Replay Studio",
        subtitle="Dynamic Corridor Inference Replay, Retrospective Ground Truth Auditing & Residual Error Diagnostics"
    ),
    unsafe_allow_html=True
)

# Operational Scope Banner
st.markdown("""
<div style="background-color: #EFF6FF; border-left: 4px solid #3B82F6; padding: 0.55rem 0.95rem; border-radius: 6px; margin-bottom: 0.9rem; font-size: 0.82rem; color: #1E3A8A; line-height: 1.45;">
    🛡️ <strong>Scenario Replay Mode:</strong> All model inference is executed dynamically on pre-trained models using saved historical benchmark data (UCI #734 & UCI #492).
    In traffic flow theory (<em>q = k · v</em>), high flow quantifies throughput volume demand, not physical congestion.
</div>
""", unsafe_allow_html=True)

# 1. Traffic Domain Selector
st.markdown("### 🌐 **Select Forecasting Domain & Corridor**")
dataset_choice = st.radio(
    "Active Traffic Corridor:",
    [
        "Dataset A: Northern Virginia / Washington D.C. Capital Region (36 Highway Sensors, 30-Min Horizon, Normalized Flow [0, 1])",
        "Dataset B: Interstate 94 Corridor (MN DoT ATR Station 301, 1-Hour Horizon, Vehicles / Hour)"
    ],
    index=0,
    horizontal=True,
    help="Both datasets represent distinct, decoupled traffic forecasting domains with verified horizons and units."
)

st.divider()

# ==============================================================================
# DATASET A: NORTHERN VIRGINIA / D.C. 36-SENSOR NETWORK
# ==============================================================================
if "Dataset A" in dataset_choice:
    st.subheader("🛰️ Dataset A: Station-Level Flow Forecast (30 Minutes Ahead)")
    st.caption("Forecast Quantity: Normalized Traffic Flow [0, 1] | Verified Horizon: 30 Minutes (+2 observation steps at 15-min cadence)")

    models_a = load_dataset_a_models()
    adj_mat = load_dataset_a_adjacency()

    # Evaluation Controls Card
    with st.container():
        st.markdown("""
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.9rem 1.1rem; margin-bottom: 1rem; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
            <div style="font-size: 0.82rem; font-weight: 800; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.6rem;">
                ⚙️ Evaluation Partition & Corridor Controls
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([1.1, 1.2, 1.4, 1.3])

        with col_ctrl1:
            partition_a = st.selectbox(
                "Evaluation Partition:",
                options=["Validation Set (Standard)", "Test Set (Strictly Held-Out)"],
                index=0,
                help="Validation Set was used for model tuning. Test Set is the strictly held-out final benchmark partition."
            )

        with col_ctrl2:
            sensor_degrees = np.sum(adj_mat, axis=1)
            selected_sensor = st.selectbox(
                "Highway Sensor:",
                options=list(range(36)),
                index=14,
                format_func=lambda s: f"Sensor S{s:02d} ({int(sensor_degrees[s])} links)"
            )

        with col_ctrl3:
            model_name_map = {
                "hist_gradient_boosting": "HistGradientBoosting (Champion, Test MAE: 0.0469)",
                "random_forest": "Random Forest (Test MAE: 0.0487)",
                "ridge": "Ridge Regression (Test MAE: 0.0543)",
                "persistence": "Persistence Baseline (Benchmark, Test MAE: 0.0577)"
            }
            selected_model_slug = st.selectbox(
                "Forecasting Model:",
                options=list(model_name_map.keys()),
                format_func=lambda k: model_name_map[k]
            )

        with col_ctrl4:
            eval_mode = st.radio(
                "Operational Evaluation Mode:",
                options=["Retrospective Audit Mode", "Prospective Live Mode"],
                index=0,
                help="Retrospective reveals ground truth targets and residual error bars. Prospective hides future ground truth."
            )
            is_retrospective = (eval_mode == "Retrospective Audit Mode")

    # Load appropriate partition
    if "Test" in partition_a:
        df_a = load_dataset_a_test()
        part_tag = "Held-Out Test Partition (Steps 1261–2100)"
    else:
        df_a = load_dataset_a_validation()
        part_tag = "Validation Partition (Steps 1000–1260)"

    # Slider for time window
    min_step = int(df_a['step_idx'].min())
    max_step = int(df_a['step_idx'].max())
    default_start = min_step + 10
    default_end = min(min_step + 90, max_step)

    step_range = st.slider(
        f"Select Continuous Operational Window ({part_tag}):",
        min_value=min_step,
        max_value=max_step,
        value=(default_start, default_end),
        step=5,
        help="Selects a continuous sequence of 15-minute telemetry intervals for model inference."
    )

    # Filter data for chosen sensor and time window
    sensor_mask = (df_a['sensor_idx'] == selected_sensor)
    sensor_df = df_a[sensor_mask].sort_values('step_idx').reset_index(drop=True)
    window_df = sensor_df[(sensor_df['step_idx'] >= step_range[0]) & (sensor_df['step_idx'] <= step_range[1])].copy().reset_index(drop=True)

    if len(window_df) > 0:
        feat_cols = [f"feat_{i:02d}" for i in range(48)]
        X_slice = window_df[feat_cols].values.astype(np.float32)
        y_actual = window_df['target'].values.astype(np.float32)
        y_persist = window_df['persistence_pred'].values.astype(np.float32)

        # Execute dynamic model inference
        model = models_a[selected_model_slug]
        y_pred = model.predict(X_slice).astype(np.float32)

        # Dynamic Window Performance Metrics
        mae_slice = float(np.mean(np.abs(y_actual - y_pred)))
        rmse_slice = float(np.sqrt(np.mean((y_actual - y_pred) ** 2)))
        mae_persist = float(np.mean(np.abs(y_actual - y_persist)))
        gain_slice = ((mae_persist - mae_slice) / (mae_persist + 1e-8)) * 100.0

        # Endpoint trajectory calculations
        latest_obs = float(y_persist[-1])
        latest_fcst = float(y_pred[-1])
        delta_latest = latest_fcst - latest_obs
        trend_info = classify_demand_trend(delta_latest, dataset="A")

        # 4-KPI Metric Ribbon
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

        with kpi_col1:
            st.markdown(textwrap.dedent(f"""
            <div class="hero-stat-card-forecast" title="Mean Absolute Error of the selected model across this specific operational window">
                <div class="hero-stat-tag" style="color: #1D4ED8;">Window MAE ({selected_model_slug.replace('_', ' ').title()})</div>
                <div class="hero-stat-val" style="color: #2563EB;">{mae_slice:.4f}</div>
                <div class="hero-stat-sub" style="color: #059669; font-weight: 700;">{'+' if gain_slice >= 0 else ''}{gain_slice:.1f}% vs Persistence ({mae_persist:.4f})</div>
            </div>
            """), unsafe_allow_html=True)

        with kpi_col2:
            st.markdown(textwrap.dedent(f"""
            <div class="hero-stat-card" title="Root Mean Squared Error penalizing large momentary deviations across the window">
                <div class="hero-stat-tag">Window RMSE (Dispersion)</div>
                <div class="hero-stat-val">{rmse_slice:.4f}</div>
                <div class="hero-stat-sub">Quadratic penalty on peak misses</div>
            </div>
            """), unsafe_allow_html=True)

        with kpi_col3:
            st.markdown(textwrap.dedent(f"""
            <div class="hero-stat-card" title="Benchmark error achieved by naively carrying forward current observed flow to t+30 min">
                <div class="hero-stat-tag">Baseline MAE (Persistence)</div>
                <div class="hero-stat-val">{mae_persist:.4f}</div>
                <div class="hero-stat-sub">Naïve benchmark (no model)</div>
            </div>
            """), unsafe_allow_html=True)

        with kpi_col4:
            st.markdown(textwrap.dedent(f"""
            <div class="hero-stat-card" title="Forward flow trajectory shift from window endpoint to t+30 minutes ahead">
                <div class="hero-stat-tag">Forward Horizon Trend (+30m)</div>
                <div class="hero-stat-val" style="font-size: 1.45rem;">{trend_info['badge_html']}</div>
                <div class="hero-stat-sub">Endpoint shift: {delta_latest:+.3f} normalized flow</div>
            </div>
            """), unsafe_allow_html=True)

        st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)

        # Grounded Analytical Advisory Ribbon
        if latest_fcst >= 0.65 and trend_info['status'] in ["Rising", "Slightly Higher"]:
            adv_banner_a = (
                f"Sensor S{selected_sensor:02d} projected flow ({latest_fcst:.3f}) exceeds upper throughput quartile (0.65) with rising trajectory (+{delta_latest:.3f}). "
                "Simulated decision support: Suggest monitoring upstream on-ramp queues and corridor merge zones to detect flow saturation."
            )
            adv_class_a = "action-warning"
        elif trend_info['status'] in ["Easing", "Slightly Lower"] and latest_obs >= 0.50:
            adv_banner_a = (
                f"Flow demand on Sensor S{selected_sensor:02d} is easing ({delta_latest:.3f}) from peak levels ({latest_obs:.3f}). "
                "Simulated decision support: Corridor throughput demand contracting; suggest transitioning toward routine baseline monitoring posture."
            )
            adv_class_a = "action-normal"
        else:
            adv_banner_a = (
                f"Sensor S{selected_sensor:02d} operating steadily within capacity ({latest_fcst:.3f}, delta: {delta_latest:+.3f}). "
                "Simulated decision support: Normal highway operating envelope. Routine background surveillance active."
            )
            adv_class_a = "action-info"

        st.markdown(textwrap.dedent(f"""
        <div class="action-box {adv_class_a}">
            <span class="action-badge-tag">HISTORICAL-DATA-BASED ANALYTICAL SUGGESTION (Simulated Decision Support — No Municipal Control Actuation)</span><br>
            <div style="font-size: 0.92rem; font-weight: 600; line-height: 1.5; margin-top: 0.35rem;">
                {adv_banner_a}
            </div>
            <div class="sys-scope-note">
                ⚠️ <em>Offline simulation advisory generated from historical {part_tag.lower()}. Does not actuate real-world highway signals or signage.</em>
            </div>
        </div>
        """), unsafe_allow_html=True)

        # Primary Time Series & Residual Chart
        x_steps = [f"Step {s}" for s in window_df['step_idx'].values]
        fig_studio_a = plot_forecasting_studio_timeseries(
            x_vals=x_steps,
            y_actual=y_actual,
            y_pred=y_pred,
            y_baseline=y_persist,
            retrospective_mode=is_retrospective,
            title=f"Sensor S{selected_sensor:02d}: Forecast vs Ground Truth ({selected_model_slug.replace('_', ' ').title()})",
            y_label="Normalized Flow [0, 1]",
            x_label="Observation Time Step (15-min intervals)",
            horizon_tag="+30 min",
            unit_str=""
        )
        st.plotly_chart(fig_studio_a, use_container_width=True)

        # Dedicated Residual Analysis Panel in Retrospective Mode
        if is_retrospective:
            residuals_a = y_pred - y_actual
            mean_bias_a = float(np.mean(residuals_a))
            max_abs_err_a = float(np.max(np.abs(residuals_a)))
            n_over_a = int(np.sum(residuals_a > 0))
            n_under_a = int(np.sum(residuals_a < 0))
            pct_over_a = (n_over_a / len(residuals_a)) * 100.0
            pct_under_a = (n_under_a / len(residuals_a)) * 100.0

            st.markdown("""
            <div class="viz-panel">
                <div class="viz-panel-title">
                    <span>🔬 Residual Error & Model Reliability Breakdown</span>
                    <span style="font-size: 0.78rem; font-weight: 600; color: #64748B; margin-left: auto;">
                        Evaluated across active window ({len(window_df)} steps)
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_res_metrics, col_res_hist = st.columns([1.1, 1.9], gap="large")

            with col_res_metrics:
                st.markdown(textwrap.dedent(f"""
                <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1.1rem; height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div style="font-size: 0.82rem; font-weight: 800; color: #475569; text-transform: uppercase; margin-bottom: 0.8rem;">
                            Error Direction & Bias Metrics
                        </div>
                        <div style="margin-bottom: 0.75rem;">
                            <div style="font-size: 0.75rem; color: #64748B; font-weight: 700;">MEAN SIGNED ERROR (BIAS)</div>
                            <div style="font-size: 1.45rem; font-weight: 800; color: {'#EA580C' if mean_bias_a > 0 else '#6366F1'};">
                                {mean_bias_a:+.4f}
                            </div>
                            <div style="font-size: 0.78rem; color: #64748B;">
                                {'Model slightly overpredicts flow on average' if mean_bias_a > 0 else 'Model slightly underpredicts flow on average'}
                            </div>
                        </div>
                        <div style="margin-bottom: 0.75rem;">
                            <div style="font-size: 0.75rem; color: #64748B; font-weight: 700;">MAXIMUM ABSOLUTE ERROR</div>
                            <div style="font-size: 1.45rem; font-weight: 800; color: #0F172A;">
                                {max_abs_err_a:.4f}
                            </div>
                            <div style="font-size: 0.78rem; color: #64748B;">Worst single step deviation in active window</div>
                        </div>
                    </div>
                    <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 6px; padding: 0.65rem 0.85rem; font-size: 0.8rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                            <span style="color: #EA580C; font-weight: 700;">⚠️ Overpredictions (Fcst > Actual):</span>
                            <strong>{n_over_a} ({pct_over_a:.1f}%)</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #6366F1; font-weight: 700;">📉 Underpredictions (Fcst < Actual):</span>
                            <strong>{n_under_a} ({pct_under_a:.1f}%)</strong>
                        </div>
                    </div>
                </div>
                """), unsafe_allow_html=True)

            with col_res_hist:
                fig_hist_a = plot_residual_distribution_histogram(
                    residuals=residuals_a,
                    unit_str="flow",
                    title=f"Residual Error Distribution — Sensor S{selected_sensor:02d} ({selected_model_slug.replace('_', ' ').title()})"
                )
                st.plotly_chart(fig_hist_a, use_container_width=True)

    else:
        st.warning("No data found for the selected sensor and step range.")


# ==============================================================================
# DATASET B: INTERSTATE 94 ATR STATION 301
# ==============================================================================
else:
    st.subheader("🚗 Dataset B: Interstate 94 Hourly Traffic Volume Forecast (1 Hour Ahead)")
    st.caption("Forecast Quantity: Hourly Traffic Volume (Vehicles / Hour) | Verified Horizon: 1 Hour Ahead | Scope: MN DoT ATR Station 301")

    models_b = load_dataset_b_models()

    # Evaluation Controls Card
    with st.container():
        st.markdown("""
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.9rem 1.1rem; margin-bottom: 1rem; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
            <div style="font-size: 0.82rem; font-weight: 800; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.6rem;">
                ⚙️ Evaluation Partition & Corridor Controls (Station ATR 301 Westbound)
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_ctrl1_b, col_ctrl2_b, col_ctrl3_b = st.columns([1.2, 1.5, 1.3])

        with col_ctrl1_b:
            partition_b = st.selectbox(
                "Evaluation Partition:",
                options=["Validation Set (Standard)", "Test Set (Strictly Held-Out)"],
                index=0,
                help="Validation Set was used for model tuning. Test Set is the strictly held-out final benchmark partition."
            )

        with col_ctrl2_b:
            selected_model_slug_b = st.selectbox(
                "Forecasting Model:",
                options=["random_forest", "hist_gradient_boosting", "ridge", "persistence"],
                format_func=lambda k: {
                    "random_forest": "Random Forest (Champion, Test MAE: 143.7 veh/h)",
                    "hist_gradient_boosting": "HistGradientBoosting (Test MAE: 149.7 veh/h)",
                    "ridge": "Ridge Regression (Test MAE: 362.0 veh/h)",
                    "persistence": "Persistence Baseline (Benchmark, Test MAE: 592.4 veh/h)"
                }[k]
            )

        with col_ctrl3_b:
            eval_mode_b = st.radio(
                "Operational Evaluation Mode:",
                options=["Retrospective Audit Mode", "Prospective Live Mode"],
                index=0,
                help="Retrospective reveals ground truth targets and residual error bars. Prospective hides future ground truth."
            )
            is_retrospective_b = (eval_mode_b == "Retrospective Audit Mode")

    # Load appropriate partition
    if "Test" in partition_b:
        df_b = load_dataset_b_test()
        part_tag_b = "Held-Out Test Partition (5,675 continuous hourly records)"
    else:
        df_b = load_dataset_b_validation()
        part_tag_b = "Validation Partition (5,673 continuous hourly records)"

    # Continuous 72-hour operational window slider
    max_start_b = max(0, len(df_b) - 75)
    window_start_idx = st.slider(
        f"Select Continuous 72-Hour Operational Window ({part_tag_b}):",
        min_value=0,
        max_value=max_start_b,
        value=min(120, max_start_b),
        step=12,
        help="Displays a continuous 3-day (72-hour) operational sequence on Interstate 94 at 1-hour cadence."
    )

    window_df_b = df_b.iloc[window_start_idx : window_start_idx + 72].copy().reset_index(drop=True)

    if len(window_df_b) > 0:
        meta_cols_b = ['target_1h', 'date_time', 'traffic_volume']
        feat_cols_b = [c for c in df_b.columns if c not in meta_cols_b]
        X_slice_b = window_df_b[feat_cols_b].values.astype(np.float32)
        y_actual_b = window_df_b['target_1h'].values.astype(np.float32)
        y_persist_b = window_df_b['traffic_volume'].values.astype(np.float32)

        # Execute dynamic model inference
        model_b = models_b[selected_model_slug_b]
        y_pred_b = model_b.predict(X_slice_b).astype(np.float32)

        # Dynamic Window Performance Metrics
        mae_slice_b = float(np.mean(np.abs(y_actual_b - y_pred_b)))
        rmse_slice_b = float(np.sqrt(np.mean((y_actual_b - y_pred_b) ** 2)))
        mae_persist_b = float(np.mean(np.abs(y_actual_b - y_persist_b)))
        gain_slice_b = ((mae_persist_b - mae_slice_b) / (mae_persist_b + 1e-8)) * 100.0

        # Endpoint trajectory calculations
        latest_vol_b = float(y_persist_b[-1])
        latest_fcst_b = float(y_pred_b[-1])
        delta_vol_b = latest_fcst_b - latest_vol_b
        trend_info_b = classify_demand_trend(delta_vol_b, dataset="B")

        # 4-KPI Metric Ribbon
        kpi_col1_b, kpi_col2_b, kpi_col3_b, kpi_col4_b = st.columns(4)

        with kpi_col1_b:
            st.markdown(textwrap.dedent(f"""
            <div class="hero-stat-card-forecast" title="Mean Absolute Error of the selected model across this 72-hour operational window">
                <div class="hero-stat-tag" style="color: #1D4ED8;">Window MAE ({selected_model_slug_b.replace('_', ' ').title()})</div>
                <div class="hero-stat-val" style="color: #2563EB;">{mae_slice_b:,.1f} <span style="font-size: 0.9rem; font-weight: 600;">veh/h</span></div>
                <div class="hero-stat-sub" style="color: #059669; font-weight: 700;">{'+' if gain_slice_b >= 0 else ''}{gain_slice_b:.1f}% vs Persistence ({mae_persist_b:,.1f})</div>
            </div>
            """), unsafe_allow_html=True)

        with kpi_col2_b:
            st.markdown(textwrap.dedent(f"""
            <div class="hero-stat-card" title="Root Mean Squared Error penalizing large momentary deviations across the 72-hour window">
                <div class="hero-stat-tag">Window RMSE (Dispersion)</div>
                <div class="hero-stat-val">{rmse_slice_b:,.1f} <span style="font-size: 0.9rem; font-weight: 600;">veh/h</span></div>
                <div class="hero-stat-sub">Quadratic penalty on large errors</div>
            </div>
            """), unsafe_allow_html=True)

        with kpi_col3:
            st.markdown(textwrap.dedent(f"""
            <div class="hero-stat-card" title="Benchmark error achieved by naively carrying forward current observed hourly volume to t+1 hour">
                <div class="hero-stat-tag">Baseline MAE (Persistence)</div>
                <div class="hero-stat-val">{mae_persist_b:,.1f} <span style="font-size: 0.9rem; font-weight: 600;">veh/h</span></div>
                <div class="hero-stat-sub">Naïve benchmark (no model)</div>
            </div>
            """), unsafe_allow_html=True)

        with kpi_col4_b:
            st.markdown(textwrap.dedent(f"""
            <div class="hero-stat-card" title="Forward volume trajectory shift from window endpoint to t+1 hour ahead">
                <div class="hero-stat-tag">Forward Horizon Trend (+1h)</div>
                <div class="hero-stat-val" style="font-size: 1.45rem;">{trend_info_b['badge_html']}</div>
                <div class="hero-stat-sub">Endpoint shift: {delta_vol_b:+,.0f} veh/h</div>
            </div>
            """), unsafe_allow_html=True)

        st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)

        # Grounded Analytical Advisory Ribbon
        if latest_fcst_b >= 4800.0 and trend_info_b['status'] in ["Rising", "Slightly Higher"]:
            adv_banner_b = (
                f"1-hour forecast volume ({latest_fcst_b:,.0f} veh/h) approaches corridor throughput capacity with rising commuter surge (+{delta_vol_b:,.0f} veh/h). "
                "Simulated decision support: Suggest staging incident surveillance patrols near known merge zones between Minneapolis and St. Paul."
            )
            adv_class_b = "action-warning"
        elif latest_fcst_b < 2000.0:
            adv_banner_b = (
                f"Forecast volume is low ({latest_fcst_b:,.0f} veh/h). "
                "Simulated decision support: Off-peak free-flow regime; favorable operational window for scheduled maintenance or non-emergency lane closures."
            )
            adv_class_b = "action-normal"
        else:
            adv_banner_b = (
                f"Corridor volume is steady at nominal throughput ({latest_fcst_b:,.0f} veh/h, delta: {delta_vol_b:+,.0f} veh/h). "
                "Simulated decision support: Routine commuter volume envelope. Standard background surveillance active."
            )
            adv_class_b = "action-info"

        st.markdown(textwrap.dedent(f"""
        <div class="action-box {adv_class_b}">
            <span class="action-badge-tag">HISTORICAL-DATA-BASED ANALYTICAL SUGGESTION (Simulated Decision Support — No Municipal Control Actuation)</span><br>
            <div style="font-size: 0.92rem; font-weight: 600; line-height: 1.5; margin-top: 0.35rem;">
                {adv_banner_b}
            </div>
            <div class="sys-scope-note">
                ⚠️ <em>Offline simulation advisory generated from historical {part_tag_b.lower()}. Does not actuate real-world highway signals or signage.</em>
            </div>
        </div>
        """), unsafe_allow_html=True)

        # Primary Time Series & Residual Chart
        x_times_b = pd.to_datetime(window_df_b['date_time']).dt.strftime('%b %d %H:00').values
        fig_studio_b = plot_forecasting_studio_timeseries(
            x_vals=x_times_b,
            y_actual=y_actual_b,
            y_pred=y_pred_b,
            y_baseline=y_persist_b,
            retrospective_mode=is_retrospective_b,
            title=f"I-94 Corridor (Station ATR 301) 72-Hour Traffic Volume Forecast vs Ground Truth ({selected_model_slug_b.replace('_', ' ').title()})",
            y_label="Traffic Volume (Vehicles / Hour)",
            x_label="Hourly Timestamp (Continuous Operational Window)",
            horizon_tag="+1 hour",
            unit_str="veh/h"
        )
        st.plotly_chart(fig_studio_b, use_container_width=True)

        # Dedicated Residual Analysis Panel in Retrospective Mode
        if is_retrospective_b:
            residuals_b = y_pred_b - y_actual_b
            mean_bias_b = float(np.mean(residuals_b))
            max_abs_err_b = float(np.max(np.abs(residuals_b)))
            n_over_b = int(np.sum(residuals_b > 0))
            n_under_b = int(np.sum(residuals_b < 0))
            pct_over_b = (n_over_b / len(residuals_b)) * 100.0
            pct_under_b = (n_under_b / len(residuals_b)) * 100.0

            st.markdown("""
            <div class="viz-panel">
                <div class="viz-panel-title">
                    <span>🔬 Residual Error & Model Reliability Breakdown</span>
                    <span style="font-size: 0.78rem; font-weight: 600; color: #64748B; margin-left: auto;">
                        Evaluated across active 72-hour operational window (72 samples)
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_res_metrics_b, col_res_hist_b = st.columns([1.1, 1.9], gap="large")

            with col_res_metrics_b:
                st.markdown(textwrap.dedent(f"""
                <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1.1rem; height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div style="font-size: 0.82rem; font-weight: 800; color: #475569; text-transform: uppercase; margin-bottom: 0.8rem;">
                            Error Direction & Bias Metrics
                        </div>
                        <div style="margin-bottom: 0.75rem;">
                            <div style="font-size: 0.75rem; color: #64748B; font-weight: 700;">MEAN SIGNED ERROR (BIAS)</div>
                            <div style="font-size: 1.45rem; font-weight: 800; color: {'#EA580C' if mean_bias_b > 0 else '#6366F1'};">
                                {mean_bias_b:+,.1f} <span style="font-size: 0.85rem; font-weight: 600;">veh/h</span>
                            </div>
                            <div style="font-size: 0.78rem; color: #64748B;">
                                {'Model slightly overpredicts volume on average' if mean_bias_b > 0 else 'Model slightly underpredicts volume on average'}
                            </div>
                        </div>
                        <div style="margin-bottom: 0.75rem;">
                            <div style="font-size: 0.75rem; color: #64748B; font-weight: 700;">MAXIMUM ABSOLUTE ERROR</div>
                            <div style="font-size: 1.45rem; font-weight: 800; color: #0F172A;">
                                {max_abs_err_b:,.0f} <span style="font-size: 0.85rem; font-weight: 600;">veh/h</span>
                            </div>
                            <div style="font-size: 0.78rem; color: #64748B;">Worst single hour deviation in active window</div>
                        </div>
                    </div>
                    <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 6px; padding: 0.65rem 0.85rem; font-size: 0.8rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                            <span style="color: #EA580C; font-weight: 700;">⚠️ Overpredictions (Fcst > Actual):</span>
                            <strong>{n_over_b} ({pct_over_b:.1f}%)</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #6366F1; font-weight: 700;">📉 Underpredictions (Fcst < Actual):</span>
                            <strong>{n_under_b} ({pct_under_b:.1f}%)</strong>
                        </div>
                    </div>
                </div>
                """), unsafe_allow_html=True)

            with col_res_hist_b:
                fig_hist_b = plot_residual_distribution_histogram(
                    residuals=residuals_b,
                    unit_str="veh/h",
                    title=f"Residual Error Distribution — Interstate 94 ({selected_model_slug_b.replace('_', ' ').title()})"
                )
                st.plotly_chart(fig_hist_b, use_container_width=True)

    else:
        st.warning("No data found for the selected continuous window.")
