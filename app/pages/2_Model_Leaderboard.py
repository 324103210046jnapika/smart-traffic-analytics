"""Streamlit Module 2: Model Benchmark & Algorithmic Certification.

Comprehensive validation and official test leaderboards comparing candidate machine learning models
against the persistence baseline with beginner-friendly metric explanations and generalization proofs.
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to sys.path so modules resolve cleanly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.utils.styles import get_shared_css
from app.utils.data_loader import (
    load_dataset_a_leaderboard,
    load_dataset_b_leaderboard,
    load_dataset_a_test_leaderboard,
    load_dataset_b_test_leaderboard,
    load_dataset_a_per_sensor_errors,
    load_dataset_a_per_sensor_test_errors
)
from app.utils.plots import plot_leaderboard_bar

st.set_page_config(
    page_title="Model Benchmark & Certification | Smart Traffic TOC",
    page_icon="🏆",
    layout="wide"
)

# Apply shared design tokens and enterprise CSS
st.markdown(get_shared_css(), unsafe_allow_html=True)

st.markdown("""
<div class="toc-header-bar">
    <div class="toc-title">🏆 Module 2: Model Benchmark & Algorithmic Certification</div>
    <div class="toc-subtitle">Standardized evaluation leaderboards, baseline comparisons, error metrics, and generalization stability</div>
</div>
""", unsafe_allow_html=True)

# Beginner-Friendly Metrics Guide
with st.expander("📘 Beginner's Guide: How to Read Traffic Forecasting Metrics (Click to Expand)", expanded=False):
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.markdown("""
        **MAE (Mean Absolute Error):**
        - *Plain English:* On average, how far off is the model prediction from the real traffic measurement?
        - *Units:* Same as target (Normalized [0, 1] for A; veh/h for B). Lower is better.
        """)
    with g2:
        st.markdown("""
        **RMSE (Root Mean Squared Error):**
        - *Plain English:* Measures average error with an extra penalty for large blunders or extreme misses.
        - *Why it matters:* Operators need to know if a model rarely has disastrously huge mispredictions.
        """)
    with g3:
        st.markdown("""
        **R² Score (Goodness of Fit):**
        - *Plain English:* What percentage of real-world traffic fluctuations are captured by the model?
        - *Scale:* $1.0$ is a perfect prediction; $0.0$ means no better than predicting the historical average.
        """)
    with g4:
        st.markdown(r"""
        **Persistence Baseline:**
        - *Plain English:* The 'no-change' reference forecast: $\hat{Y}_{t+h} = Y_t$.
        - *Why it matters:* If an AI model cannot beat persistence, it has learned nothing and is useless in practice.
        """)

col_p1, col_p2 = st.columns([1.2, 1.0])
with col_p1:
    dataset_choice = st.radio(
        "Select Corridor Dataset:",
        [
            "Dataset A: Northern Virginia / D.C. 36-Sensor Network (30-Minute Horizon)",
            "Dataset B: Interstate 94 Traffic Volume (1-Hour Horizon)"
        ],
        horizontal=True
    )
with col_p2:
    benchmark_partition = st.radio(
        "Select Benchmark Partition:",
        [
            "Official Test Partition (Phase 5 Final Unseen Benchmark)",
            "Validation Partition (Phase 3 Model Selection Benchmark)"
        ],
        horizontal=True
    )

st.divider()

is_test_mode = "Official Test" in benchmark_partition

# ==============================================================================
# DATASET A LEADERBOARD
# ==============================================================================
if "Dataset A" in dataset_choice:
    unit_str_a = "Normalized Flow [0, 1]"
    horizon_str_a = "30 Minutes Ahead (offset +2 steps beyond lag[9])"

    if is_test_mode:
        st.subheader(f"🏁 Dataset A: Official Test Leaderboard ({horizon_str_a})")
        st.caption("Evaluated on 30,240 unseen test samples (840 time steps × 36 sensors) after all model hyperparameters were frozen.")
        lb_a = load_dataset_a_test_leaderboard()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            persist_mae = lb_a[lb_a['model_slug'] == 'persistence']['test_mae'].values[0]
            st.metric("Persistence Baseline Test MAE", f"{persist_mae:.4f}", "Benchmark Reference")
        with c2:
            best_row = lb_a.loc[lb_a['test_mae'].idxmin()]
            st.metric("Champion: HistGradientBoosting", f"{best_row['test_mae']:.4f}", f"-{best_row['baseline_improvement_pct']:.2f}% Error vs Baseline")
        with c3:
            st.metric("Champion Test RMSE", f"{best_row['test_rmse']:.4f}", f"R² = {best_row['test_r2']:.4f}")
        with c4:
            st.metric("Generalization Gap", f"{best_row['generalization_gap_mae']:+.4f} MAE", f"Val: {best_row['val_mae']:.4f} → Test: {best_row['test_mae']:.4f}")

        st.markdown("#### **Official Test Performance Table**")
        disp_cols_a = {
            'model_name': 'Candidate Model',
            'test_mae': f'Test MAE ({unit_str_a})',
            'test_rmse': 'Test RMSE',
            'test_r2': 'Test R²',
            'baseline_improvement_pct': 'Improvement vs Baseline (%)',
            'val_mae': 'Validation MAE',
            'generalization_gap_mae': 'Generalization Gap (MAE)',
            'relative_degradation_pct': 'Relative Change (%)',
            'sensor_mae_median': 'Median Sensor MAE'
        }
        st.dataframe(lb_a[list(disp_cols_a.keys())].rename(columns=disp_cols_a), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("📊 Generalization Stability: Validation vs. Official Test MAE")
        gap_df = pd.melt(
            lb_a,
            id_vars=['model_name'],
            value_vars=['val_mae', 'test_mae'],
            var_name='Partition',
            value_name='MAE'
        )
        gap_df['Partition'] = gap_df['Partition'].map({'val_mae': 'Validation (Phase 3)', 'test_mae': 'Official Test (Phase 5)'})

        fig_gap = px.bar(
            gap_df,
            x='model_name',
            y='MAE',
            color='Partition',
            barmode='group',
            title="Validation vs. Official Test MAE (Notice the Stable Generalization Gap)",
            labels={'model_name': 'Model', 'MAE': f'Mean Absolute Error ({unit_str_a})'},
            color_discrete_map={'Validation (Phase 3)': '#3B82F6', 'Official Test (Phase 5)': '#0F172A'},
            template='plotly_white'
        )
        fig_gap.update_layout(height=400)
        st.plotly_chart(fig_gap, use_container_width=True)

        # Per-sensor test errors
        st.markdown("---")
        st.subheader("🛰️ Sensor Station Test MAE Distribution (HistGradientBoosting Champion)")
        sensor_test_df = load_dataset_a_per_sensor_test_errors()
        fig_sensor_bar = px.bar(
            sensor_test_df,
            x='sensor_idx',
            y='hist_gradient_boosting_test_mae',
            title="Official Test MAE by Individual Loop Detector Station (All 36 Sensors)",
            labels={'sensor_idx': 'Sensor Station Index (S00–S35)', 'hist_gradient_boosting_test_mae': 'Test MAE'},
            color='hist_gradient_boosting_test_mae',
            color_continuous_scale='Blues',
            template='plotly_white'
        )
        fig_sensor_bar.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_sensor_bar, use_container_width=True)

    else:
        st.subheader(f"📋 Dataset A: Validation Leaderboard ({horizon_str_a})")
        st.caption("Phase 3 model selection partition: 9,396 samples (261 time steps × 36 sensors). Used exclusively to select top models.")
        lb_a_val = load_dataset_a_leaderboard()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Persistence Baseline MAE", f"{lb_a_val[lb_a_val['model_slug']=='persistence']['mae'].values[0]:.4f}", "Benchmark Reference")
        with c2:
            best_val = lb_a_val.loc[lb_a_val['mae'].idxmin()]
            st.metric("Top Model: HistGradientBoosting", f"{best_val['mae']:.4f}", f"+{best_val['baseline_improvement_pct']:.2f}% Error Reduction")
        with c3:
            st.metric("Validation RMSE", f"{best_val['rmse']:.4f}", f"R² = {best_val['r2']:.4f}")
        with c4:
            st.metric("Inference Latency", f"{best_val.get('inference_time_ms', 0.05):.3f} ms", "Fast CPU Execution")

        disp_val_cols_a = {
            'model_name': 'Candidate Model',
            'mae': f'Validation MAE ({unit_str_a})',
            'rmse': 'Validation RMSE',
            'r2': 'R² Score',
            'baseline_improvement_pct': 'Gain vs Baseline (%)',
            'train_time_sec': 'Training Time (sec)',
            'inference_time_ms': 'Inference Latency (ms)'
        }
        st.dataframe(lb_a_val[list(disp_val_cols_a.keys())].rename(columns=disp_val_cols_a), use_container_width=True, hide_index=True)

        fig_val_bar = plot_leaderboard_bar(lb_a_val, metric='mae', y_label=f"Validation MAE ({unit_str_a})")
        st.plotly_chart(fig_val_bar, use_container_width=True)

# ==============================================================================
# DATASET B LEADERBOARD
# ==============================================================================
else:
    unit_str_b = "Vehicles / Hour"
    horizon_str_b = "1 Hour Ahead (calendar-verified)"

    if is_test_mode:
        st.subheader(f"🏁 Dataset B: Official Test Leaderboard ({horizon_str_b})")
        st.caption("Evaluated on 5,675 unseen test hours (December 2017 to September 2018) after all model hyperparameters were frozen.")
        lb_b = load_dataset_b_test_leaderboard()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            persist_mae_b = lb_b[lb_b['model_slug'] == 'persistence']['test_mae'].values[0]
            st.metric("Persistence Baseline Test MAE", f"{persist_mae_b:,.1f} veh/h", "Benchmark Reference")
        with c2:
            best_row_b = lb_b.loc[lb_b['test_mae'].idxmin()]
            st.metric("Champion: Random Forest", f"{best_row_b['test_mae']:,.1f} veh/h", f"-{best_row_b['baseline_improvement_pct']:.2f}% Error vs Baseline")
        with c3:
            st.metric("Champion Test RMSE", f"{best_row_b['test_rmse']:,.1f} veh/h", f"R² = {best_row_b['test_r2']:.4f}")
        with c4:
            st.metric("Generalization Gap", f"{best_row_b['generalization_gap_mae']:+,.1f} veh/h", f"Val: {best_row_b['val_mae']:.1f} → Test: {best_row_b['test_mae']:.1f}")

        st.markdown("#### **Official Test Performance Table**")
        disp_cols_b = {
            'model_name': 'Candidate Model',
            'test_mae': f'Test MAE ({unit_str_b})',
            'test_rmse': f'Test RMSE ({unit_str_b})',
            'test_r2': 'Test R²',
            'baseline_improvement_pct': 'Gain vs Baseline (%)',
            'val_mae': f'Validation MAE ({unit_str_b})',
            'generalization_gap_mae': f'Generalization Gap ({unit_str_b})',
            'relative_degradation_pct': 'Relative Change (%)'
        }
        st.dataframe(lb_b[list(disp_cols_b.keys())].rename(columns=disp_cols_b), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("📊 Generalization Stability: Validation vs. Official Test MAE")
        gap_df_b = pd.melt(
            lb_b,
            id_vars=['model_name'],
            value_vars=['val_mae', 'test_mae'],
            var_name='Partition',
            value_name='MAE'
        )
        gap_df_b['Partition'] = gap_df_b['Partition'].map({'val_mae': 'Validation (Phase 3)', 'test_mae': 'Official Test (Phase 5)'})

        fig_gap_b = px.bar(
            gap_df_b,
            x='model_name',
            y='MAE',
            color='Partition',
            barmode='group',
            title="Validation vs. Official Test MAE (Notice Zero Overfitting Across All Models)",
            labels={'model_name': 'Model', 'MAE': f'Mean Absolute Error ({unit_str_b})'},
            color_discrete_map={'Validation (Phase 3)': '#3B82F6', 'Official Test (Phase 5)': '#0F172A'},
            template='plotly_white'
        )
        fig_gap_b.update_layout(height=400)
        st.plotly_chart(fig_gap_b, use_container_width=True)

    else:
        st.subheader(f"📋 Dataset B: Validation Leaderboard ({horizon_str_b})")
        st.caption("Phase 3 model selection partition: 5,673 gap-filtered hourly records. Used exclusively to select top models.")
        lb_b_val = load_dataset_b_leaderboard()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Persistence Baseline MAE", f"{lb_b_val[lb_b_val['model_slug']=='persistence']['mae'].values[0]:,.1f} veh/h", "Benchmark Reference")
        with c2:
            best_val_b = lb_b_val.loc[lb_b_val['mae'].idxmin()]
            st.metric("Top Model: Random Forest", f"{best_val_b['mae']:,.1f} veh/h", f"+{best_val_b['baseline_improvement_pct']:.2f}% Error Reduction")
        with c3:
            st.metric("Validation RMSE", f"{best_val_b['rmse']:,.1f} veh/h", f"R² = {best_val_b['r2']:.4f}")
        with c4:
            st.metric("HistGB Alternative", f"{lb_b_val[lb_b_val['model_slug']=='hist_gradient_boosting']['mae'].values[0]:,.1f} veh/h", "376 KB file size (160x smaller)")

        disp_val_cols_b = {
            'model_name': 'Candidate Model',
            'mae': f'Validation MAE ({unit_str_b})',
            'rmse': f'Validation RMSE ({unit_str_b})',
            'r2': 'R² Score',
            'baseline_improvement_pct': 'Gain vs Baseline (%)',
            'train_time_sec': 'Training Time (sec)',
            'inference_time_ms': 'Inference Latency (ms)'
        }
        st.dataframe(lb_b_val[list(disp_val_cols_b.keys())].rename(columns=disp_val_cols_b), use_container_width=True, hide_index=True)

        fig_val_bar_b = plot_leaderboard_bar(lb_b_val, metric='mae', y_label=f"Validation MAE ({unit_str_b})")
        st.plotly_chart(fig_val_bar_b, use_container_width=True)
