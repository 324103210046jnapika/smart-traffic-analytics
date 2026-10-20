"""Automated tests for Phase 4: Streamlit Interactive Dashboard components, data loaders, and models."""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.utils.data_loader import (
    load_dataset_a_metadata,
    load_dataset_b_metadata,
    load_dataset_a_validation,
    load_dataset_b_validation,
    load_dataset_a_test,
    load_dataset_b_test,
    load_dataset_a_adjacency,
    load_dataset_a_leaderboard,
    load_dataset_b_leaderboard,
    load_dataset_a_per_sensor_errors,
    load_dataset_a_models,
    load_dataset_b_models
)
from app.utils.plots import (
    classify_demand_trend,
    plot_sensor_network_topology,
    plot_diurnal_traffic_profile,
    plot_actual_vs_predicted_series,
    plot_leaderboard_bar,
    plot_fundamental_diagram_curve,
    plot_centerpiece_forecast_dataset_a,
    plot_centerpiece_forecast_dataset_b,
    plot_forecasting_studio_timeseries,
    plot_residual_distribution_histogram
)


def test_dashboard_data_loaders():
    """Verifies that all cached dashboard data loader utilities return valid datasets."""
    meta_a = load_dataset_a_metadata()
    assert isinstance(meta_a, dict)
    assert meta_a.get('num_sensors') == 36
    assert meta_a.get('val_samples') == 9396

    meta_b = load_dataset_b_metadata()
    assert isinstance(meta_b, dict)
    assert meta_b.get('val_samples') == 5673
    assert meta_b.get('invalid_gap_targets_excluded') == 2589

    val_a = load_dataset_a_validation()
    assert isinstance(val_a, pd.DataFrame)
    assert len(val_a) == 9396
    assert 'target' in val_a.columns
    assert 'sensor_idx' in val_a.columns

    val_b = load_dataset_b_validation()
    assert isinstance(val_b, pd.DataFrame)
    assert len(val_b) == 5673
    assert 'target_1h' in val_b.columns
    assert 'traffic_volume' in val_b.columns

    adj_mat = load_dataset_a_adjacency()
    assert isinstance(adj_mat, np.ndarray)
    assert adj_mat.shape == (36, 36)
    assert int(np.sum(adj_mat)) == 86

    lb_a = load_dataset_a_leaderboard()
    assert isinstance(lb_a, pd.DataFrame)
    assert len(lb_a) == 4
    assert 'mae' in lb_a.columns
    assert 'baseline_improvement_pct' in lb_a.columns

    lb_b = load_dataset_b_leaderboard()
    assert isinstance(lb_b, pd.DataFrame)
    assert len(lb_b) == 4
    assert 'mae' in lb_b.columns

    sensor_errors = load_dataset_a_per_sensor_errors()
    assert isinstance(sensor_errors, pd.DataFrame)
    assert len(sensor_errors) == 36


def test_dashboard_model_loading():
    """Verifies that all 8 trained model artifacts load and have callable predict methods."""
    models_a = load_dataset_a_models()
    expected_slugs = ["persistence", "ridge", "random_forest", "hist_gradient_boosting"]
    for slug in expected_slugs:
        assert slug in models_a, f"Model {slug} missing from Dataset A registry"
        assert hasattr(models_a[slug], "predict"), f"Model {slug} does not have a predict method"

    models_b = load_dataset_b_models()
    for slug in expected_slugs:
        assert slug in models_b, f"Model {slug} missing from Dataset B registry"
        assert hasattr(models_b[slug], "predict"), f"Model {slug} does not have a predict method"


def test_forecast_input_output_consistency_dataset_a():
    """Verifies that Dataset A models produce valid finite predictions matching feature batch shapes."""
    val_a = load_dataset_a_validation()
    feat_cols = [f"feat_{i:02d}" for i in range(48)]
    sample_X = val_a[feat_cols].iloc[:50].values.astype(np.float32)

    models_a = load_dataset_a_models()
    for slug, model in models_a.items():
        preds = model.predict(sample_X)
        assert preds.shape == (50,), f"Dataset A model {slug} output shape mismatch"
        assert np.all(np.isfinite(preds)), f"Dataset A model {slug} returned non-finite values"
        assert np.all(preds >= -0.5) and np.all(preds <= 1.5), f"Dataset A model {slug} outputs out of range"


def test_forecast_input_output_consistency_dataset_b():
    """Verifies that Dataset B models produce valid finite predictions matching feature batch shapes."""
    val_b = load_dataset_b_validation()
    meta_cols = ['target_1h', 'date_time', 'traffic_volume']
    feat_cols_b = [c for c in val_b.columns if c not in meta_cols]
    assert len(feat_cols_b) == 33

    sample_X = val_b[feat_cols_b].iloc[:50].values.astype(np.float32)

    models_b = load_dataset_b_models()
    for slug, model in models_b.items():
        preds = model.predict(sample_X)
        assert preds.shape == (50,), f"Dataset B model {slug} output shape mismatch"
        assert np.all(np.isfinite(preds)), f"Dataset B model {slug} returned non-finite values"


def test_plot_generation_functions():
    """Verifies that all visualization generator functions execute and return valid Plotly Figures."""
    adj_mat = load_dataset_a_adjacency()
    fig_topo = plot_sensor_network_topology(adj_mat)
    assert isinstance(fig_topo, go.Figure)

    val_b = load_dataset_b_validation()
    fig_diurnal = plot_diurnal_traffic_profile(val_b)
    assert isinstance(fig_diurnal, go.Figure)

    x_dummy = np.arange(20)
    y_actual = np.random.rand(20)
    y_pred = y_actual + 0.05
    fig_ts = plot_actual_vs_predicted_series(x_dummy, y_actual, y_pred, y_actual, "Test Series")
    assert isinstance(fig_ts, go.Figure)

    lb_a = load_dataset_a_leaderboard()
    fig_bar = plot_leaderboard_bar(lb_a, metric='mae')
    assert isinstance(fig_bar, go.Figure)

    fig_fund = plot_fundamental_diagram_curve()
    assert isinstance(fig_fund, go.Figure)


def test_dashboard_files_syntax():
    """Ensures that all dashboard page scripts compile cleanly without any syntax errors."""
    app_files = [
        os.path.join(PROJECT_ROOT, "app", "streamlit_app.py"),
        os.path.join(PROJECT_ROOT, "app", "pages", "1_Overview_and_EDA.py"),
        os.path.join(PROJECT_ROOT, "app", "pages", "2_Model_Leaderboard.py"),
        os.path.join(PROJECT_ROOT, "app", "pages", "3_Forecasting_Studio.py"),
        os.path.join(PROJECT_ROOT, "app", "pages", "4_Weather_Benchmark.py"),
        os.path.join(PROJECT_ROOT, "app", "pages", "5_Risks_and_Limitations.py"),
    ]

    for fpath in app_files:
        assert os.path.exists(fpath), f"File {fpath} does not exist"
        with open(fpath, "r", encoding="utf-8") as f:
            code = f.read()
            # Compile to check for syntax errors
            compiled = compile(code, fpath, 'exec')
            assert compiled is not None


def test_classify_demand_trend():
    """Verifies that 5-tier demand trend classification correctly partitions deltas for both datasets."""
    # Dataset A (Normalized flow [0, 1])
    res_a_rising = classify_demand_trend(+0.050, dataset="A")
    assert res_a_rising["status"] == "Rising"
    assert "trend-rising" in res_a_rising["badge_html"]

    res_a_slightly_higher = classify_demand_trend(+0.025, dataset="A")
    assert res_a_slightly_higher["status"] == "Slightly Higher"
    assert "trend-slightly-higher" in res_a_slightly_higher["badge_html"]

    res_a_steady = classify_demand_trend(+0.005, dataset="A")
    assert res_a_steady["status"] == "Steady"
    assert "trend-steady" in res_a_steady["badge_html"]

    res_a_slightly_lower = classify_demand_trend(-0.025, dataset="A")
    assert res_a_slightly_lower["status"] == "Slightly Lower"
    assert "trend-slightly-lower" in res_a_slightly_lower["badge_html"]

    res_a_easing = classify_demand_trend(-0.050, dataset="A")
    assert res_a_easing["status"] == "Easing"
    assert "trend-easing" in res_a_easing["badge_html"]

    # Dataset B (Vehicles / hour)
    res_b_rising = classify_demand_trend(+450.0, dataset="B")
    assert res_b_rising["status"] == "Rising"
    assert "trend-rising" in res_b_rising["badge_html"]

    res_b_slightly_higher = classify_demand_trend(+180.0, dataset="B")
    assert res_b_slightly_higher["status"] == "Slightly Higher"
    assert "trend-slightly-higher" in res_b_slightly_higher["badge_html"]

    res_b_steady = classify_demand_trend(-45.0, dataset="B")
    assert res_b_steady["status"] == "Steady"
    assert "trend-steady" in res_b_steady["badge_html"]

    res_b_slightly_lower = classify_demand_trend(-150.0, dataset="B")
    assert res_b_slightly_lower["status"] == "Slightly Lower"
    assert "trend-slightly-lower" in res_b_slightly_lower["badge_html"]

    res_b_easing = classify_demand_trend(-500.0, dataset="B")
    assert res_b_easing["status"] == "Easing"
    assert "trend-easing" in res_b_easing["badge_html"]


def test_centerpiece_forecast_plots_retrospective_modes():
    """Verifies that centerpiece forecast plots render properly with prospective and retrospective modes."""
    # Dataset A prospective mode (ground truth hidden)
    fig_a_prospective = plot_centerpiece_forecast_dataset_a(
        history_steps=[10, 11, 12, 13, 14, 15, 16, 17, 18],
        history_flows=[0.4, 0.42, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60],
        current_step=18,
        current_flow=0.60,
        forecast_step=20,
        forecast_flow=0.65,
        actual_target=0.64,
        sensor_id=14,
        trend_str="Rising",
        retrospective_mode=False
    )
    assert isinstance(fig_a_prospective, go.Figure)
    trace_names_a_prosp = [t.name for t in fig_a_prospective.data if t.name is not None]
    assert "Model Forecast (t+30m)" in trace_names_a_prosp
    assert not any("Retrospective" in name or "Ground Truth" in name for name in trace_names_a_prosp)

    # Dataset A retrospective mode (ground truth shown)
    fig_a_retro = plot_centerpiece_forecast_dataset_a(
        history_steps=[10, 11, 12, 13, 14, 15, 16, 17, 18],
        history_flows=[0.4, 0.42, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60],
        current_step=18,
        current_flow=0.60,
        forecast_step=20,
        forecast_flow=0.65,
        actual_target=0.64,
        sensor_id=14,
        trend_str="Rising",
        retrospective_mode=True
    )
    assert isinstance(fig_a_retro, go.Figure)
    trace_names_a_retro = [t.name for t in fig_a_retro.data if t.name is not None]
    assert any("Retrospective" in name for name in trace_names_a_retro)

    # Dataset B prospective mode (ground truth hidden)
    fig_b_prospective = plot_centerpiece_forecast_dataset_b(
        history_times=[f"Jun 11, {h:02d}:00" for h in range(10, 24)] + ["Jun 12, 00:00", "Jun 12, 01:00"],
        history_vols=[3000.0 + h * 50 for h in range(16)],
        current_time="Jun 12, 01:00",
        current_vol=3750.0,
        forecast_time="Jun 12, 02:00",
        forecast_vol=3900.0,
        actual_target=3850.0,
        trend_str="Rising",
        weather_str="Clear",
        retrospective_mode=False
    )
    assert isinstance(fig_b_prospective, go.Figure)
    trace_names_b_prosp = [t.name for t in fig_b_prospective.data if t.name is not None]
    assert "Model Forecast (t+1h)" in trace_names_b_prosp
    assert not any("Retrospective" in name for name in trace_names_b_prosp)

    # Dataset B retrospective mode (ground truth shown)
    fig_b_retro = plot_centerpiece_forecast_dataset_b(
        history_times=[f"Jun 11, {h:02d}:00" for h in range(10, 24)] + ["Jun 12, 00:00", "Jun 12, 01:00"],
        history_vols=[3000.0 + h * 50 for h in range(16)],
        current_time="Jun 12, 01:00",
        current_vol=3750.0,
        forecast_time="Jun 12, 02:00",
        forecast_vol=3900.0,
        actual_target=3850.0,
        trend_str="Rising",
        weather_str="Clear",
        retrospective_mode=True
    )
    assert isinstance(fig_b_retro, go.Figure)
    trace_names_b_retro = [t.name for t in fig_b_retro.data if t.name is not None]
    assert any("Retrospective" in name for name in trace_names_b_retro)


def test_forecasting_studio_new_plots_and_loaders():
    """Verifies that held-out test set loaders and new forecasting studio plots operate correctly."""
    # Test loaders
    test_a = load_dataset_a_test()
    assert isinstance(test_a, pd.DataFrame)
    assert len(test_a) == 30240
    assert 'target' in test_a.columns
    assert 'sensor_idx' in test_a.columns

    test_b = load_dataset_b_test()
    assert isinstance(test_b, pd.DataFrame)
    assert len(test_b) == 5675
    assert 'target_1h' in test_b.columns
    assert 'traffic_volume' in test_b.columns

    # Test forecasting studio plots
    x_steps = [f"Step {i}" for i in range(20)]
    y_act = np.linspace(0.4, 0.7, 20)
    y_fcst = y_act + 0.03
    y_base = y_act - 0.02

    # Retrospective mode
    fig_retro = plot_forecasting_studio_timeseries(
        x_vals=x_steps,
        y_actual=y_act,
        y_pred=y_fcst,
        y_baseline=y_base,
        retrospective_mode=True,
        title="Test Studio Retrospective",
        y_label="Normalized Flow [0, 1]",
        x_label="Step Index",
        unit_str=""
    )
    assert isinstance(fig_retro, go.Figure)
    assert len(fig_retro.data) >= 3

    # Prospective mode
    fig_prosp = plot_forecasting_studio_timeseries(
        x_vals=x_steps,
        y_actual=y_act,
        y_pred=y_fcst,
        y_baseline=y_base,
        retrospective_mode=False,
        title="Test Studio Prospective",
        y_label="Normalized Flow [0, 1]",
        x_label="Step Index",
        unit_str=""
    )
    assert isinstance(fig_prosp, go.Figure)
    assert len(fig_prosp.data) >= 2

    # Residual distribution histogram
    residuals = y_fcst - y_act
    fig_hist = plot_residual_distribution_histogram(
        residuals=residuals,
        unit_str="flow",
        title="Residual Error Distribution Test"
    )
    assert isinstance(fig_hist, go.Figure)
    assert len(fig_hist.data) >= 1


