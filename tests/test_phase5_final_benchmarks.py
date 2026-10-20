"""Automated tests for Phase 5: Official Test Set Evaluation, Final Benchmarks, and Generalization Stability."""

import os
import json
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from app.utils.data_loader import (
    load_dataset_a_test_leaderboard,
    load_dataset_b_test_leaderboard,
    load_dataset_a_per_sensor_test_errors,
    load_final_benchmark_summary
)


def test_official_test_leaderboard_files_exist_and_populated():
    """Verifies that all Phase 5 test evaluation metric artifacts exist and contain valid results."""
    lb_a_path = os.path.join(PROJECT_ROOT, "reports", "metrics", "dataset_a_test_leaderboard.json")
    lb_b_path = os.path.join(PROJECT_ROOT, "reports", "metrics", "dataset_b_test_leaderboard.json")
    summary_path = os.path.join(PROJECT_ROOT, "reports", "metrics", "final_model_benchmark_summary.json")

    assert os.path.exists(lb_a_path), "Dataset A test leaderboard JSON missing"
    assert os.path.exists(lb_b_path), "Dataset B test leaderboard JSON missing"
    assert os.path.exists(summary_path), "Final unified benchmark summary JSON missing"

    with open(lb_a_path, "r", encoding="utf-8") as f:
        data_a = json.load(f)
    assert len(data_a) == 4
    for row in data_a:
        assert 'test_mae' in row
        assert 'test_rmse' in row
        assert 'test_r2' in row
        assert 'baseline_improvement_pct' in row
        assert 'generalization_gap_mae' in row

    with open(lb_b_path, "r", encoding="utf-8") as f:
        data_b = json.load(f)
    assert len(data_b) == 4
    for row in data_b:
        assert 'test_mae' in row
        assert 'test_rmse' in row
        assert 'test_r2' in row
        assert 'baseline_improvement_pct' in row


def test_official_test_sample_counts_strictly_match():
    """Verifies that test evaluations ran on the complete official test partitions."""
    df_test_a = pd.read_parquet(os.path.join(PROJECT_ROOT, "data", "processed", "dataset_a", "test.parquet"))
    assert len(df_test_a) == 30240, f"Dataset A test samples {len(df_test_a)} != 30240"
    assert len(df_test_a['step_idx'].unique()) == 840
    assert len(df_test_a['sensor_idx'].unique()) == 36

    df_test_b = pd.read_parquet(os.path.join(PROJECT_ROOT, "data", "processed", "dataset_b", "test.parquet"))
    assert len(df_test_b) == 5675, f"Dataset B test samples {len(df_test_b)} != 5675"


def test_generalization_stability_no_catastrophic_overfitting():
    """Verifies that test set errors demonstrate robust generalization without overfitting."""
    lb_a = load_dataset_a_test_leaderboard()
    best_a = lb_a[lb_a['model_slug'] == 'hist_gradient_boosting'].iloc[0]
    # Test MAE should be within 0.010 of validation MAE
    assert abs(best_a['generalization_gap_mae']) < 0.010
    assert best_a['test_r2'] > 0.95

    lb_b = load_dataset_b_test_leaderboard()
    best_b = lb_b[lb_b['model_slug'] == 'random_forest'].iloc[0]
    # Test MAE should be within 20 veh/h of validation MAE
    assert abs(best_b['generalization_gap_mae']) < 20.0
    assert best_b['test_r2'] > 0.98


def test_dataset_a_per_sensor_test_breakdown_consistency():
    """Verifies per-sensor test error summary table integrity."""
    per_sensor_df = load_dataset_a_per_sensor_test_errors()
    assert len(per_sensor_df) == 36
    assert 'hist_gradient_boosting_test_mae' in per_sensor_df.columns
    assert np.all(per_sensor_df['hist_gradient_boosting_test_mae'] > 0.0)
    assert np.all(per_sensor_df['hist_gradient_boosting_test_mae'] < 0.10)


def test_dashboard_test_leaderboard_loaders():
    """Verifies that all Phase 5 data loaders return non-empty DataFrames."""
    df_a = load_dataset_a_test_leaderboard()
    assert isinstance(df_a, pd.DataFrame)
    assert len(df_a) == 4

    df_b = load_dataset_b_test_leaderboard()
    assert isinstance(df_b, pd.DataFrame)
    assert len(df_b) == 4

    summary = load_final_benchmark_summary()
    assert isinstance(summary, dict)
    assert 'dataset_a' in summary
    assert 'dataset_b' in summary
