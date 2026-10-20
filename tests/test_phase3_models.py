"""Unit tests for Phase 3: model fitting, prediction shapes, finite outputs, and metric calculations."""

import os
import json
import joblib
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import calculate_metrics
from src.evaluation.per_sensor import evaluate_per_sensor
from src.models.persistence import PersistenceForecaster
from src.models.ridge_model import RidgeForecaster
from src.models.random_forest import RandomForestForecaster
from src.models.hist_gradient_boosting import HistGradientBoostingForecaster

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_A_DIR = os.path.join(PROJECT_ROOT, "artifacts", "models", "dataset_a")
MODELS_B_DIR = os.path.join(PROJECT_ROOT, "artifacts", "models", "dataset_b")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "metrics")


def test_metric_calculations_precision():
    """Verify mathematical exactness of MAE, RMSE, R2, and baseline improvement."""
    y_true = np.array([2.0, 4.0, 6.0, 8.0])
    y_pred = np.array([2.5, 3.5, 6.5, 7.5]) # errors: +0.5, -0.5, +0.5, -0.5
    baseline_mae = 1.0

    metrics = calculate_metrics(y_true, y_pred, baseline_mae=baseline_mae)

    assert pytest.approx(metrics['mae'], 1e-5) == 0.5
    assert pytest.approx(metrics['rmse'], 1e-5) == 0.5
    # Variance of y_true is 5.0; MSE is 0.25 => R2 = 1 - 0.25/5.0 = 0.95
    assert pytest.approx(metrics['r2'], 1e-5) == 0.95
    # Improvement: (1.0 - 0.5) / 1.0 * 100 = 50.0%
    assert pytest.approx(metrics['baseline_improvement_pct'], 1e-5) == 50.0


def test_persistence_forecaster_logic():
    """Verify that PersistenceForecaster extracts the designated column with zero corruption."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 15)).astype(np.float32)
    forecaster = PersistenceForecaster(persistence_col_idx=9)
    forecaster.fit(X)
    preds = forecaster.predict(X)

    assert preds.shape == (50,)
    assert np.allclose(preds, X[:, 9])
    assert forecaster.get_params()['persistence_col_idx'] == 9


def test_model_fitting_and_finite_predictions():
    """Verify that all ML wrappers train on synthetic data and yield finite predictions."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 10)).astype(np.float32)
    y = (X[:, 0] * 2.0 + X[:, 1] * -1.5 + rng.normal(scale=0.1, size=200)).astype(np.float32)

    models = [
        RidgeForecaster(alpha=1.0, random_state=42),
        RandomForestForecaster(n_estimators=10, max_depth=5, random_state=42),
        HistGradientBoostingForecaster(max_iter=15, random_state=42)
    ]

    for model in models:
        model.fit(X, y)
        preds = model.predict(X)

        assert preds.shape == (200,), f"{model.model_name} prediction shape mismatch"
        assert np.isfinite(preds).all(), f"{model.model_name} generated non-finite predictions"
        assert np.isnan(preds).sum() == 0, f"{model.model_name} generated NaNs"


def test_per_sensor_evaluation():
    """Verify per-sensor metric aggregation and distribution statistics."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y_pred = np.array([1.2, 1.9, 3.1, 4.5, 4.8, 6.2])
    y_pers = np.array([0.5, 1.5, 2.5, 3.0, 4.0, 5.0])
    sensors = np.array([0, 0, 0, 1, 1, 1])

    per_sensor_df, dist = evaluate_per_sensor(y_true, y_pred, y_pers, sensors)

    assert len(per_sensor_df) == 2
    assert dist['num_sensors_evaluated'] == 2
    assert 'mae_median' in dist
    assert 'mae_min' in dist
    assert 'mae_max' in dist


def test_dataset_a_serialized_artifacts_exist_and_load():
    """Verify that all 4 models for Dataset A are saved and loadable."""
    expected_slugs = ["persistence", "ridge", "random_forest", "hist_gradient_boosting"]
    for slug in expected_slugs:
        artifact_path = os.path.join(MODELS_A_DIR, f"{slug}.joblib")
        assert os.path.isfile(artifact_path), f"Missing serialized model: {artifact_path}"
        model = joblib.load(artifact_path)
        assert hasattr(model, "predict"), f"Loaded model {slug} lacks predict method"


def test_dataset_b_serialized_artifacts_exist_and_load():
    """Verify that all 4 models for Dataset B are saved and loadable."""
    expected_slugs = ["persistence", "ridge", "random_forest", "hist_gradient_boosting"]
    for slug in expected_slugs:
        artifact_path = os.path.join(MODELS_B_DIR, f"{slug}.joblib")
        assert os.path.isfile(artifact_path), f"Missing serialized model: {artifact_path}"
        model = joblib.load(artifact_path)
        assert hasattr(model, "predict"), f"Loaded model {slug} lacks predict method"


def test_leaderboard_files_contain_all_models():
    """Verify that exported validation leaderboards contain valid entries for all 4 models."""
    for dataset_key in ["dataset_a", "dataset_b"]:
        json_path = os.path.join(REPORTS_DIR, f"{dataset_key}_validation_leaderboard.json")
        csv_path = os.path.join(REPORTS_DIR, f"{dataset_key}_validation_leaderboard.csv")

        assert os.path.isfile(json_path), f"Missing JSON leaderboard: {json_path}"
        assert os.path.isfile(csv_path), f"Missing CSV leaderboard: {csv_path}"

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert len(data) == 4, f"Expected 4 models in {dataset_key} leaderboard, found {len(data)}"
        for row in data:
            assert "mae" in row
            assert "rmse" in row
            assert "r2" in row
            assert "baseline_improvement_pct" in row
            assert "fit_time_seconds" in row
