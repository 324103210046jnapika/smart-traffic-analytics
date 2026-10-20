"""Model training, validation evaluation, and artifact serialization pipeline."""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from src.evaluation.metrics import calculate_metrics
from src.evaluation.per_sensor import evaluate_per_sensor
from src.models.persistence import PersistenceForecaster
from src.models.ridge_model import RidgeForecaster
from src.models.random_forest import RandomForestForecaster
from src.models.hist_gradient_boosting import HistGradientBoostingForecaster


def run_dataset_a_benchmarks(project_root: str) -> Dict[str, Any]:
    """Trains and benchmarks all 4 models on Dataset A strictly using validation partition."""
    print("\n" + "="*70)
    print("RUNNING BENCHMARKS: DATASET A (UCI 734 Traffic Flow Forecasting)")
    print("="*70)

    proc_dir = os.path.join(project_root, "data", "processed", "dataset_a")
    model_save_dir = os.path.join(project_root, "artifacts", "models", "dataset_a")
    reports_dir = os.path.join(project_root, "reports", "metrics")
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    df_train = pd.read_parquet(os.path.join(proc_dir, "train.parquet"))
    df_val = pd.read_parquet(os.path.join(proc_dir, "val.parquet"))

    feat_cols = [f"feat_{i:02d}" for i in range(48)]
    X_train = df_train[feat_cols].values.astype(np.float32)
    y_train = df_train['target'].values.astype(np.float32)

    X_val = df_val[feat_cols].values.astype(np.float32)
    y_val = df_val['target'].values.astype(np.float32)
    val_sensors = df_val['sensor_idx'].values
    y_persistence = df_val['persistence_pred'].values.astype(np.float32)

    print(f"Dataset A Training samples:   {len(X_train)} ({len(X_train)//36} steps * 36 sensors)")
    print(f"Dataset A Validation samples: {len(X_val)} ({len(X_val)//36} steps * 36 sensors)")

    models = [
        ("persistence", PersistenceForecaster(persistence_col_idx=9)),
        ("ridge", RidgeForecaster(alpha=1.0, random_state=42)),
        ("random_forest", RandomForestForecaster(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1)),
        ("hist_gradient_boosting", HistGradientBoostingForecaster(max_iter=100, learning_rate=0.1, random_state=42))
    ]

    leaderboard = []
    per_sensor_results_all = {}
    baseline_mae = None

    for slug, model in models:
        print(f"\n--- Training {model.model_name} ---")
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        val_preds = model.predict(X_val)
        infer_time = time.perf_counter() - t1
        latency_ms_per_sample = (infer_time / len(X_val)) * 1000.0

        if slug == "persistence":
            metrics = calculate_metrics(y_val, val_preds)
            baseline_mae = metrics['mae']
            metrics['baseline_improvement_pct'] = 0.0
        else:
            metrics = calculate_metrics(y_val, val_preds, baseline_mae=baseline_mae)

        # Per-sensor evaluation
        per_sensor_df, dist_summary = evaluate_per_sensor(y_val, val_preds, y_persistence, val_sensors)
        per_sensor_results_all[slug] = per_sensor_df

        # Save serialized model artifact
        artifact_path = os.path.join(model_save_dir, f"{slug}.joblib")
        joblib.dump(model, artifact_path)
        artifact_size_mb = os.path.getsize(artifact_path) / (1024 * 1024)

        entry = {
            'model_slug': slug,
            'model_name': model.model_name,
            'mae': round(metrics['mae'], 4),
            'rmse': round(metrics['rmse'], 4),
            'r2': round(metrics['r2'], 4),
            'baseline_improvement_pct': round(metrics['baseline_improvement_pct'], 2),
            'fit_time_seconds': round(fit_time, 3),
            'inference_latency_ms_per_sample': round(latency_ms_per_sample, 4),
            'artifact_size_mb': round(artifact_size_mb, 3),
            'sensor_mae_median': round(dist_summary['mae_median'], 4),
            'sensor_mae_min': round(dist_summary['mae_min'], 4),
            'sensor_mae_max': round(dist_summary['mae_max'], 4)
        }
        leaderboard.append(entry)

        print(f"Results for {model.model_name}:")
        print(f"  Validation MAE:  {entry['mae']:.4f}")
        print(f"  Validation RMSE: {entry['rmse']:.4f}")
        print(f"  Validation R^2:  {entry['r2']:.4f}")
        print(f"  Improvement vs Persistence: {entry['baseline_improvement_pct']:.2f}%")
        print(f"  Training Time: {fit_time:.2f}s, Latency: {latency_ms_per_sample*1000:.2f} us/sample")

    # Export leaderboard CSV & JSON
    df_leaderboard = pd.DataFrame(leaderboard)
    csv_path = os.path.join(reports_dir, "dataset_a_validation_leaderboard.csv")
    json_path = os.path.join(reports_dir, "dataset_a_validation_leaderboard.json")
    df_leaderboard.to_csv(csv_path, index=False)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(leaderboard, f, indent=2)

    # Export combined per-sensor errors
    per_sensor_export = per_sensor_results_all['persistence'][['sensor_idx', 'persistence_mae']].copy()
    for slug in ['ridge', 'random_forest', 'hist_gradient_boosting']:
        per_sensor_export[f'{slug}_mae'] = per_sensor_results_all[slug]['mae'].values
        per_sensor_export[f'{slug}_improvement_pct'] = per_sensor_results_all[slug]['baseline_improvement_pct'].values

    sensor_csv_path = os.path.join(reports_dir, "dataset_a_per_sensor_errors.csv")
    per_sensor_export.to_csv(sensor_csv_path, index=False)

    return {
        'leaderboard': leaderboard,
        'leaderboard_df': df_leaderboard,
        'per_sensor_df': per_sensor_export
    }


def run_dataset_b_benchmarks(project_root: str) -> Dict[str, Any]:
    """Trains and benchmarks all 4 models on Dataset B strictly using validation partition."""
    print("\n" + "="*70)
    print("RUNNING BENCHMARKS: DATASET B (UCI 492 Metro Interstate Traffic Volume)")
    print("="*70)

    proc_dir = os.path.join(project_root, "data", "processed", "dataset_b")
    model_save_dir = os.path.join(project_root, "artifacts", "models", "dataset_b")
    reports_dir = os.path.join(project_root, "reports", "metrics")
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    df_train = pd.read_parquet(os.path.join(proc_dir, "train.parquet"))
    df_val = pd.read_parquet(os.path.join(proc_dir, "val.parquet"))

    # Exclude metadata and target columns from model feature inputs
    exclude_cols = ['target_1h', 'date_time', 'traffic_volume']
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]
    persistence_col_idx = feature_cols.index('persistence_pred')

    X_train = df_train[feature_cols].values.astype(np.float32)
    y_train = df_train['target_1h'].values.astype(np.float32)

    X_val = df_val[feature_cols].values.astype(np.float32)
    y_val = df_val['target_1h'].values.astype(np.float32)

    print(f"Dataset B Training samples:   {len(X_train)} hours")
    print(f"Dataset B Validation samples: {len(X_val)} hours")
    print(f"Features: {len(feature_cols)} (Traffic lags, rolling stats, cyclic calendar, weather)")

    models = [
        ("persistence", PersistenceForecaster(persistence_col_idx=persistence_col_idx)),
        ("ridge", RidgeForecaster(alpha=1.0, random_state=42)),
        ("random_forest", RandomForestForecaster(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1)),
        ("hist_gradient_boosting", HistGradientBoostingForecaster(max_iter=100, learning_rate=0.1, random_state=42))
    ]

    leaderboard = []
    baseline_mae = None

    for slug, model in models:
        print(f"\n--- Training {model.model_name} ---")
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        val_preds = model.predict(X_val)
        infer_time = time.perf_counter() - t1
        latency_ms_per_sample = (infer_time / len(X_val)) * 1000.0

        if slug == "persistence":
            metrics = calculate_metrics(y_val, val_preds)
            baseline_mae = metrics['mae']
            metrics['baseline_improvement_pct'] = 0.0
        else:
            metrics = calculate_metrics(y_val, val_preds, baseline_mae=baseline_mae)

        # Save serialized model artifact
        artifact_path = os.path.join(model_save_dir, f"{slug}.joblib")
        joblib.dump(model, artifact_path)
        artifact_size_mb = os.path.getsize(artifact_path) / (1024 * 1024)

        entry = {
            'model_slug': slug,
            'model_name': model.model_name,
            'mae': round(metrics['mae'], 2),
            'rmse': round(metrics['rmse'], 2),
            'r2': round(metrics['r2'], 4),
            'baseline_improvement_pct': round(metrics['baseline_improvement_pct'], 2),
            'fit_time_seconds': round(fit_time, 3),
            'inference_latency_ms_per_sample': round(latency_ms_per_sample, 4),
            'artifact_size_mb': round(artifact_size_mb, 3)
        }
        leaderboard.append(entry)

        print(f"Results for {model.model_name}:")
        print(f"  Validation MAE:  {entry['mae']:.2f} vehicles/hour")
        print(f"  Validation RMSE: {entry['rmse']:.2f} vehicles/hour")
        print(f"  Validation R^2:  {entry['r2']:.4f}")
        print(f"  Improvement vs Persistence: {entry['baseline_improvement_pct']:.2f}%")
        print(f"  Training Time: {fit_time:.2f}s, Latency: {latency_ms_per_sample*1000:.2f} us/sample")

    # Export leaderboard CSV & JSON
    df_leaderboard = pd.DataFrame(leaderboard)
    csv_path = os.path.join(reports_dir, "dataset_b_validation_leaderboard.csv")
    json_path = os.path.join(reports_dir, "dataset_b_validation_leaderboard.json")
    df_leaderboard.to_csv(csv_path, index=False)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(leaderboard, f, indent=2)

    return {
        'leaderboard': leaderboard,
        'leaderboard_df': df_leaderboard
    }


if __name__ == '__main__':
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    res_a = run_dataset_a_benchmarks(root)
    res_b = run_dataset_b_benchmarks(root)
    print("\n" + "="*70)
    print("ALL PHASE 3 VALIDATION BENCHMARKS COMPLETED SUCCESSFULLY.")
    print("="*70)
