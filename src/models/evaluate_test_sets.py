"""Official test set evaluation and final benchmarking pipeline for Phase 5."""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from src.evaluation.metrics import calculate_metrics
from src.evaluation.per_sensor import evaluate_per_sensor


def run_dataset_a_test_evaluation(project_root: str) -> Dict[str, Any]:
    """Evaluates frozen models on Dataset A official test set (840 steps * 36 sensors = 30,240 samples)."""
    print("\n" + "=" * 70)
    print("PHASE 5 OFFICIAL TEST EVALUATION: DATASET A (UCI 734 Traffic Flow)")
    print("=" * 70)

    proc_dir = os.path.join(project_root, "data", "processed", "dataset_a")
    model_save_dir = os.path.join(project_root, "artifacts", "models", "dataset_a")
    reports_dir = os.path.join(project_root, "reports", "metrics")
    os.makedirs(reports_dir, exist_ok=True)

    df_test = pd.read_parquet(os.path.join(proc_dir, "test.parquet"))
    val_lb_path = os.path.join(reports_dir, "dataset_a_validation_leaderboard.json")
    with open(val_lb_path, "r", encoding="utf-8") as f:
        val_lb = {row['model_slug']: row for row in json.load(f)}

    feat_cols = [f"feat_{i:02d}" for i in range(48)]
    X_test = df_test[feat_cols].values.astype(np.float32)
    y_test = df_test['target'].values.astype(np.float32)
    test_sensors = df_test['sensor_idx'].values
    y_persistence = df_test['persistence_pred'].values.astype(np.float32)

    print(f"Dataset A Official Test samples: {len(X_test)} (840 time steps * 36 sensors)")

    model_slugs = ["persistence", "ridge", "random_forest", "hist_gradient_boosting"]
    leaderboard = []
    per_sensor_results_all = {}
    baseline_mae = None

    for slug in model_slugs:
        model_path = os.path.join(model_save_dir, f"{slug}.joblib")
        model = joblib.load(model_path)

        t0 = time.perf_counter()
        test_preds = model.predict(X_test).astype(np.float32)
        inference_time = time.perf_counter() - t0
        latency_ms_per_sample = (inference_time / len(X_test)) * 1000.0

        if slug == "persistence":
            metrics = calculate_metrics(y_test, test_preds)
            baseline_mae = metrics['mae']
            metrics['baseline_improvement_pct'] = 0.0
        else:
            metrics = calculate_metrics(y_test, test_preds, baseline_mae=baseline_mae)

        per_sensor_df, dist_summary = evaluate_per_sensor(y_test, test_preds, y_persistence, test_sensors)
        per_sensor_results_all[slug] = per_sensor_df

        val_mae = val_lb[slug]['mae']
        gen_gap_mae = round(metrics['mae'] - val_mae, 4)
        rel_degradation_pct = round(((metrics['mae'] - val_mae) / val_mae) * 100.0, 2)

        entry = {
            'model_slug': slug,
            'model_name': model.model_name,
            'test_mae': round(metrics['mae'], 4),
            'test_rmse': round(metrics['rmse'], 4),
            'test_r2': round(metrics['r2'], 4),
            'baseline_improvement_pct': round(metrics['baseline_improvement_pct'], 2),
            'val_mae': round(val_mae, 4),
            'generalization_gap_mae': gen_gap_mae,
            'relative_degradation_pct': rel_degradation_pct,
            'inference_latency_ms_per_sample': round(latency_ms_per_sample, 4),
            'sensor_mae_median': round(dist_summary['mae_median'], 4),
            'sensor_mae_min': round(dist_summary['mae_min'], 4),
            'sensor_mae_max': round(dist_summary['mae_max'], 4)
        }
        leaderboard.append(entry)

        print(f"\nResults for {model.model_name}:")
        print(f"  Test MAE:  {entry['test_mae']:.4f}  (Val MAE: {entry['val_mae']:.4f}, Gap: {gen_gap_mae:+.4f})")
        print(f"  Test RMSE: {entry['test_rmse']:.4f}")
        print(f"  Test R^2:  {entry['test_r2']:.4f}")
        print(f"  Improvement vs Persistence: {entry['baseline_improvement_pct']:.2f}%")
        print(f"  Latency: {latency_ms_per_sample * 1000.0:.2f} us/sample")

    # Export test leaderboard CSV & JSON
    df_leaderboard = pd.DataFrame(leaderboard)
    csv_path = os.path.join(reports_dir, "dataset_a_test_leaderboard.csv")
    json_path = os.path.join(reports_dir, "dataset_a_test_leaderboard.json")
    df_leaderboard.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2)

    # Export per-sensor test comparison table
    per_sensor_summary = pd.DataFrame({
        'sensor_idx': per_sensor_results_all['persistence']['sensor_idx'],
        'persistence_test_mae': per_sensor_results_all['persistence']['mae'],
        'ridge_test_mae': per_sensor_results_all['ridge']['mae'],
        'ridge_improvement_pct': per_sensor_results_all['ridge']['baseline_improvement_pct'],
        'random_forest_test_mae': per_sensor_results_all['random_forest']['mae'],
        'random_forest_improvement_pct': per_sensor_results_all['random_forest']['baseline_improvement_pct'],
        'hist_gradient_boosting_test_mae': per_sensor_results_all['hist_gradient_boosting']['mae'],
        'hist_gradient_boosting_improvement_pct': per_sensor_results_all['hist_gradient_boosting']['baseline_improvement_pct'],
    })
    per_sensor_csv = os.path.join(reports_dir, "dataset_a_per_sensor_test_errors.csv")
    per_sensor_summary.to_csv(per_sensor_csv, index=False)

    return {
        'leaderboard': leaderboard,
        'per_sensor_summary': per_sensor_summary.to_dict(orient='records'),
        'test_samples': len(X_test)
    }


def run_dataset_b_test_evaluation(project_root: str) -> Dict[str, Any]:
    """Evaluates frozen models on Dataset B official test set (5,675 gap-filtered hours)."""
    print("\n" + "=" * 70)
    print("PHASE 5 OFFICIAL TEST EVALUATION: DATASET B (UCI 492 Interstate 94)")
    print("=" * 70)

    proc_dir = os.path.join(project_root, "data", "processed", "dataset_b")
    model_save_dir = os.path.join(project_root, "artifacts", "models", "dataset_b")
    reports_dir = os.path.join(project_root, "reports", "metrics")
    os.makedirs(reports_dir, exist_ok=True)

    df_test = pd.read_parquet(os.path.join(proc_dir, "test.parquet"))
    val_lb_path = os.path.join(reports_dir, "dataset_b_validation_leaderboard.json")
    with open(val_lb_path, "r", encoding="utf-8") as f:
        val_lb = {row['model_slug']: row for row in json.load(f)}

    meta_cols = ['target_1h', 'date_time', 'traffic_volume']
    feat_cols = [c for c in df_test.columns if c not in meta_cols]
    X_test = df_test[feat_cols].values.astype(np.float32)
    y_test = df_test['target_1h'].values.astype(np.float32)

    print(f"Dataset B Official Test samples: {len(X_test)} hours")
    print(f"Features: {len(feat_cols)}")

    model_slugs = ["persistence", "ridge", "random_forest", "hist_gradient_boosting"]
    leaderboard = []
    baseline_mae = None

    for slug in model_slugs:
        model_path = os.path.join(model_save_dir, f"{slug}.joblib")
        model = joblib.load(model_path)

        t0 = time.perf_counter()
        test_preds = model.predict(X_test).astype(np.float32)
        inference_time = time.perf_counter() - t0
        latency_ms_per_sample = (inference_time / len(X_test)) * 1000.0

        if slug == "persistence":
            metrics = calculate_metrics(y_test, test_preds)
            baseline_mae = metrics['mae']
            metrics['baseline_improvement_pct'] = 0.0
        else:
            metrics = calculate_metrics(y_test, test_preds, baseline_mae=baseline_mae)

        val_mae = val_lb[slug]['mae']
        gen_gap_mae = round(metrics['mae'] - val_mae, 2)
        rel_degradation_pct = round(((metrics['mae'] - val_mae) / val_mae) * 100.0, 2)

        entry = {
            'model_slug': slug,
            'model_name': model.model_name,
            'test_mae': round(metrics['mae'], 2),
            'test_rmse': round(metrics['rmse'], 2),
            'test_r2': round(metrics['r2'], 4),
            'baseline_improvement_pct': round(metrics['baseline_improvement_pct'], 2),
            'val_mae': round(val_mae, 2),
            'generalization_gap_mae': gen_gap_mae,
            'relative_degradation_pct': rel_degradation_pct,
            'inference_latency_ms_per_sample': round(latency_ms_per_sample, 4)
        }
        leaderboard.append(entry)

        print(f"\nResults for {model.model_name}:")
        print(f"  Test MAE:  {entry['test_mae']:.2f} veh/h  (Val MAE: {entry['val_mae']:.2f}, Gap: {gen_gap_mae:+.2f})")
        print(f"  Test RMSE: {entry['test_rmse']:.2f} veh/h")
        print(f"  Test R^2:  {entry['test_r2']:.4f}")
        print(f"  Improvement vs Persistence: {entry['baseline_improvement_pct']:.2f}%")
        print(f"  Latency: {latency_ms_per_sample * 1000.0:.2f} us/sample")

    # Export test leaderboard CSV & JSON
    df_leaderboard = pd.DataFrame(leaderboard)
    csv_path = os.path.join(reports_dir, "dataset_b_test_leaderboard.csv")
    json_path = os.path.join(reports_dir, "dataset_b_test_leaderboard.json")
    df_leaderboard.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2)

    return {
        'leaderboard': leaderboard,
        'test_samples': len(X_test)
    }


def export_final_benchmark_summary(project_root: str, res_a: Dict[str, Any], res_b: Dict[str, Any]) -> str:
    """Exports a unified JSON summary of validation vs official test performance."""
    reports_dir = os.path.join(project_root, "reports", "metrics")
    summary = {
        'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        'phase': 'Phase 5: Official Test Evaluation',
        'dataset_a': {
            'domain': 'Northern Virginia / Washington D.C. Capital Region (UCI 734)',
            'forecast_horizon': '30 minutes ahead (t + 2 steps verified)',
            'test_samples': res_a['test_samples'],
            'test_leaderboard': res_a['leaderboard']
        },
        'dataset_b': {
            'domain': 'Interstate 94 Hourly Traffic Volume (UCI 492)',
            'forecast_horizon': '1 hour ahead (calendar-verified)',
            'test_samples': res_b['test_samples'],
            'test_leaderboard': res_b['leaderboard']
        }
    }
    summary_path = os.path.join(reports_dir, "final_model_benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFinal unified benchmark summary saved to: {summary_path}")
    return summary_path


if __name__ == '__main__':
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    results_a = run_dataset_a_test_evaluation(root)
    results_b = run_dataset_b_test_evaluation(root)
    export_final_benchmark_summary(root, results_a, results_b)
    print("\n======================================================================")
    print("ALL PHASE 5 OFFICIAL TEST EVALUATIONS COMPLETED SUCCESSFULLY.")
    print("======================================================================")
