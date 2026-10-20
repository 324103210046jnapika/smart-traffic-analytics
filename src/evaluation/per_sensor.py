"""Per-sensor evaluation breakdown and spatial distribution analysis for Dataset A."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from src.evaluation.metrics import calculate_metrics


def evaluate_per_sensor(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_persistence: np.ndarray,
    sensor_indices: np.ndarray
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Computes accuracy metrics separately for each traffic sensor station:
    
    Args:
        y_true: Ground truth target array (N,)
        y_pred: Model prediction array (N,)
        y_persistence: Persistence baseline prediction array (N,)
        sensor_indices: Sensor ID array (N,) corresponding to each sample
        
    Returns:
        per_sensor_df: DataFrame of metrics for each of the 36 sensors
        distribution_summary: Dict of summary statistics across all sensors
    """
    df = pd.DataFrame({
        'sensor_idx': sensor_indices,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_persistence': y_persistence
    })

    records = []
    unique_sensors = sorted(df['sensor_idx'].unique())

    for s in unique_sensors:
        sub = df[df['sensor_idx'] == s]
        pers_metrics = calculate_metrics(sub['y_true'], sub['y_persistence'])
        base_mae = pers_metrics['mae']

        model_metrics = calculate_metrics(sub['y_true'], sub['y_pred'], baseline_mae=base_mae)

        records.append({
            'sensor_idx': int(s),
            'samples': len(sub),
            'mae': model_metrics['mae'],
            'rmse': model_metrics['rmse'],
            'r2': model_metrics['r2'],
            'persistence_mae': base_mae,
            'baseline_improvement_pct': model_metrics['baseline_improvement_pct']
        })

    per_sensor_df = pd.DataFrame(records)

    # Compute distribution summary across sensors
    distribution_summary = {
        'num_sensors_evaluated': len(unique_sensors),
        'mae_min': float(per_sensor_df['mae'].min()),
        'mae_q25': float(per_sensor_df['mae'].quantile(0.25)),
        'mae_median': float(per_sensor_df['mae'].median()),
        'mae_q75': float(per_sensor_df['mae'].quantile(0.75)),
        'mae_max': float(per_sensor_df['mae'].max()),
        'mae_mean': float(per_sensor_df['mae'].mean()),
        'mae_std': float(per_sensor_df['mae'].std()),
        'median_baseline_improvement_pct': float(per_sensor_df['baseline_improvement_pct'].median())
    }

    return per_sensor_df, distribution_summary
