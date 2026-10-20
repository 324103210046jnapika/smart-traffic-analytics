"""Metric calculation utilities for time-series traffic forecasting evaluation."""

import numpy as np
from typing import Dict, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_mae: Optional[float] = None
) -> Dict[str, float]:
    """
    Computes standard regression evaluation metrics:
    - MAE: Mean Absolute Error (physical unit accuracy, robust to extreme outliers)
    - RMSE: Root Mean Squared Error (penalizes large mispredictions)
    - R2: Coefficient of Determination (explained variance relative to historical mean)
    - baseline_improvement_pct: Percentage improvement over naive persistence heuristic
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(f"Length mismatch: y_true ({len(y_true_arr)}) vs y_pred ({len(y_pred_arr)})")

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    mse = float(mean_squared_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mse))

    # Guard against zero-variance target when computing R2
    target_var = float(np.var(y_true_arr))
    if target_var < 1e-9:
        r2 = 0.0
    else:
        r2 = float(r2_score(y_true_arr, y_pred_arr))

    results = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

    if baseline_mae is not None and baseline_mae > 1e-9:
        improvement = ((baseline_mae - mae) / baseline_mae) * 100.0
        results['baseline_improvement_pct'] = float(improvement)
    else:
        results['baseline_improvement_pct'] = 0.0

    return results
