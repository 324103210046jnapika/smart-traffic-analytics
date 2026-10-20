"""Naive persistence heuristic baseline for time-series forecasting."""

import numpy as np
from typing import Optional, Dict, Any


class PersistenceForecaster:
    """
    Naive Persistence Baseline Forecaster:
    Forecasts future traffic using the most recent observation actually available at forecast time:
    y_hat(t + h) = y(t)
    
    In short-term forecasting, persistence is a strict benchmark that any candidate ML model must beat.
    """

    def __init__(self, persistence_col_idx: int = 9, model_name: str = "Persistence Baseline"):
        self.persistence_col_idx = persistence_col_idx
        self.model_name = model_name
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PersistenceForecaster":
        """No parameter optimization required for persistence; records training shape."""
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Extracts the latest available observation for each sample."""
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got ndim={X_arr.ndim}")
        if self.persistence_col_idx >= X_arr.shape[1]:
            raise IndexError(
                f"persistence_col_idx {self.persistence_col_idx} out of range for X shape {X_arr.shape}"
            )
        return X_arr[:, self.persistence_col_idx].copy()

    def get_params(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'persistence_col_idx': self.persistence_col_idx,
            'description': 'Predicts future measurement equals latest available historical observation'
        }
