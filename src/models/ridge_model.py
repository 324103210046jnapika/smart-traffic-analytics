"""Regularized Linear Regression (Ridge) model wrapper for time-series forecasting."""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class RidgeForecaster:
    """
    Ridge Regression forecaster with internal StandardScaler in a leakage-safe Pipeline.
    Applies L2 shrinkage penalty to prevent overfitting on collinear lag features.
    """

    def __init__(self, alpha: float = 1.0, random_state: int = 42, model_name: str = "Ridge Regression"):
        self.alpha = alpha
        self.random_state = random_state
        self.model_name = model_name
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=self.alpha, random_state=self.random_state))
        ])
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeForecaster":
        """Fits scaler and Ridge regression coefficients strictly on training data."""
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        self.pipeline.fit(X_arr, y_arr)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates predictions using fitted pipeline."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        X_arr = np.asarray(X, dtype=np.float32)
        return self.pipeline.predict(X_arr).astype(np.float32)

    def get_params(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'alpha': self.alpha,
            'random_state': self.random_state,
            'scaler': 'StandardScaler'
        }
