"""Histogram-based Gradient Boosting forecaster optimized for fast CPU time-series regression."""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import HistGradientBoostingRegressor


class HistGradientBoostingForecaster:
    """
    Histogram-Based Gradient Boosting Regressor.
    Bins continuous numerical features into 256 integer bins, dramatically reducing memory bandwidth
    and accelerating tree building on multi-core CPUs.
    """

    def __init__(
        self,
        max_iter: int = 100,
        learning_rate: float = 0.1,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        random_state: int = 42,
        model_name: str = "HistGradientBoosting"
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model_name = model_name

        self.regressor = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HistGradientBoostingForecaster":
        """Fits gradient boosting trees on training data."""
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        self.regressor.fit(X_arr, y_arr)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates gradient boosted prediction."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        X_arr = np.asarray(X, dtype=np.float32)
        return self.regressor.predict(X_arr).astype(np.float32)

    def get_params(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'max_iter': self.max_iter,
            'learning_rate': self.learning_rate,
            'max_leaf_nodes': self.max_leaf_nodes,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state
        }
