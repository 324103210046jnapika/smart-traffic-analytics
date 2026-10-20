"""Random Forest non-linear ensemble forecaster for time-series tabular modeling."""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestRegressor


class RandomForestForecaster:
    """
    Random Forest Regressor ensemble forecaster.
    Captures non-linear feature interactions and threshold effects across spatial and temporal lags.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 16,
        min_samples_split: int = 4,
        random_state: int = 42,
        n_jobs: int = -1,
        model_name: str = "Random Forest"
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model_name = model_name

        self.regressor = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestForecaster":
        """Fits ensemble of decision trees on training data."""
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        self.regressor.fit(X_arr, y_arr)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates ensemble mean prediction."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        X_arr = np.asarray(X, dtype=np.float32)
        return self.regressor.predict(X_arr).astype(np.float32)

    def get_params(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs
        }
