"""Feature engineering and tabular formatting for Dataset A (UCI 734)."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.preprocessing import StandardScaler


class DatasetAFeaturePipeline:
    """Prepares leakage-safe feature matrices and tabular representations for Dataset A."""

    def __init__(self, fit_scaler: bool = False):
        self.fit_scaler = fit_scaler
        self.scaler = StandardScaler() if fit_scaler else None
        self.is_fitted = False

    def transform_split_to_tabular(
        self,
        split_dict: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Converts 3D tensor X_3d (T, 36, 48) and 2D target Y (36, T) into tabular format:
        - Each row corresponds to a single (timestep, sensor) observation.
        - Number of samples N = T * 36.
        - Number of features = 48 (or 49 including sensor_id).
        - Target y has shape (N,).
        
        Returns:
            X_tab: shape (T * 36, 48) float32 feature matrix
            y_tab: shape (T * 36,) float32 target vector
            metadata_df: DataFrame with columns ['step_idx', 'sensor_idx', 'persistence_pred']
        """
        X_3d = split_dict['X_3d'] # (T, 36, 48)
        Y = split_dict['Y']       # (36, T)
        T, num_sensors, num_features = X_3d.shape
        step_indices = split_dict.get('step_indices', list(range(T)))

        assert Y.shape == (num_sensors, T), f"Target shape {Y.shape} mismatch with ({num_sensors}, {T})"

        # Flatten into tabular
        rows_X = []
        rows_y = []
        meta_records = []

        for t_idx, step_num in enumerate(step_indices):
            x_t = X_3d[t_idx] # (36, 48)
            y_t = Y[:, t_idx] # (36,)
            
            for s in range(num_sensors):
                feat_vec = x_t[s, :]
                target_val = y_t[s]
                # Persistence baseline value is lag 9 (the most recent historical lag)
                persistence_val = feat_vec[9]

                rows_X.append(feat_vec)
                rows_y.append(target_val)
                meta_records.append({
                    'step_idx': step_num,
                    'sensor_idx': s,
                    'persistence_pred': persistence_val
                })

        X_tab = np.array(rows_X, dtype=np.float32)
        y_tab = np.array(rows_y, dtype=np.float32)
        meta_df = pd.DataFrame(meta_records)

        return X_tab, y_tab, meta_df

    def fit_transform_train(
        self,
        train_split: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Fits scalers (if enabled) strictly on training split and transforms features."""
        X_train, y_train, meta_train = self.transform_split_to_tabular(train_split)

        if self.fit_scaler:
            X_train = self.scaler.fit_transform(X_train)
            self.is_fitted = True

        return X_train, y_train, meta_train

    def transform(
        self,
        split_dict: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Transforms validation or test split using parameters fitted strictly on train."""
        X_out, y_out, meta_out = self.transform_split_to_tabular(split_dict)

        if self.fit_scaler:
            if not self.is_fitted:
                raise RuntimeError("Scaler is not fitted. Call fit_transform_train first.")
            X_out = self.scaler.transform(X_out)

        return X_out, y_out, meta_out
