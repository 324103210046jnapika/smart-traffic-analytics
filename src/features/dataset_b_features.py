"""Feature engineering pipeline for Dataset B (UCI 492 Metro Interstate Traffic Volume)."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Any
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class DatasetBFeaturePipeline:
    """Creates leakage-safe lag, rolling, cyclic, and weather features for Dataset B."""

    def __init__(self, fit_scaler: bool = True):
        self.fit_scaler = fit_scaler
        self.scaler = StandardScaler() if fit_scaler else None
        self.ohe_weather = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.is_fitted = False
        self.feature_names: List[str] = []

    def engineer_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates backward-looking features on a clean hourly DataFrame:
        - Target: strictly 1-hour-ahead traffic volume (t + 1h), verified by calendar timestamp.
          Rows where the succeeding row is NOT exactly 1 hour ahead are assigned NaN and excluded.
        - Historical traffic lags: t, t-1, t-2, t-3, t-24, t-168
        - Rolling means & stds using strictly past observations up to time t (closed='left')
        - Cyclic sinusoidal encodings for hour, day of week, and month
        - Weather observations at time t and lagged weather (t-1)
        """
        feat_df = df.copy()

        # 1. Target: strictly 1-hour-ahead traffic volume
        # Verify that row t+1 is exactly 1 hour after row t
        next_dt = feat_df['date_time'].shift(-1)
        delta_to_next = next_dt - feat_df['date_time']
        is_exact_1h = (delta_to_next == pd.Timedelta(hours=1))

        # Assign target ONLY when next row is exactly 1 hour ahead
        raw_next_vol = feat_df['traffic_volume'].shift(-1)
        feat_df['target_1h'] = np.where(is_exact_1h, raw_next_vol, np.nan)
        feat_df['is_valid_1h_target'] = is_exact_1h

        # 2. Persistence baseline prediction: current traffic volume y(t)
        feat_df['persistence_pred'] = feat_df['traffic_volume'].copy()

        # 3. Traffic volume lags (strictly past data)
        feat_df['vol_lag1'] = feat_df['traffic_volume'].shift(1)
        feat_df['vol_lag2'] = feat_df['traffic_volume'].shift(2)
        feat_df['vol_lag3'] = feat_df['traffic_volume'].shift(3)
        feat_df['vol_lag24'] = feat_df['traffic_volume'].shift(24)
        feat_df['vol_lag168'] = feat_df['traffic_volume'].shift(168)

        # 4. Rolling statistics (calculated over past windows ending at t)
        feat_df['vol_roll_mean_3h'] = feat_df['traffic_volume'].rolling(window=3, closed='left').mean()
        feat_df['vol_roll_std_3h'] = feat_df['traffic_volume'].rolling(window=3, closed='left').std()
        feat_df['vol_roll_mean_24h'] = feat_df['traffic_volume'].rolling(window=24, closed='left').mean()
        feat_df['vol_roll_std_24h'] = feat_df['traffic_volume'].rolling(window=24, closed='left').std()

        # 5. Cyclic calendar features
        hours = feat_df['date_time'].dt.hour
        dayofweek = feat_df['date_time'].dt.dayofweek
        months = feat_df['date_time'].dt.month

        feat_df['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
        feat_df['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)
        feat_df['day_sin'] = np.sin(2 * np.pi * dayofweek / 7.0)
        feat_df['day_cos'] = np.cos(2 * np.pi * dayofweek / 7.0)
        feat_df['month_sin'] = np.sin(2 * np.pi * (months - 1) / 12.0)
        feat_df['month_cos'] = np.cos(2 * np.pi * (months - 1) / 12.0)

        # 6. Weather features at time t and t-1 (no future weather used!)
        feat_df['weather_temp'] = feat_df['temp']
        feat_df['weather_rain_1h'] = feat_df['rain_1h']
        feat_df['weather_snow_1h'] = feat_df['snow_1h']
        feat_df['weather_clouds'] = feat_df['clouds_all']
        feat_df['temp_lag1'] = feat_df['temp'].shift(1)
        feat_df['rain_lag1'] = feat_df['rain_1h'].shift(1)

        # 7. Is holiday flag
        feat_df['is_holiday'] = (feat_df['holiday'] != 'None').astype(float)

        return feat_df

    def process_splits(
        self,
        clean_df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15
    ) -> Dict[str, Any]:
        """
        Builds feature matrices across chronological splits:
        1. Computes features.
        2. Drops warmup rows (first 168 rows needed for week-long lag).
        3. Drops invalid target pairs (where delta_to_next != 1 hour, including final row).
        4. Splits chronologically into train, val, and test.
        5. Fits scaler and weather one-hot encoder EXCLUSIVELY on training data.
        """
        feat_df = self.engineer_raw_features(clean_df)

        # Count how many target pairs were invalid due to sensor gaps
        invalid_gap_targets = int((~feat_df['is_valid_1h_target']).sum())

        # Drop initial warmup window and rows where target_1h is invalid
        valid_df = feat_df.dropna(subset=['target_1h', 'vol_lag168', 'vol_roll_std_24h']).copy()
        valid_df = valid_df.reset_index(drop=True)

        n_total = len(valid_df)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        train_raw = valid_df.iloc[:n_train].copy().reset_index(drop=True)
        val_raw = valid_df.iloc[n_train:n_train + n_val].copy().reset_index(drop=True)
        test_raw = valid_df.iloc[n_train + n_val:].copy().reset_index(drop=True)

        numerical_cols = [
            'persistence_pred', 'vol_lag1', 'vol_lag2', 'vol_lag3', 'vol_lag24', 'vol_lag168',
            'vol_roll_mean_3h', 'vol_roll_std_3h', 'vol_roll_mean_24h', 'vol_roll_std_24h',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'weather_temp', 'weather_rain_1h', 'weather_snow_1h', 'weather_clouds',
            'temp_lag1', 'rain_lag1', 'is_holiday'
        ]

        # Fit encoders ONLY on train
        self.ohe_weather.fit(train_raw[['weather_main']])
        weather_cat_cols = list(self.ohe_weather.get_feature_names_out(['weather_main']))

        def transform_features(split_df: pd.DataFrame) -> np.ndarray:
            num_mat = split_df[numerical_cols].values.astype(np.float32)
            cat_mat = self.ohe_weather.transform(split_df[['weather_main']]).astype(np.float32)
            return np.hstack([num_mat, cat_mat])

        X_train_raw = transform_features(train_raw)
        X_val_raw = transform_features(val_raw)
        X_test_raw = transform_features(test_raw)

        self.feature_names = numerical_cols + weather_cat_cols

        if self.fit_scaler:
            X_train = self.scaler.fit_transform(X_train_raw).astype(np.float32)
            X_val = self.scaler.transform(X_val_raw).astype(np.float32)
            X_test = self.scaler.transform(X_test_raw).astype(np.float32)
            self.is_fitted = True
        else:
            X_train, X_val, X_test = X_train_raw, X_val_raw, X_test_raw

        y_train = train_raw['target_1h'].values.astype(np.float32)
        y_val = val_raw['target_1h'].values.astype(np.float32)
        y_test = test_raw['target_1h'].values.astype(np.float32)

        return {
            'train': {
                'X': X_train,
                'y': y_train,
                'meta': train_raw[['date_time', 'persistence_pred', 'traffic_volume', 'target_1h']]
            },
            'val': {
                'X': X_val,
                'y': y_val,
                'meta': val_raw[['date_time', 'persistence_pred', 'traffic_volume', 'target_1h']]
            },
            'test': {
                'X': X_test,
                'y': y_test,
                'meta': test_raw[['date_time', 'persistence_pred', 'traffic_volume', 'target_1h']]
            },
            'feature_names': self.feature_names,
            'metadata': {
                'total_samples': n_total,
                'train_samples': len(train_raw),
                'val_samples': len(val_raw),
                'test_samples': len(test_raw),
                'invalid_gap_targets_excluded': invalid_gap_targets,
                'num_features': len(self.feature_names)
            }
        }
