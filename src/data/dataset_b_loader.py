"""Data loading, causal cleaning, deduplication, and chronological splitting for Dataset B (UCI 492)."""

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any


class DatasetBLoader:
    """Loads, causally cleans, and chronologically splits UCI 492 Metro Interstate Traffic Volume dataset."""

    def __init__(self, raw_data_dir: str):
        self.raw_data_dir = raw_data_dir
        self.csv_path = os.path.join(raw_data_dir, "dataset_b_uci_492", "Metro_Interstate_Traffic_Volume.csv")

    def load_raw_dataframe(self) -> pd.DataFrame:
        """Loads the raw CSV into a pandas DataFrame without modifying the source file."""
        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(f"Metro_Interstate_Traffic_Volume.csv not found at {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        required_cols = [
            'holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all',
            'weather_main', 'weather_description', 'date_time', 'traffic_volume'
        ]
        for c in required_cols:
            if c not in df.columns:
                raise KeyError(f"Expected column '{c}' missing from Dataset B")

        df['date_time'] = pd.to_datetime(df['date_time'])
        return df

    def clean_and_deduplicate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Cleans Dataset B strictly respecting causality and preventing future lookahead leakage:

        1. Duplicate Timestamps (Concurrent Weather Phenomena):
           - OpenWeatherMap records multiple weather descriptions occurring in the same 60-minute window.
           - In 100% of duplicate timestamp instances, traffic_volume is identical (0 variance).
           - Aggregating rows sharing the identical timestamp t uses ZERO future information because
             all duplicates represent the exact same historical hour t.
           - Rule: traffic_volume uses 'first', numerical weather uses 'median' of contemporaneous readings,
             and weather_main selects the highest-severity meteorological event occurring in that hour
             (Squall > Thunderstorm > Snow > Rain > Drizzle > Fog/Mist/Haze > Clouds > Clear).

        2. Causal Temperature Glitch Repair (temp <= 100 K):
           - Exactly 10 rows in Jan/Feb 2014 logged 0.0 K (-273.15 C) due to temporary sensor drops.
           - LEAKAGE PREVENTION: We do NOT use two-sided interpolation (which peeks into future readings).
             Instead, we use strictly causal forward-fill (.ffill()) from the preceding valid winter
             temperature (~255.93 K / -17 C). This relies strictly on past known physical measurements.

        3. Rainfall Anomaly Handling (rain_1h == 9831.3 mm):
           - Exactly 1 row on 2016-07-11 17:00:00 logged 9,831.3 mm (nearly 10 meters of rain in 1 hour).
           - World record 1-hour rainfall is 305 mm; Minnesota state record is 132 mm.
           - Maximum valid non-anomalous 1-hour rainfall in the remaining 48,203 rows is 55.63 mm.
           - Rule: Capped to the verified non-anomalous maximum (55.63 mm), preserving row continuity
             without introducing an extreme distortive outlier into linear or distance-based models.
        """
        initial_rows = len(df)
        # Ensure chronological sorting before any operation
        df_sorted = df.sort_values('date_time').copy()

        # Audit anomalies before cleaning
        zero_temp_count = int((df_sorted['temp'] <= 100).sum())
        extreme_rain_count = int((df_sorted['rain_1h'] > 100).sum())

        # Severity hierarchy for consolidating concurrent hourly weather events
        severity_order = {
            'Squall': 10,
            'Thunderstorm': 9,
            'Snow': 8,
            'Rain': 7,
            'Drizzle': 6,
            'Fog': 5,
            'Mist': 4,
            'Haze': 3,
            'Smoke': 2,
            'Clouds': 1,
            'Clear': 0
        }

        def pick_most_severe(series: pd.Series) -> str:
            return max(series, key=lambda w: severity_order.get(w, 0))

        # Deduplicate contemporaneous readings
        agg_rules = {
            'traffic_volume': 'first',
            'temp': 'median',
            'rain_1h': 'median',
            'snow_1h': 'median',
            'clouds_all': 'median',
            'weather_main': pick_most_severe,
            'weather_description': 'first',
            'holiday': 'first'
        }
        df_clean = df_sorted.groupby('date_time', as_index=False).agg(agg_rules)
        dedup_rows = len(df_clean)

        # Strictly causal repair: forward-fill from past valid temperature
        temp_mask = df_clean['temp'] <= 100
        df_clean.loc[temp_mask, 'temp'] = np.nan
        df_clean['temp'] = df_clean['temp'].ffill()

        # Causal rainfall cap: capped to verified maximum non-anomalous rain (55.63 mm)
        rain_mask = df_clean['rain_1h'] > 100
        df_clean.loc[rain_mask, 'rain_1h'] = 55.63

        # Standardize holiday missing values to 'None'
        df_clean['holiday'] = df_clean['holiday'].fillna('None')

        audit_log = {
            'initial_rows': initial_rows,
            'unique_hourly_rows': dedup_rows,
            'duplicate_rows_collapsed': initial_rows - dedup_rows,
            'zero_temp_corrections': zero_temp_count,
            'extreme_rain_corrections': extreme_rain_count,
            'rain_cap_mm': 55.63,
            'temp_repair_strategy': 'causal_forward_fill_no_lookahead',
            'min_date': str(df_clean['date_time'].min()),
            'max_date': str(df_clean['date_time'].max())
        }

        return df_clean, audit_log

    def prepare_chronological_splits(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Dict[str, Any]:
        """Chronologically partitions cleaned Dataset B into train, val, and test splits."""
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5

        raw_df = self.load_raw_dataframe()
        clean_df, audit_log = self.clean_and_deduplicate(raw_df)

        n_total = len(clean_df)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        train_df = clean_df.iloc[:n_train].copy().reset_index(drop=True)
        val_df = clean_df.iloc[n_train:n_train + n_val].copy().reset_index(drop=True)
        test_df = clean_df.iloc[n_train + n_val:].copy().reset_index(drop=True)

        return {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'audit_log': audit_log,
            'metadata': {
                'total_rows': n_total,
                'train_rows': len(train_df),
                'val_rows': len(val_df),
                'test_rows': len(test_df),
                'train_date_range': (str(train_df['date_time'].min()), str(train_df['date_time'].max())),
                'val_date_range': (str(val_df['date_time'].min()), str(val_df['date_time'].max())),
                'test_date_range': (str(test_df['date_time'].min()), str(test_df['date_time'].max()))
            }
        }
