"""Unit tests for Phase 2: chronological splitting, leakage prevention, verified horizon, and causal preprocessing."""

import os
import json
import numpy as np
import pandas as pd
import pytest

from src.data.dataset_a_loader import DatasetALoader
from src.features.dataset_a_features import DatasetAFeaturePipeline
from src.data.dataset_b_loader import DatasetBLoader
from src.features.dataset_b_features import DatasetBFeaturePipeline

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def test_dataset_a_option_a_split_boundaries():
    """Verify that Dataset A strictly adheres to Option A: 1000 train, 261 val, 840 test steps."""
    loader = DatasetALoader(RAW_DIR)
    splits = loader.prepare_chronological_splits_option_a(
        train_steps=1000,
        val_steps=261,
        test_steps=840
    )

    assert splits['train']['X_3d'].shape == (1000, 36, 48)
    assert splits['train']['Y'].shape == (36, 1000)
    assert splits['val']['X_3d'].shape == (261, 36, 48)
    assert splits['val']['Y'].shape == (36, 261)
    assert splits['test']['X_3d'].shape == (840, 36, 48)
    assert splits['test']['Y'].shape == (36, 840)

    # Verify step indices are contiguous and non-overlapping
    train_steps = splits['train']['step_indices']
    val_steps = splits['val']['step_indices']
    test_steps = splits['test']['step_indices']

    assert train_steps == list(range(0, 1000))
    assert val_steps == list(range(1000, 1261))
    assert test_steps == list(range(1261, 2101))


def test_dataset_a_multi_sensor_synchronization():
    """Verify that all 36 sensors at step T stay synchronized in the exact same split."""
    loader = DatasetALoader(RAW_DIR)
    splits = loader.prepare_chronological_splits_option_a()
    pipeline = DatasetAFeaturePipeline(fit_scaler=False)

    X_train, y_train, meta_train = pipeline.transform_split_to_tabular(splits['train'])
    X_val, y_val, meta_val = pipeline.transform_split_to_tabular(splits['val'])
    X_test, y_test, meta_test = pipeline.transform_split_to_tabular(splits['test'])

    for df_meta, expected_steps in [(meta_train, 1000), (meta_val, 261), (meta_test, 840)]:
        sensor_counts_per_step = df_meta.groupby('step_idx')['sensor_idx'].count()
        assert len(sensor_counts_per_step) == expected_steps
        assert (sensor_counts_per_step == 36).all()


def test_dataset_a_verified_target_horizon_and_zero_leakage():
    """
    Formally verifies the target horizon of Dataset A:
    - Asserts that Y[:, t] matches lag 9 at offset +2 across 100% of transitions.
    - Proves the physical horizon is 2 quarter-hours (30 minutes) ahead of the latest input feature.
    - Asserts zero contemporaneous leakage at offset 0.
    """
    loader = DatasetALoader(RAW_DIR)
    raw = loader.load_raw_matrices()
    X_tr_cells = raw['X_tr_cells']
    Y_tr = raw['Y_tr']
    T_tr = X_tr_cells.shape[1]

    # Check offset 0: contemporaneous match across all 36 sensors must be 0%
    contemporaneous_full_matches = 0
    for t in range(T_tr):
        x_t = DatasetALoader.cell_to_dense(X_tr_cells[0, t])
        if np.allclose(x_t[:, 9], Y_tr[:, t], atol=1e-5):
            contemporaneous_full_matches += 1
    assert contemporaneous_full_matches == 0, "Target Y[:, t] matches input lag[9] at offset 0 (leakage!)"

    # Check offset +2: must match lag 9 with 100% precision for all valid transitions
    offset_plus_two_matches = 0
    total_valid_transitions = T_tr - 2
    for t in range(total_valid_transitions):
        x_future = DatasetALoader.cell_to_dense(X_tr_cells[0, t + 2])
        if np.allclose(x_future[:, 9], Y_tr[:, t], atol=1e-5):
            offset_plus_two_matches += 1

    assert offset_plus_two_matches == total_valid_transitions, (
        f"Expected 100% match at offset +2, got {offset_plus_two_matches}/{total_valid_transitions}"
    )


def test_dataset_b_causal_cleaning_and_no_lookahead():
    """Verify Dataset B causal forward-fill and documented rainfall cap."""
    loader = DatasetBLoader(RAW_DIR)
    raw_df = loader.load_raw_dataframe()
    clean_df, audit_log = loader.clean_and_deduplicate(raw_df)

    # 1. Cleaned dataframe has strictly unique timestamps
    assert clean_df['date_time'].nunique() == len(clean_df)
    assert clean_df['date_time'].is_monotonic_increasing

    # 2. Causal temperature repair: no temp <= 100 K remains
    assert (clean_df['temp'] <= 100).sum() == 0

    # Verify that the repair was strictly causal forward-fill:
    # During the Jan 31, 2014 drop, temp must equal the preceding valid reading (255.93 K)
    drop_period = clean_df[(clean_df['date_time'] >= '2014-01-31 03:00:00') & 
                           (clean_df['date_time'] <= '2014-01-31 06:00:00')]
    assert len(drop_period) == 4
    assert np.allclose(drop_period['temp'].values, 255.93, atol=1e-2), (
        "Temperature during drop must match preceding known physical measurement (causal forward fill)"
    )

    # 3. Documented rainfall cap: no value > 55.63 mm remains
    assert clean_df['rain_1h'].max() <= 55.63 + 1e-5
    assert audit_log['rain_cap_mm'] == 55.63
    assert audit_log['temp_repair_strategy'] == 'causal_forward_fill_no_lookahead'


def test_dataset_b_lag_and_target_alignment():
    """Verify that lag features look strictly into the past and target looks 1 hour ahead."""
    loader = DatasetBLoader(RAW_DIR)
    raw_df = loader.load_raw_dataframe()
    clean_df, _ = loader.clean_and_deduplicate(raw_df)

    pipeline = DatasetBFeaturePipeline(fit_scaler=False)
    feat_df = pipeline.engineer_raw_features(clean_df)

    # Check alignment at arbitrary row 500
    idx = 500
    curr_time = feat_df.loc[idx, 'date_time']
    prev_time = feat_df.loc[idx - 1, 'date_time']
    next_time = feat_df.loc[idx + 1, 'date_time']

    if next_time - curr_time == pd.Timedelta(hours=1):
        assert feat_df.loc[idx, 'target_1h'] == feat_df.loc[idx + 1, 'traffic_volume']
        assert feat_df.loc[idx, 'persistence_pred'] == feat_df.loc[idx, 'traffic_volume']

    if curr_time - prev_time == pd.Timedelta(hours=1):
        assert feat_df.loc[idx, 'vol_lag1'] == feat_df.loc[idx - 1, 'traffic_volume']


def test_dataset_b_chronological_splits_and_isolation():
    """Verify chronological non-overlapping boundaries for Dataset B and train-only scaler fitting."""
    loader = DatasetBLoader(RAW_DIR)
    raw_df = loader.load_raw_dataframe()
    clean_df, _ = loader.clean_and_deduplicate(raw_df)

    pipeline = DatasetBFeaturePipeline(fit_scaler=True)
    splits = pipeline.process_splits(clean_df, train_ratio=0.70, val_ratio=0.15)

    meta_train = splits['train']['meta']
    meta_val = splits['val']['meta']
    meta_test = splits['test']['meta']

    # Chronological boundaries: train < val < test
    assert meta_train['date_time'].max() < meta_val['date_time'].min()
    assert meta_val['date_time'].max() < meta_test['date_time'].min()

    # Scaler was fitted strictly on train: scaler.n_samples_seen_ must equal len(train)
    assert pipeline.scaler.n_samples_seen_ == len(splits['train']['X']), (
        "Scaler must be fitted strictly on the training set samples"
    )


def test_raw_data_remains_unmodified():
    """Verify that raw files in data/raw/ are completely unmodified and preserved."""
    raw_mat = os.path.join(RAW_DIR, "dataset_a_uci_734", "traffic_dataset.mat")
    raw_csv = os.path.join(RAW_DIR, "dataset_b_uci_492", "Metro_Interstate_Traffic_Volume.csv")

    assert os.path.isfile(raw_mat)
    assert os.path.isfile(raw_csv)

    # Dataset B raw CSV still contains the original uncleaned 0.0 K temperature and 9831.3 mm rainfall
    df_raw = pd.read_csv(raw_csv)
    assert (df_raw['temp'] == 0.0).sum() == 10, "Raw CSV must retain original 0.0 K readings"
    assert (df_raw['rain_1h'] == 9831.3).sum() == 1, "Raw CSV must retain original 9831.3 mm rain reading"
