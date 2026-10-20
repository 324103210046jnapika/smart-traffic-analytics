"""CLI execution script to build, validate, and save processed chronological datasets."""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.data.dataset_a_loader import DatasetALoader
from src.features.dataset_a_features import DatasetAFeaturePipeline
from src.data.dataset_b_loader import DatasetBLoader
from src.features.dataset_b_features import DatasetBFeaturePipeline


def build_and_save_dataset_a(project_root: str) -> Dict[str, Any]:
    """Loads, partitions (Option A), transforms, and saves processed Dataset A."""
    print("=== Processing Dataset A (UCI 734) ===")
    raw_dir = os.path.join(project_root, "data", "raw")
    proc_dir = os.path.join(project_root, "data", "processed", "dataset_a")
    meta_dir = os.path.join(project_root, "artifacts", "metadata")
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    loader = DatasetALoader(raw_dir)
    splits = loader.prepare_chronological_splits_option_a(
        train_steps=1000,
        val_steps=261,
        test_steps=840
    )

    pipeline = DatasetAFeaturePipeline(fit_scaler=False)
    X_train, y_train, meta_train = pipeline.transform_split_to_tabular(splits['train'])
    X_val, y_val, meta_val = pipeline.transform_split_to_tabular(splits['val'])
    X_test, y_test, meta_test = pipeline.transform_split_to_tabular(splits['test'])

    # Build DataFrames for Parquet export
    feat_cols = [f"feat_{i:02d}" for i in range(48)]

    def make_df(X_mat, y_vec, meta):
        df = pd.DataFrame(X_mat, columns=feat_cols)
        df['target'] = y_vec
        df['step_idx'] = meta['step_idx'].values
        df['sensor_idx'] = meta['sensor_idx'].values
        df['persistence_pred'] = meta['persistence_pred'].values
        return df

    df_train = make_df(X_train, y_train, meta_train)
    df_val = make_df(X_val, y_val, meta_val)
    df_test = make_df(X_test, y_test, meta_test)

    train_path = os.path.join(proc_dir, "train.parquet")
    val_path = os.path.join(proc_dir, "val.parquet")
    test_path = os.path.join(proc_dir, "test.parquet")
    adj_path = os.path.join(proc_dir, "adj_mat.npy")

    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)
    df_test.to_parquet(test_path, index=False)
    np.save(adj_path, splits['adj_mat'])

    metadata = {
        'dataset': 'Dataset A (UCI 734 Traffic Flow Forecasting)',
        'split_strategy': 'Option A (Author Benchmark-Aligned)',
        'train_samples': len(df_train),
        'val_samples': len(df_val),
        'test_samples': len(df_test),
        'num_sensors': 36,
        'features_per_sensor': 48,
        'train_steps': 1000,
        'val_steps': 261,
        'test_steps': 840,
        'total_steps': 2101,
        'target_range': [float(df_train['target'].min()), float(df_train['target'].max())],
        'artifacts_saved': {
            'train_parquet': train_path,
            'val_parquet': val_path,
            'test_parquet': test_path,
            'adj_mat_npy': adj_path
        }
    }

    meta_file = os.path.join(meta_dir, "dataset_a_processed_metadata.json")
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset A processed: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
    return metadata


def build_and_save_dataset_b(project_root: str) -> Dict[str, Any]:
    """Loads, cleans, creates features, and saves processed Dataset B."""
    print("\n=== Processing Dataset B (UCI 492) ===")
    raw_dir = os.path.join(project_root, "data", "raw")
    proc_dir = os.path.join(project_root, "data", "processed", "dataset_b")
    meta_dir = os.path.join(project_root, "artifacts", "metadata")
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    loader = DatasetBLoader(raw_dir)
    raw_df = loader.load_raw_dataframe()
    clean_df, audit_log = loader.clean_and_deduplicate(raw_df)

    pipeline = DatasetBFeaturePipeline(fit_scaler=False)
    processed = pipeline.process_splits(clean_df, train_ratio=0.70, val_ratio=0.15)

    feat_names = processed['feature_names']

    def make_df(split_dict):
        df = pd.DataFrame(split_dict['X'], columns=feat_names)
        df['target_1h'] = split_dict['y']
        meta = split_dict['meta']
        for col in meta.columns:
            df[col] = meta[col].values
        return df

    df_train = make_df(processed['train'])
    df_val = make_df(processed['val'])
    df_test = make_df(processed['test'])

    train_path = os.path.join(proc_dir, "train.parquet")
    val_path = os.path.join(proc_dir, "val.parquet")
    test_path = os.path.join(proc_dir, "test.parquet")

    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)
    df_test.to_parquet(test_path, index=False)

    metadata = {
        'dataset': 'Dataset B (UCI 492 Metro Interstate Traffic Volume)',
        'split_strategy': 'Chronological 70% / 15% / 15%',
        'audit_log': audit_log,
        'train_samples': len(df_train),
        'val_samples': len(df_val),
        'test_samples': len(df_test),
        'num_features': len(feat_names),
        'invalid_gap_targets_excluded': processed['metadata']['invalid_gap_targets_excluded'],
        'feature_list': feat_names,
        'target_range': [float(df_train['target_1h'].min()), float(df_train['target_1h'].max())],
        'artifacts_saved': {
            'train_parquet': train_path,
            'val_parquet': val_path,
            'test_parquet': test_path
        }
    }

    meta_file = os.path.join(meta_dir, "dataset_b_processed_metadata.json")
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset B processed: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
    print(f"Dataset B invalid gap targets excluded: {processed['metadata']['invalid_gap_targets_excluded']}")
    return metadata


if __name__ == '__main__':
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    meta_a = build_and_save_dataset_a(root)
    meta_b = build_and_save_dataset_b(root)
    print("\nAll data processing pipelines completed successfully.")
