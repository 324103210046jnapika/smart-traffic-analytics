"""Data loading, validation, and chronological split preparation for Dataset A (UCI 734)."""

import os
import scipy.io
import scipy.sparse
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any


class DatasetALoader:
    """Loads and chronologically partitions UCI 734 Traffic Flow Forecasting dataset."""

    def __init__(self, raw_data_dir: str):
        self.raw_data_dir = raw_data_dir
        self.mat_path = os.path.join(raw_data_dir, "dataset_a_uci_734", "traffic_dataset.mat")

    def load_raw_matrices(self) -> Dict[str, Any]:
        """Loads the raw matrices from traffic_dataset.mat with validation."""
        if not os.path.isfile(self.mat_path):
            raise FileNotFoundError(f"traffic_dataset.mat not found at {self.mat_path}")

        mat = scipy.io.loadmat(self.mat_path)
        required_keys = ['tra_X_tr', 'tra_X_te', 'tra_Y_tr', 'tra_Y_te', 'tra_adj_mat']
        for k in required_keys:
            if k not in mat:
                raise KeyError(f"Expected key '{k}' missing from traffic_dataset.mat")

        return {
            'X_tr_cells': mat['tra_X_tr'], # Shape (1, 1261) cell array
            'X_te_cells': mat['tra_X_te'], # Shape (1, 840) cell array
            'Y_tr': mat['tra_Y_tr'],       # Shape (36, 1261) dense float64
            'Y_te': mat['tra_Y_te'],       # Shape (36, 840) dense float64
            'adj_mat': mat['tra_adj_mat']  # Shape (36, 36) uint8 binary
        }

    @staticmethod
    def cell_to_dense(cell: Any) -> np.ndarray:
        """Converts a cell containing a sparse CSC array or matrix into a dense float32 array."""
        if scipy.sparse.issparse(cell):
            return cell.toarray().astype(np.float32)
        return np.asarray(cell, dtype=np.float32)

    def prepare_chronological_splits_option_a(
        self,
        train_steps: int = 1000,
        val_steps: int = 261,
        test_steps: int = 840
    ) -> Dict[str, Any]:
        """
        Partitions Dataset A using Option A:
        - Subdivides the 1,261 author training steps into:
          * Train: steps 0 to 999 (1,000 steps)
          * Validation: steps 1,000 to 1,260 (261 steps)
        - Preserves the official 840 author test steps strictly untouched as the final test set.
        """
        raw = self.load_raw_matrices()
        X_tr_cells = raw['X_tr_cells']
        Y_tr = raw['Y_tr']
        X_te_cells = raw['X_te_cells']
        Y_te = raw['Y_te']
        adj_mat = raw['adj_mat']

        total_tr_steps = X_tr_cells.shape[1]
        total_te_steps = X_te_cells.shape[1]

        assert total_tr_steps == 1261, f"Expected 1261 train steps, got {total_tr_steps}"
        assert total_te_steps == 840, f"Expected 840 test steps, got {total_te_steps}"
        assert train_steps + val_steps == total_tr_steps, (
            f"train_steps ({train_steps}) + val_steps ({val_steps}) must equal {total_tr_steps}"
        )
        assert test_steps == total_te_steps, (
            f"test_steps ({test_steps}) must equal official author test steps ({total_te_steps})"
        )

        # 1. Training split: [0 : train_steps]
        train_X_cells = [self.cell_to_dense(X_tr_cells[0, t]) for t in range(train_steps)]
        train_Y = Y_tr[:, :train_steps].astype(np.float32)

        # 2. Validation split: [train_steps : train_steps + val_steps]
        val_X_cells = [self.cell_to_dense(X_tr_cells[0, t]) for t in range(train_steps, total_tr_steps)]
        val_Y = Y_tr[:, train_steps:total_tr_steps].astype(np.float32)

        # 3. Test split: official 840 test steps [0 : 840] of X_te
        test_X_cells = [self.cell_to_dense(X_te_cells[0, t]) for t in range(test_steps)]
        test_Y = Y_te[:, :test_steps].astype(np.float32)

        return {
            'train': {
                'X_3d': np.stack(train_X_cells, axis=0), # Shape: (1000, 36, 48)
                'Y': train_Y,                           # Shape: (36, 1000)
                'steps': train_steps,
                'step_indices': list(range(0, train_steps))
            },
            'val': {
                'X_3d': np.stack(val_X_cells, axis=0),   # Shape: (261, 36, 48)
                'Y': val_Y,                             # Shape: (36, 261)
                'steps': val_steps,
                'step_indices': list(range(train_steps, total_tr_steps))
            },
            'test': {
                'X_3d': np.stack(test_X_cells, axis=0),  # Shape: (840, 36, 48)
                'Y': test_Y,                            # Shape: (36, 840)
                'steps': test_steps,
                'step_indices': list(range(total_tr_steps, total_tr_steps + test_steps))
            },
            'adj_mat': adj_mat,
            'metadata': {
                'num_sensors': 36,
                'features_per_sensor': 48,
                'split_strategy': 'Option A (Author Benchmark-Aligned)',
                'train_steps': train_steps,
                'val_steps': val_steps,
                'test_steps': test_steps,
                'total_steps': train_steps + val_steps + test_steps,
                'sampling_interval_minutes': 15
            }
        }
