"""Automated data integrity and schema verification tests for Phase 1."""

import os
import scipy.io
import pandas as pd
import pytest

RAW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))


def test_dataset_a_raw_files_exist():
    """Verify Dataset A (UCI 734) files are downloaded and extracted."""
    dir_a = os.path.join(RAW_ROOT, "dataset_a_uci_734")
    mat_file = os.path.join(dir_a, "traffic_dataset.mat")
    doc_file = os.path.join(dir_a, "Traffic Flow Prediction Dataset.docx")
    
    assert os.path.isdir(dir_a), "Dataset A directory missing"
    assert os.path.isfile(mat_file), "traffic_dataset.mat missing"
    assert os.path.isfile(doc_file), "Dataset A documentation missing"


def test_dataset_a_mat_structure_and_dimensions():
    """Verify Dataset A matrices conform to verified specifications."""
    mat_file = os.path.join(RAW_ROOT, "dataset_a_uci_734", "traffic_dataset.mat")
    mat = scipy.io.loadmat(mat_file)
    
    for key in ['tra_X_tr', 'tra_X_te', 'tra_Y_tr', 'tra_Y_te', 'tra_adj_mat']:
        assert key in mat, f"Missing matrix {key} in traffic_dataset.mat"

    # Check shapes
    assert mat['tra_X_tr'].shape == (1, 1261), "tra_X_tr unexpected shape"
    assert mat['tra_X_te'].shape == (1, 840), "tra_X_te unexpected shape"
    assert mat['tra_Y_tr'].shape == (36, 1261), "tra_Y_tr unexpected shape"
    assert mat['tra_Y_te'].shape == (36, 840), "tra_Y_te unexpected shape"
    assert mat['tra_adj_mat'].shape == (36, 36), "tra_adj_mat unexpected shape"

    # Target bounds [0, 1]
    assert mat['tra_Y_tr'].min() >= 0.0 and mat['tra_Y_tr'].max() <= 1.0, "tra_Y_tr out of bounds [0, 1]"
    assert mat['tra_Y_te'].min() >= 0.0 and mat['tra_Y_te'].max() <= 1.0, "tra_Y_te out of bounds [0, 1]"


def test_dataset_b_raw_files_exist_and_schema():
    """Verify Dataset B (UCI 492) CSV exists and has expected columns."""
    dir_b = os.path.join(RAW_ROOT, "dataset_b_uci_492")
    csv_file = os.path.join(dir_b, "Metro_Interstate_Traffic_Volume.csv")
    
    assert os.path.isdir(dir_b), "Dataset B directory missing"
    assert os.path.isfile(csv_file), "Metro_Interstate_Traffic_Volume.csv missing"

    df = pd.read_csv(csv_file, nrows=100)
    expected_cols = [
        'holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all',
        'weather_main', 'weather_description', 'date_time', 'traffic_volume'
    ]
    assert list(df.columns) == expected_cols, f"Columns mismatch in Dataset B: {list(df.columns)}"


def test_datasets_are_strictly_isolated():
    """Verify that directories and files are isolated without cross-pollution."""
    dir_a = os.path.join(RAW_ROOT, "dataset_a_uci_734")
    dir_b = os.path.join(RAW_ROOT, "dataset_b_uci_492")
    
    files_a = os.listdir(dir_a)
    files_b = os.listdir(dir_b)
    
    assert not any("Metro" in f for f in files_a), "Dataset B file found in Dataset A directory!"
    assert not any("traffic_dataset" in f for f in files_b), "Dataset A file found in Dataset B directory!"
