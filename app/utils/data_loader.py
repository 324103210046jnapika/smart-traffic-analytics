"""Data and model loading utilities with caching for Streamlit dashboard."""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import streamlit as st


def get_project_root() -> str:
    """Returns absolute path to project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@st.cache_data
def load_dataset_a_metadata() -> Dict[str, Any]:
    """Loads metadata for processed Dataset A."""
    meta_path = os.path.join(get_project_root(), "artifacts", "metadata", "dataset_a_processed_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_dataset_b_metadata() -> Dict[str, Any]:
    """Loads metadata for processed Dataset B."""
    meta_path = os.path.join(get_project_root(), "artifacts", "metadata", "dataset_b_processed_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_dataset_a_validation() -> pd.DataFrame:
    """Loads processed validation partition for Dataset A."""
    val_path = os.path.join(get_project_root(), "data", "processed", "dataset_a", "val.parquet")
    return pd.read_parquet(val_path)


@st.cache_data
def load_dataset_b_validation() -> pd.DataFrame:
    """Loads processed validation partition for Dataset B."""
    val_path = os.path.join(get_project_root(), "data", "processed", "dataset_b", "val.parquet")
    return pd.read_parquet(val_path)


@st.cache_data
def load_dataset_a_test() -> pd.DataFrame:
    """Loads processed strictly held-out test partition for Dataset A."""
    test_path = os.path.join(get_project_root(), "data", "processed", "dataset_a", "test.parquet")
    return pd.read_parquet(test_path)


@st.cache_data
def load_dataset_b_test() -> pd.DataFrame:
    """Loads processed strictly held-out test partition for Dataset B."""
    test_path = os.path.join(get_project_root(), "data", "processed", "dataset_b", "test.parquet")
    return pd.read_parquet(test_path)


@st.cache_data
def load_dataset_a_adjacency() -> np.ndarray:
    """Loads 36x36 adjacency matrix for Dataset A sensor network."""
    adj_path = os.path.join(get_project_root(), "data", "processed", "dataset_a", "adj_mat.npy")
    if os.path.exists(adj_path):
        return np.load(adj_path)
    # Fallback to zeros
    return np.zeros((36, 36), dtype=np.float32)


@st.cache_data
def load_dataset_a_leaderboard() -> pd.DataFrame:
    """Loads validation leaderboard for Dataset A."""
    path = os.path.join(get_project_root(), "reports", "metrics", "dataset_a_validation_leaderboard.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_dataset_b_leaderboard() -> pd.DataFrame:
    """Loads validation leaderboard for Dataset B."""
    path = os.path.join(get_project_root(), "reports", "metrics", "dataset_b_validation_leaderboard.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_dataset_a_per_sensor_errors() -> pd.DataFrame:
    """Loads per-sensor validation errors for Dataset A."""
    path = os.path.join(get_project_root(), "reports", "metrics", "dataset_a_per_sensor_errors.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_resource
def load_dataset_a_models() -> Dict[str, Any]:
    """Loads all trained model artifacts for Dataset A."""
    model_dir = os.path.join(get_project_root(), "artifacts", "models", "dataset_a")
    models = {}
    slugs = ["persistence", "ridge", "random_forest", "hist_gradient_boosting"]
    for slug in slugs:
        file_path = os.path.join(model_dir, f"{slug}.joblib")
        if os.path.exists(file_path):
            models[slug] = joblib.load(file_path)
    return models


@st.cache_resource
def load_dataset_b_models() -> Dict[str, Any]:
    """Loads all trained model artifacts for Dataset B."""
    model_dir = os.path.join(get_project_root(), "artifacts", "models", "dataset_b")
    models = {}
    slugs = ["persistence", "ridge", "random_forest", "hist_gradient_boosting"]
    for slug in slugs:
        file_path = os.path.join(model_dir, f"{slug}.joblib")
        if os.path.exists(file_path):
            models[slug] = joblib.load(file_path)
    return models


@st.cache_data
def load_dataset_a_test_leaderboard() -> pd.DataFrame:
    """Loads official test leaderboard for Dataset A."""
    path = os.path.join(get_project_root(), "reports", "metrics", "dataset_a_test_leaderboard.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_dataset_b_test_leaderboard() -> pd.DataFrame:
    """Loads official test leaderboard for Dataset B."""
    path = os.path.join(get_project_root(), "reports", "metrics", "dataset_b_test_leaderboard.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_dataset_a_per_sensor_test_errors() -> pd.DataFrame:
    """Loads per-sensor official test errors for Dataset A."""
    path = os.path.join(get_project_root(), "reports", "metrics", "dataset_a_per_sensor_test_errors.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_final_benchmark_summary() -> Dict[str, Any]:
    """Loads unified final benchmark summary for both datasets."""
    path = os.path.join(get_project_root(), "reports", "metrics", "final_model_benchmark_summary.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

