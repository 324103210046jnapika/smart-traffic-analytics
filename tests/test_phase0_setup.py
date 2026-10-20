"""Unit and smoke tests for Phase 0 setup and repository integrity."""

import os
import yaml
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_required_directories_exist():
    """Verify that all foundational project directories are present."""
    required_dirs = [
        "data/raw",
        "data/processed",
        "configs",
        "docs",
        "src/data",
        "src/features",
        "src/models",
        "src/evaluation",
        "src/forecasting",
        "src/monitoring",
        "src/reporting",
        "app/pages",
        "app/components",
        "artifacts/models",
        "artifacts/metadata",
        "reports/figures",
        "reports/metrics",
        "tests",
        "notebooks",
    ]
    for rel_dir in required_dirs:
        full_path = os.path.join(PROJECT_ROOT, rel_dir.replace("/", os.sep))
        assert os.path.isdir(full_path), f"Required directory missing: {rel_dir}"


def test_required_documentation_files_exist():
    """Verify that all required documentation files exist and are not empty."""
    required_docs = [
        "README.md",
        "data/README.md",
        "docs/PROJECT_SPECIFICATION.md",
        "docs/DATASET_PLAN.md",
        "docs/VALIDATION_AND_METRICS.md",
        "docs/ARCHITECTURE.md",
        "docs/RISKS_AND_LIMITATIONS.md",
    ]
    for rel_doc in required_docs:
        full_path = os.path.join(PROJECT_ROOT, rel_doc.replace("/", os.sep))
        assert os.path.isfile(full_path), f"Required documentation file missing: {rel_doc}"
        assert os.path.getsize(full_path) > 100, f"Documentation file appears empty or incomplete: {rel_doc}"


def test_base_configuration_validity():
    """Verify that configs/base_config.yaml loads cleanly and has valid structure."""
    config_path = os.path.join(PROJECT_ROOT, "configs", "base_config.yaml")
    assert os.path.isfile(config_path), "configs/base_config.yaml does not exist."

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert isinstance(config, dict), "Configuration root must be a dictionary."
    
    # Check top-level sections
    for section in ["project", "environment", "directories", "datasets", "validation", "models", "evaluation"]:
        assert section in config, f"Missing section in base_config.yaml: {section}"

    # Verify split ratios sum to 1.0
    val_cfg = config["validation"]
    split_sum = val_cfg["train_ratio"] + val_cfg["val_ratio"] + val_cfg["test_ratio"]
    assert pytest.approx(split_sum, 1e-5) == 1.0, f"Split ratios must sum to 1.0, got {split_sum}"

    # Verify datasets section contains Dataset A and Dataset B definitions
    assert "dataset_a" in config["datasets"], "Missing dataset_a in config"
    assert "dataset_b" in config["datasets"], "Missing dataset_b in config"


def test_requirements_and_gitignore_exist():
    """Verify requirements.txt and .gitignore are present and configured."""
    req_path = os.path.join(PROJECT_ROOT, "requirements.txt")
    git_path = os.path.join(PROJECT_ROOT, ".gitignore")

    assert os.path.isfile(req_path), "requirements.txt missing"
    assert os.path.isfile(git_path), ".gitignore missing"

    with open(req_path, "r", encoding="utf-8") as f:
        req_content = f.read()
    for pkg in ["numpy", "pandas", "scikit-learn", "pytest", "streamlit"]:
        assert pkg in req_content, f"Expected {pkg} in requirements.txt"

    with open(git_path, "r", encoding="utf-8") as f:
        git_content = f.read()
    assert "data/raw/*" in git_content, "Missing raw data exclusion in .gitignore"
    assert "artifacts/models/*" in git_content, "Missing model artifact exclusion in .gitignore"
