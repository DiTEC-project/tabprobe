"""
Dataset loading utilities for the rulepfn project.

Provides functions to load datasets from:
- data/ucimlrepo: UCI ML Repository datasets
- data/gene_expression: Gene expression datasets
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd

from .discretization import discretize_numerical_features

# Project root directory (assumes utils is at src/utils/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Available UCI datasets
UCIMLREPO_NORMAL_DATASETS = [
    'breast_cancer',
    'congressional_voting',
    'mushroom',
    'chess_king_rook_vs_king_pawn',
    'spambase',
]

UCIMLREPO_SMALL_DATASETS = [
    # 'lung_cancer',
    'hepatitis',
    # 'breast_cancer_coimbra',
    'cervical_cancer_behavior_risk',
    'autism_screening_adolescent',
    'acute_inflammations',
    'fertility',
]

# Datasets that need discretization (have numerical features)
DATASETS_TO_DISCRETIZE = {
    'spambase',
    'hepatitis',
    # 'breast_cancer_coimbra',
    'cervical_cancer_behavior_risk',
    'autism_screening_adolescent',
    'acute_inflammations',
    'fertility',
}


def get_ucimlrepo_datasets(
        names: Optional[List[str]] = None,
        size: str = 'normal',
        discretize: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load UCI ML Repository datasets from local CSV files.

    Args:
        names: List of dataset names to load. If None, loads all available datasets for the given size.
        size: 'normal' for normal_size_tables, 'small' for small_size_tables.
        discretize: If True, automatically discretize numerical datasets (default: True).

    Returns:
        List of dictionaries with 'name' and 'data' (DataFrame) keys.
    """
    if size == 'normal':
        data_dir = DATA_DIR / "normal_size_tables"
        default_datasets = UCIMLREPO_NORMAL_DATASETS
    elif size == 'small':
        data_dir = DATA_DIR / "small_size_tables"
        default_datasets = UCIMLREPO_SMALL_DATASETS
    else:
        raise ValueError(f"Invalid size '{size}'. Must be 'normal' or 'small'.")

    if names is None:
        names = default_datasets.copy()

    datasets = []
    for name in names:
        filepath = data_dir / f"{name}.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")

        df = pd.read_csv(filepath)

        if discretize and name in DATASETS_TO_DISCRETIZE:
            df = discretize_numerical_features(df)

        datasets.append({
            'name': name,
            'data': df
        })

    return datasets
