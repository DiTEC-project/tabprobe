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
UCIMLREPO_DIR = DATA_DIR / "ucimlrepo"
GENE_EXPRESSION_DIR = DATA_DIR / "gene_expression"

# Available UCI datasets
UCIMLREPO_DATASETS = [
    'congressional_voting',
    'breast_cancer',
    'mushroom',
    'chess_king_rook_vs_king_pawn',
    'spambase',
]

# Datasets that need discretization (have numerical features)
DATASETS_TO_DISCRETIZE = {'spambase'}


def get_ucimlrepo_datasets(
    names: Optional[List[str]] = None,
    discretize: bool = True
) -> List[Dict[str, Any]]:
    """
    Load UCI ML Repository datasets from local CSV files.

    Args:
        names: List of dataset names to load. If None, loads all available datasets.
        discretize: If True, automatically discretize numerical datasets (default: True).

    Returns:
        List of dictionaries with 'name' and 'data' (DataFrame) keys.
    """
    if names is None:
        names = UCIMLREPO_DATASETS.copy()

    datasets = []
    for name in names:
        filepath = UCIMLREPO_DIR / f"{name}.csv"
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


def get_gene_expression_datasets(
    names: Optional[List[str]] = None,
    max_columns: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load gene expression datasets from local CSV files.

    Args:
        names: List of dataset names (without extension) to load.
               If None, loads all available datasets.
        max_columns: If specified, return only the first D columns of each table.

    Returns:
        List of dictionaries with 'name' and 'data' (DataFrame) keys.
    """
    if names is None:
        csv_files = list(GENE_EXPRESSION_DIR.glob("*.csv"))
        names = [f.stem for f in csv_files]

    datasets = []
    for name in names:
        filepath = GENE_EXPRESSION_DIR / f"{name}.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")

        df = pd.read_csv(filepath)

        if max_columns is not None:
            df = df.iloc[:, :max_columns]

        datasets.append({
            'name': name,
            'data': df
        })

    return datasets


def list_available_datasets(source: str = 'all') -> Dict[str, List[str]]:
    """
    List all available datasets.

    Args:
        source: 'ucimlrepo', 'gene_expression', or 'all'

    Returns:
        Dictionary with source as key and list of dataset names as value.
    """
    available = {}

    if source in ('ucimlrepo', 'all'):
        if UCIMLREPO_DIR.exists():
            available['ucimlrepo'] = [f.stem for f in UCIMLREPO_DIR.glob("*.csv")]
        else:
            available['ucimlrepo'] = []

    if source in ('gene_expression', 'all'):
        if GENE_EXPRESSION_DIR.exists():
            available['gene_expression'] = [f.stem for f in GENE_EXPRESSION_DIR.glob("*.csv")]
        else:
            available['gene_expression'] = []

    return available
