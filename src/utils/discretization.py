"""
Discretization utilities for continuous features.

This module provides functions for discretizing continuous features into categorical bins.
"""

import pandas as pd
import numpy as np


def is_likely_categorical(series, max_unique_ratio=0.01, max_unique_count=10):
    """
    Determine if a numerical column is likely categorical.

    A column is considered categorical if:
    - It has few unique values relative to total rows (ratio)
    - OR it has few unique values in absolute terms
    - AND all values are integers (no decimal parts)

    Args:
        series: pandas Series to check
        max_unique_ratio: Max ratio of unique values to total values (default: 0.01 = 1%)
        max_unique_count: Max absolute number of unique values (default: 10)

    Returns:
        True if the column is likely categorical
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return True

    n_unique = non_null.nunique()
    n_total = len(non_null)

    unique_ratio = n_unique / n_total if n_total > 0 else 0
    few_unique = (unique_ratio <= max_unique_ratio) or (n_unique <= max_unique_count)

    all_integers = np.allclose(non_null, non_null.astype(int))

    return few_unique and all_integers


def discretize_numerical_features(
        df,
        columns_to_drop=None,
        n_bins=10,
        categorical_columns=None,
        max_unique_ratio=0.05,
        max_unique_count=15
):
    """
    Discretize numerical features in a DataFrame into categorical bins.

    Columns are preserved as-is (not discretized) if:
    - They are binary (0/1 only)
    - They have few unique values and are integers (likely categorical)
    - They are explicitly listed in categorical_columns

    Truly continuous columns are discretized using quantile binning.

    Args:
        df: pandas DataFrame with features
        columns_to_drop: List of column names to exclude from processing
        n_bins: Number of bins for discretization (default: 10)
        categorical_columns: List of column names to treat as categorical (skip discretization)
        max_unique_ratio: Max ratio of unique values for auto-detecting categorical (default: 0.05)
        max_unique_count: Max unique values for auto-detecting categorical (default: 15)

    Returns:
        DataFrame with discretized numerical features
    """
    if columns_to_drop is None:
        columns_to_drop = []
    if categorical_columns is None:
        categorical_columns = []

    X = df.drop(columns=columns_to_drop, errors='ignore').copy()
    numerical_columns = X.select_dtypes(include=[np.number]).columns

    skip_columns = set(categorical_columns)

    for col in numerical_columns:
        unique_vals = set(X[col].dropna().unique())
        if unique_vals.issubset({0, 1}):
            skip_columns.add(col)
        elif is_likely_categorical(X[col], max_unique_ratio, max_unique_count):
            skip_columns.add(col)
            X[col] = X[col].fillna(-9999).astype(int).astype(str).replace('-9999', np.nan)

    columns_to_discretize = [col for col in numerical_columns if col not in skip_columns]

    discretized_columns = {}
    for col in columns_to_discretize:
        try:
            discretized = pd.qcut(X[col], q=n_bins, duplicates='drop')
            discretized_columns[col + "_discretized"] = (
                discretized.astype(str)
                .str.replace(" ", "", regex=False)
                .str.replace(",", "-", regex=False)
            )
        except ValueError:
            X[col] = X[col].astype(str)
            continue

    if discretized_columns:
        X = pd.concat([X, pd.DataFrame(discretized_columns)], axis=1)
        X = X.drop(columns=columns_to_discretize)

    return X
