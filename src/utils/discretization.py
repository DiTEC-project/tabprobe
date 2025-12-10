"""
Discretization utilities for continuous features.

This module provides functions for discretizing continuous features into categorical bins.
"""

import pandas as pd
import numpy as np


def discretize_numerical_features(df, columns_to_drop=None, n_bins=10):
    """
    Discretize numerical features in a DataFrame into categorical bins.

    Binary columns (0/1 only) are preserved as-is.
    Non-binary numerical columns are discretized using quantile binning.

    Args:
        df: pandas DataFrame with features
        columns_to_drop: List of column names to exclude from processing
        n_bins: Number of bins for discretization (default: 10)

    Returns:
        DataFrame with discretized numerical features
    """
    if columns_to_drop is None:
        columns_to_drop = []

    X = df.drop(columns=columns_to_drop, errors='ignore').copy()
    numerical_columns = X.select_dtypes(include=[np.number]).columns

    binary_columns = [
        col for col in numerical_columns
        if set(X[col].dropna().unique()).issubset({0, 1})
    ]

    non_binary_columns = [col for col in numerical_columns if col not in binary_columns]
    discretized_columns = {
        col + "_discretized": pd.qcut(X[col], q=n_bins, duplicates='drop')
        .astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace(",", "-", regex=False)
        for col in non_binary_columns
    }

    X = pd.concat([X, pd.DataFrame(discretized_columns)], axis=1)
    X = X.drop(columns=non_binary_columns)

    return X
