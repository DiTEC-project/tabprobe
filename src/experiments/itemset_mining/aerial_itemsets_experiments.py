"""
PyAerial Frequent Itemset Mining Experiments

This module runs frequent itemset mining experiments using PyAerial.
"""
import time
import os
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from aerial import model, rule_extraction


def aerial_itemset_learning(dataset, max_length=3, similarity=0.5, batch_size=None,
                            layer_dims=None, epochs=2, random_state=42):
    """
    End-to-end unsupervised frequent itemset learning using PyAerial.

    Args:
        dataset: DataFrame with categorical features
        max_length: Maximum itemset length (default: 3)
        similarity: Similarity threshold for itemset validation (default: 0.5)
        layer_dims: Hidden layer dimensions for autoencoder (default: auto)
        batch_size: Batch size for training (default: auto)
        epochs: Number of training epochs
        random_state: Random seed for reproducibility

    Returns:
        itemsets: List of extracted frequent itemsets
        stats: Statistics from pyaerial (itemset_count, average_support)
    """
    # Set seed for PyTorch (aerial uses PyTorch internally)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_state)

    feature_names = list(dataset.columns)

    print(f"Dataset shape: {dataset.shape}")
    print(f"Number of features: {len(feature_names)}")

    # Train the autoencoder
    print(f"\nTraining Aerial autoencoder...")
    print(f"  epochs={epochs}, layer_dims={layer_dims}, batch_size={batch_size}")
    trained_autoencoder = model.train(
        dataset,
        layer_dims=layer_dims,
        batch_size=batch_size,
        epochs=epochs
    )

    # Extract frequent itemsets using Aerial's native function
    print(f"\nExtracting frequent itemsets...")
    print(f"  max_length={max_length}")
    print(f"  similarity={similarity}")

    result = rule_extraction.generate_frequent_itemsets(
        trained_autoencoder,
        similarity=similarity,
        max_length=max_length
    )

    itemsets = result['itemsets']
    stats = result['statistics']

    print(f"\n{len(itemsets)} frequent itemsets found!")
    print(f"  PyAerial stats: avg_support={stats.get('average_support', 0):.4f}")

    return itemsets, stats


def get_dataset_parameters(dataset_name, dataset_size):
    """
    Get dataset-specific training parameters.

    Args:
        dataset_name: Name of the dataset
        dataset_size: Size category ('normal' or 'small')

    Returns:
        dict: Parameters (batch_size, layer_dims, epochs)
    """
    # Dataset-specific parameter overrides
    DATASET_PARAMS = {
        'breast_cancer': {
            'batch_size': 2,
            'layer_dims': [4],
            'epochs': 2
        },
        'congressional_voting': {
            'batch_size': 4,
            'layer_dims': [2],
            'epochs': 2
        }
    }

    # Default parameters by dataset size
    DEFAULT_PARAMS = {
        'normal': {
            'batch_size': 64,
            'layer_dims': [4],
            'epochs': 2
        },
        'small': {
            'batch_size': 2,
            'layer_dims': [4],
            'epochs': 10
        }
    }

    # Check for dataset-specific override first
    if dataset_name in DATASET_PARAMS:
        return DATASET_PARAMS[dataset_name]

    # Otherwise use default for the dataset size
    return DEFAULT_PARAMS.get(dataset_size, DEFAULT_PARAMS['normal'])
