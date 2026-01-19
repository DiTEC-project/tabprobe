"""
TabPFN-based Frequent Itemset Mining

Adapts TabPFN for frequent itemset discovery using TabProbe
"""
import time
import os
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from tabpfn import TabPFNClassifier
from tabpfn_extensions.many_class import ManyClassClassifier

from src.utils.data_prep import prepare_categorical_data, add_gaussian_noise
from src.utils.test_matrix import generate_test_matrix
from src.utils.rule_extraction import extract_frequent_itemsets_from_reconstruction


def adapt_tabpfn_for_reconstruction(tabpfn_model, context_table, query_matrix,
                                    feature_value_indices, n_samples=None, noise_factor=0.5):
    """
    Adapt TabPFN for frequent itemset mining using reconstruction logic.

    For each feature, trains TabPFN to predict that feature from all others,
    then reconstructs ALL features for each query to get itemset patterns.
    """
    # Limit context size if needed
    if n_samples and len(context_table) > n_samples:
        context_table = context_table[:n_samples]

    # Add Gaussian noise to context
    print(f"    Adding Gaussian noise (factor={noise_factor}) to context...")
    noisy_context = add_gaussian_noise(context_table, noise_factor=noise_factor)

    n_queries = query_matrix.shape[0]
    n_features_total = query_matrix.shape[1]

    # Initialize reconstruction matrix
    reconstruction_probs = np.zeros((n_queries, n_features_total))

    print(f"    Reconstructing all features for {n_queries} queries...")
    print(f"    Training {len(feature_value_indices)} feature predictors...")

    for feat_idx, feat_info in enumerate(feature_value_indices):
        start_idx = feat_info['start']
        end_idx = feat_info['end']
        n_classes = end_idx - start_idx

        print(f"        Feature {feat_idx}: classes={n_classes}, range=[{start_idx}:{end_idx}]")

        # Prepare context: X = all OTHER features, y = current feature
        x_context = np.delete(noisy_context, range(start_idx, end_idx), axis=1)
        y_context_onehot = context_table[:, start_idx:end_idx]
        y_context = np.argmax(y_context_onehot, axis=1)

        # Use ManyClassClassifier for features with >10 classes
        if n_classes > 10:
            print(f"        Using ManyClassClassifier wrapper (classes={n_classes} > 10)")
            model_to_use = ManyClassClassifier(
                estimator=tabpfn_model,
                alphabet_size=10,
                random_state=tabpfn_model.random_state if hasattr(tabpfn_model, 'random_state') else None
            )
        else:
            model_to_use = tabpfn_model

        # Fit model
        model_to_use.fit(x_context, y_context)

        # Prepare query matrix
        x_query = np.delete(query_matrix, range(start_idx, end_idx), axis=1)

        if hasattr(model_to_use, 'predict_proba'):
            probs = model_to_use.predict_proba(x_query)

            if probs.shape[1] != n_classes:
                print(f"        WARNING: predict_proba returned {probs.shape[1]} classes, expected {n_classes}")
                proper_probs = np.zeros((n_queries, n_classes))
                if hasattr(model_to_use, 'classes_'):
                    for i, cls in enumerate(model_to_use.classes_):
                        if cls < n_classes:
                            proper_probs[:, cls] = probs[:, i]
                else:
                    min_cols = min(probs.shape[1], n_classes)
                    proper_probs[:, :min_cols] = probs[:, :min_cols]
                reconstruction_probs[:, start_idx:end_idx] = proper_probs
            else:
                reconstruction_probs[:, start_idx:end_idx] = probs

    return reconstruction_probs


def tabpfn_itemset_learning(dataset, max_itemset_length=3, context_samples=100,
                            similarity=0.5, random_state=42):
    """
    End-to-end frequent itemset mining using TabPFN.

    Args:
        dataset: DataFrame with categorical features
        max_itemset_length: Maximum itemset length (default: 3)
        context_samples: Number of samples to use as context
        similarity: Similarity threshold for itemset validation (default: 0.5)
        random_state: Random seed for TabPFN model

    Returns:
        itemsets: List of extracted frequent itemsets with support
        stats: Statistics dictionary
    """
    # Prepare data
    encoded_data, classes_per_feature, feature_names, encoder = prepare_categorical_data(dataset)

    print(f"Dataset shape: {encoded_data.shape}")
    print(f"Number of features: {len(classes_per_feature)}")
    print(f"Classes per feature: {classes_per_feature}")

    # Generate test matrix (itemset patterns)
    test_matrix, test_descriptions, feature_value_indices = generate_test_matrix(
        n_features=len(classes_per_feature),
        classes_per_feature=classes_per_feature,
        max_antecedents=max_itemset_length,
        use_zeros_for_unmarked=False
    )

    print(f"\nTest matrix shape: {test_matrix.shape}")
    print(f"Number of test vectors: {len(test_descriptions)}")

    # Initialize TabPFN
    tabpfn_model = TabPFNClassifier(
        n_estimators=8,
        random_state=random_state,
        average_before_softmax=True,
        inference_precision='auto'
    )

    # Adapt TabPFN for reconstruction
    print(f"\nUsing TabPFN for pattern reconstruction...")
    reconstruction_probs = adapt_tabpfn_for_reconstruction(
        tabpfn_model=tabpfn_model,
        context_table=encoded_data,
        query_matrix=test_matrix,
        feature_value_indices=feature_value_indices,
        n_samples=context_samples
    )

    print(f"Reconstruction shape: {reconstruction_probs.shape}")

    # Extract frequent itemsets
    print(f"\nExtracting frequent itemsets...")
    result = extract_frequent_itemsets_from_reconstruction(
        prob_matrix=reconstruction_probs,
        test_descriptions=test_descriptions,
        feature_value_indices=feature_value_indices,
        data=dataset,  # Original data for support calculation
        similarity=similarity,
        feature_names=feature_names,
        encoder=encoder
    )

    itemsets = result['itemsets']
    stats = result['statistics']

    print(f"{len(itemsets)} frequent itemsets found!")

    return itemsets, stats, feature_names, dataset.values
