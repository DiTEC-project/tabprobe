"""
TabICL-based Frequent Itemset Mining

Adapts TabICL for frequent itemset discovery using TabProbe
"""
import time
import os
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from tabicl import TabICLClassifier

from src.utils.data_prep import prepare_categorical_data, add_gaussian_noise
from src.utils.test_matrix import generate_test_matrix
from src.utils.rule_extraction import extract_frequent_itemsets_from_reconstruction


def adapt_tabicl_for_reconstruction(context_table, query_matrix, feature_value_indices, n_samples=None,
                                    noise_factor=0.5, n_estimators=8):
    """Adapt TabICL for itemset mining using reconstruction logic."""
    if n_samples and len(context_table) > n_samples:
        context_table = context_table[:n_samples]

    print(f"    Adding Gaussian noise (factor={noise_factor}) to context...")
    noisy_context = add_gaussian_noise(context_table, noise_factor=noise_factor)

    n_queries = query_matrix.shape[0]
    n_features_total = query_matrix.shape[1]
    reconstruction_probs = np.zeros((n_queries, n_features_total))

    print(f"    Reconstructing all features for {n_queries} queries...")

    for feat_idx, feat_info in enumerate(feature_value_indices):
        start_idx = feat_info['start']
        end_idx = feat_info['end']
        n_classes = end_idx - start_idx

        print(f"        Feature {feat_idx}: classes={n_classes}")

        x_context = np.delete(noisy_context, range(start_idx, end_idx), axis=1)
        y_context = np.argmax(context_table[:, start_idx:end_idx], axis=1)

        tabicl_model = TabICLClassifier(n_estimators=n_estimators)
        tabicl_model.fit(x_context, y_context)

        x_query = np.delete(query_matrix, range(start_idx, end_idx), axis=1)
        probs = tabicl_model.predict_proba(x_query)

        if probs.shape[1] != n_classes:
            proper_probs = np.zeros((n_queries, n_classes))
            if hasattr(tabicl_model, 'classes_'):
                for i, cls in enumerate(tabicl_model.classes_):
                    if cls < n_classes:
                        proper_probs[:, cls] = probs[:, i]
            reconstruction_probs[:, start_idx:end_idx] = proper_probs
        else:
            reconstruction_probs[:, start_idx:end_idx] = probs

    return reconstruction_probs


def tabicl_itemset_learning(dataset, max_itemset_length=2, context_samples=100, similarity=0.5):
    """Frequent itemset mining using TabICL."""
    encoded_data, classes_per_feature, feature_names, encoder = prepare_categorical_data(dataset)

    test_matrix, test_descriptions, feature_value_indices = generate_test_matrix(
        n_features=len(classes_per_feature),
        classes_per_feature=classes_per_feature,
        max_antecedents=max_itemset_length,
        use_zeros_for_unmarked=False
    )

    reconstruction_probs = adapt_tabicl_for_reconstruction(
        context_table=encoded_data,
        query_matrix=test_matrix,
        feature_value_indices=feature_value_indices,
        n_samples=context_samples
    )

    result = extract_frequent_itemsets_from_reconstruction(
        prob_matrix=reconstruction_probs,
        test_descriptions=test_descriptions,
        feature_value_indices=feature_value_indices,
        data=dataset,
        similarity=similarity,
        feature_names=feature_names,
        encoder=encoder
    )

    return result['itemsets'], result['statistics'], feature_names, dataset.values
