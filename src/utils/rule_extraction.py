"""
Rule extraction utilities.

This module contains functions for extracting association rules and frequent itemsets
from reconstruction probability matrices.
"""

import numpy as np
import pandas as pd


def extract_rules_from_reconstruction(prob_matrix, test_descriptions, feature_value_indices,
                                      ant_similarity=0.5, cons_similarity=0.8, feature_names=None,
                                      encoder=None):
    """
    Extract association rules from reconstruction probability matrix

    Args:
        prob_matrix: Reconstruction probability matrix (n_test_vectors, total_dim)
        test_descriptions: List of tuples describing antecedents for each test vector
            e.g., ((0, 1), (2, 3)) --> 1st class of 0th feature and 3rd class of 2nd feature are marked
            (see test_matrix.py for the exact format)
        feature_value_indices: List of dicts with 'start', 'end', 'feature' for each feature
        ant_similarity: Threshold for antecedent validation (default 0.5)
        cons_similarity: Threshold for consequent extraction (default 0.8)
        feature_names: Optional list of feature names for readable output
        encoder: Optional OneHotEncoder to map class indices to actual category values

    Returns:
        association_rules: List of dicts with 'antecedents' and 'consequent'
    """
    association_rules = []

    for i, antecedent_desc in enumerate(test_descriptions):
        # Get reconstruction probabilities for this test vector
        implication_probabilities = prob_matrix[i]

        # Convert antecedent description to indices
        candidate_antecedents = []
        for feat_idx, class_idx in antecedent_desc:
            feat_info = feature_value_indices[feat_idx]
            ant_idx = feat_info['start'] + class_idx
            candidate_antecedents.append(ant_idx)

        # Check if antecedents have high support
        low_support = False
        for ant_idx in candidate_antecedents:
            if implication_probabilities[ant_idx] <= ant_similarity:
                low_support = True
                break

        if low_support:
            continue

        # Find high-support consequents (not in antecedents)
        # Extract antecedent feature indices (not just class indices)
        antecedent_feature_indices = set(feat_idx for feat_idx, _ in antecedent_desc)

        consequent_list = []
        for feat_info in feature_value_indices:
            # Skip if this feature is already in the antecedents
            # This prevents rules like f1=a → f1=b (same feature, different classes)
            if feat_info['feature'] in antecedent_feature_indices:
                continue

            feat_probs = implication_probabilities[feat_info['start']:feat_info['end']]

            # Find class with the highest probability for this feature
            max_class_idx = np.argmax(feat_probs)
            max_prob = feat_probs[max_class_idx]

            # Check if this is a valid consequent
            if max_prob >= cons_similarity:
                consequent_list.append((feat_info['feature'], max_class_idx, max_prob))

        # Create rules for each consequent
        if consequent_list:
            # Build antecedents
            antecedent_list = []
            for feat_idx, class_idx in antecedent_desc:
                feature_name = feature_names[feat_idx] if feature_names else f"F{feat_idx}"
                # Map class index to actual category value if encoder is provided
                if encoder is not None:
                    value = encoder.categories_[feat_idx][class_idx]
                else:
                    value = class_idx
                antecedent_list.append({
                    'feature': feature_name,
                    'value': value
                })

            for cons_feat, cons_class, cons_prob in consequent_list:
                feature_name = feature_names[cons_feat] if feature_names else f"F{cons_feat}"
                # Map class index to actual category value if encoder is provided
                if encoder is not None:
                    cons_value = encoder.categories_[cons_feat][cons_class]
                else:
                    cons_value = cons_class
                consequent = {
                    'feature': feature_name,
                    'value': cons_value
                }

                association_rules.append({
                    'antecedents': antecedent_list,
                    'consequent': consequent,
                })

    return association_rules


def calculate_itemset_support(itemset, data, feature_names):
    """
    Calculate support for an itemset by checking how many rows in the data contain all items.

    Args:
        itemset: List of dicts with 'feature' and 'value' keys
        data: Original dataset (numpy array or DataFrame)
        feature_names: List of feature names

    Returns:
        support: Fraction of rows that contain all items in the itemset (rounded to 3 decimals)
    """
    # Convert data to DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        if feature_names is not None:
            data = pd.DataFrame(data, columns=feature_names)
        else:
            raise ValueError("feature_names must be provided when data is not a DataFrame")

    num_rows = len(data)
    if num_rows == 0:
        return 0.0

    # Start with all rows matching
    matching_rows = np.ones(num_rows, dtype=bool)

    # For each item in the itemset, filter rows
    for item in itemset:
        feature = item['feature']
        value = item['value']

        # Get the column values
        column_values = data[feature].values

        # Compare with the target value
        # Convert numpy scalar types to Python native types for consistent comparison
        if hasattr(value, 'item'):
            # numpy scalar - convert to Python native type
            value = value.item()

        # Direct comparison (works for strings, numbers, etc.)
        matches = column_values == value

        matching_rows &= matches

    # Support is the fraction of matching rows
    support = float(np.sum(matching_rows)) / num_rows
    return round(support, 3)


def extract_frequent_itemsets_from_reconstruction(prob_matrix, test_descriptions, feature_value_indices,
                                                   data, similarity=0.5, feature_names=None, encoder=None):
    """
    Extract frequent itemsets from reconstruction probability matrix.

    Support values are calculated automatically and included in the output.

    Args:
        prob_matrix: Reconstruction probability matrix (n_test_vectors, total_dim)
        test_descriptions: List of tuples describing antecedents for each test vector
            e.g., ((0, 1), (2, 3)) --> 1st class of 0th feature and 3rd class of 2nd feature are marked
            (see test_matrix.py for the exact format)
        feature_value_indices: List of dicts with 'start', 'end', 'feature' for each feature
        data: Original dataset (numpy array or DataFrame) for support calculation
        similarity: Similarity threshold for antecedent validation (default 0.5)
        feature_names: Optional list of feature names for readable output
        encoder: Optional OneHotEncoder to map class indices to actual category values

    Returns:
        dict with 'itemsets' (list of itemsets with support) and 'statistics' (aggregate stats)
        Example: {
            'itemsets': [
                {'itemset': [{'feature': 'age', 'value': '30-39'}], 'support': 0.524},
                {'itemset': [{'feature': 'age', 'value': '30-39'}, {'feature': 'tumor-size', 'value': '20-24'}], 'support': 0.312}
            ],
            'statistics': {'itemset_count': 2, 'average_support': 0.418}
        }
    """
    frequent_itemsets = []

    for i, antecedent_desc in enumerate(test_descriptions):
        # Get reconstruction probabilities for this test vector
        implication_probabilities = prob_matrix[i]

        # Convert antecedent description to indices
        candidate_antecedents = []
        for feat_idx, class_idx in antecedent_desc:
            feat_info = feature_value_indices[feat_idx]
            ant_idx = feat_info['start'] + class_idx
            candidate_antecedents.append(ant_idx)

        # Check if all antecedents have high reconstruction probability
        low_support = False
        for ant_idx in candidate_antecedents:
            if implication_probabilities[ant_idx] <= similarity:
                low_support = True
                break

        if low_support:
            continue

        # This is a frequent itemset
        itemset = []
        for feat_idx, class_idx in antecedent_desc:
            feature_name = feature_names[feat_idx] if feature_names else f"F{feat_idx}"
            # Map class index to actual category value if encoder is provided
            if encoder is not None:
                value = encoder.categories_[feat_idx][class_idx]
            else:
                value = class_idx
            itemset.append({
                'feature': feature_name,
                'value': value
            })

        # Calculate support for this itemset against the actual data
        support = calculate_itemset_support(itemset, data, feature_names)

        frequent_itemsets.append({
            'itemset': itemset,
            'support': support
        })

    # Calculate statistics
    if len(frequent_itemsets) == 0:
        return {'itemsets': [], 'statistics': {'itemset_count': 0, 'average_support': 0.0}}

    avg_support = float(round(np.mean([item['support'] for item in frequent_itemsets]), 3))
    stats = {
        'itemset_count': len(frequent_itemsets),
        'average_support': avg_support
    }

    return {'itemsets': frequent_itemsets, 'statistics': stats}