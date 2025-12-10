"""
Aerial rule extraction utilities.

This module contains functions for extracting association rules from reconstruction probability matrices.
"""

import numpy as np


def extract_rules_from_reconstruction(prob_matrix, test_descriptions, feature_value_indices,
                                      ant_similarity=0.5, cons_similarity=0.8, feature_names=None,
                                      encoder=None):
    """
    Extract association rules from reconstruction probability matrix following PyAerial logic.

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
        association_rules: List of dicts in PyAerial format with 'antecedents' and 'consequent'
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

        # Check if antecedents have high support (PyAerial's validation)
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

            # Find class with highest probability for this feature
            max_class_idx = np.argmax(feat_probs)
            max_prob = feat_probs[max_class_idx]

            # Check if this is a valid consequent
            if max_prob >= cons_similarity:
                consequent_list.append((feat_info['feature'], max_class_idx, max_prob))

        # Create rules for each consequent in PyAerial format
        if consequent_list:
            # Build antecedents in PyAerial format
            antecedents_pyaerial = []
            for feat_idx, class_idx in antecedent_desc:
                feature_name = feature_names[feat_idx] if feature_names else f"F{feat_idx}"
                # Map class index to actual category value if encoder is provided
                if encoder is not None:
                    value = encoder.categories_[feat_idx][class_idx]
                else:
                    value = class_idx
                antecedents_pyaerial.append({
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
                consequent_pyaerial = {
                    'feature': feature_name,
                    'value': cons_value
                }

                association_rules.append({
                    'antecedents': antecedents_pyaerial,
                    'consequent': consequent_pyaerial,
                })

    return association_rules
