"""
Aerial rule extraction utilities.

This module contains functions for extracting association rules from reconstruction probability matrices.
"""

import numpy as np


def extract_rules_from_reconstruction(prob_matrix, test_descriptions, feature_value_indices,
                                      ant_similarity=0.5, cons_similarity=0.8, feature_names=None):
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

        # Check if antecedents have high support (PyAerial's validation)
        low_support = False
        for ant_idx in candidate_antecedents:
            if implication_probabilities[ant_idx] <= ant_similarity:
                low_support = True
                break

        if low_support:
            continue

        # Find high-support consequents (not in antecedents)
        consequent_list = []
        for feat_info in feature_value_indices:
            feat_probs = implication_probabilities[feat_info['start']:feat_info['end']]

            # Find class with highest probability for this feature
            max_class_idx = np.argmax(feat_probs)
            max_prob = feat_probs[max_class_idx]
            global_idx = feat_info['start'] + max_class_idx

            # Check if this is a valid consequent
            if max_prob >= cons_similarity and global_idx not in candidate_antecedents:
                consequent_list.append((feat_info['feature'], max_class_idx, max_prob))

        # Create rules for each consequent
        if consequent_list:
            antecedent_str = []
            for feat_idx, class_idx in antecedent_desc:
                if feature_names:
                    antecedent_str.append(f"{feature_names[feat_idx]}={class_idx}")
                else:
                    antecedent_str.append(f"F{feat_idx}={class_idx}")

            for cons_feat, cons_class, cons_prob in consequent_list:
                if feature_names:
                    consequent_str = f"{feature_names[cons_feat]}={cons_class}"
                else:
                    consequent_str = f"F{cons_feat}={cons_class}"

                association_rules.append({
                    'antecedents': antecedent_str,
                    'consequent': consequent_str,
                    'confidence': float(cons_prob)
                })

    return association_rules
