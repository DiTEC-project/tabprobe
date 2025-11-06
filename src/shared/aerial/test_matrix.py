"""
Aerial test matrix generation utilities.

This module contains functions for generating test matrices used in the Aerial
rule extraction approach.
"""

import numpy as np
from itertools import combinations, product


def generate_aerial_test_matrix(n_features, classes_per_feature, max_antecedents=2):
    """
    Generate test matrix (query) - creates ALL antecedent combinations at once.
    Unlike PyAerial which generates incrementally, we create all combinations upfront.

    Args:
        n_features: Number of features (columns) in the data
        classes_per_feature: List of number of classes for each feature
        max_antecedents: Maximum number of antecedents to combine

    Returns:
        test_matrix: numpy array of shape (n_test_vectors, total_dimensions)
        test_descriptions: List of tuples (feature_idx, class_idx) describing antecedents
        feature_value_indices: List of dicts with 'start', 'end', 'feature' for each feature
    """
    # Calculate total input dimension
    total_dim = sum(classes_per_feature)

    # Create feature_value_indices
    feature_value_indices = []
    start_idx = 0
    for feat_idx, n_classes in enumerate(classes_per_feature):
        feature_value_indices.append({
            'start': start_idx,
            'end': start_idx + n_classes,
            'feature': feat_idx
        })
        start_idx += n_classes

    # Initialize unmarked features with equal probabilities
    unmarked_features = _initialize_input_vectors(total_dim, feature_value_indices)

    test_vectors = []
    test_descriptions = []

    # Generate ALL combinations at once for all antecedent lengths
    # This is more efficient than PyAerial's incremental approach
    for r in range(1, max_antecedents + 1):
        # Get all feature combinations of size r
        for feature_indices in combinations(range(n_features), r):
            # For each feature combination, get all class combinations
            class_ranges = [list(range(classes_per_feature[f_idx])) for f_idx in feature_indices]

            # Generate all class combinations using product
            for class_combo in product(*class_ranges):
                # Create test vector
                test_vec = unmarked_features.copy()

                # Mark each selected feature-class pair
                description = []
                for feat_idx, class_idx in zip(feature_indices, class_combo):
                    feat_info = feature_value_indices[feat_idx]
                    # Set all classes of this feature to 0
                    test_vec[feat_info['start']:feat_info['end']] = 0.0
                    # Set the selected class to 1
                    test_vec[feat_info['start'] + class_idx] = 1.0
                    description.append((feat_idx, class_idx))

                test_vectors.append(test_vec)
                test_descriptions.append(tuple(description))

    return np.array(test_vectors), test_descriptions, feature_value_indices


def _initialize_input_vectors(input_vector_size, categories):
    """
    Initialize the input vectors with equal probabilities for each feature range.
    This follows PyAerial's initialization logic.
    """
    vector_with_unmarked_features = np.zeros(input_vector_size)
    for category in categories:
        vector_with_unmarked_features[category['start']:category['end']] = 1 / (
                category['end'] - category['start'])
    return vector_with_unmarked_features