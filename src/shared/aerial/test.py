import torch
from ucimlrepo import fetch_ucirepo

from src.shared.aerial import (
    generate_aerial_test_matrix,
    extract_rules_from_reconstruction,
    prepare_categorical_data
)

breast_cancer = fetch_ucirepo(id=14).data.features
encoded_data, classes_per_feature, feature_names, encoder = prepare_categorical_data(breast_cancer)

# Generate test matrix
test_matrix, test_descriptions, feature_value_indices = generate_aerial_test_matrix(
    n_features=len(classes_per_feature),
    classes_per_feature=classes_per_feature,
    max_antecedents=2
)

# Your foundation model code here
# foundation_model_output = your_foundation_model(test_matrix)

# Extract rules
rules = extract_rules_from_reconstruction(
    prob_matrix=foundation_model_output,
    test_descriptions=test_descriptions,
    feature_value_indices=feature_value_indices,
    feature_names=feature_names
)
