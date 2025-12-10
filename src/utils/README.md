# Shared Utilities

This directory contains modularized, reusable code that can be shared across different experiments.

## Structure

```
shared/
├── __init__.py
└── aerial/
    ├── __init__.py
    ├── test_matrix.py         # Test matrix generation
    ├── rule_extraction.py     # Rule extraction from reconstructions
    └── data_prep.py           # Data preparation utilities
```

## Usage

```python
import torch
from shared.aerial import (
    generate_aerial_test_matrix,
    extract_rules_from_reconstruction,
    prepare_categorical_data
)

# Prepare data
encoded_data, classes_per_feature, feature_names, encoder = prepare_categorical_data(df)

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
```

## Important Notes

- The `initial_experiments/` directory remains **completely untouched** and independent
- This shared code can evolve and be modified without affecting past experiments
- Code duplication between `shared/` and `initial_experiments/` is intentional for isolation
