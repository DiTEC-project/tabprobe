"""
TabICL-based Rule Extraction Baseline

This is a BASELINE implementation/adaptation of the tabular foundation model TabICL to do association
rule mining from tabular data using a Aerial-like logic (association by reconstruction/prediction success)

Adaptation Strategy:
- For each feature: Train TabICL to predict that feature from all others
- Add Gaussian noise to context (matching PyAerial's denoising approach)
- Apply to test vectors (antecedent patterns) to get reconstructions
- Extract rules using Aerial's rule extraction logic

Limitations of this approach:
1. TabICL is supervised (needs discrete labels), while rule learning is unsupervised
2. We must predict each feature independently, losing holistic pattern reconstruction
3. Much slower than Aerial (must retrain/fit for each feature)
4. Quality expected to be inferior to specialized rule learning models
5. CRITICAL: Antecedent validation is fundamentally broken (see below)

FUNDAMENTAL INCOMPATIBILITY:
PyAerial: "Given marked features A, reconstruct ALL features"
          - Antecedents A are in the query
          - Can validate A reconstructs well

TabICL:   "Given all features EXCEPT i, predict feature i"
          - When i ∈ A (i is antecedent), we MUST remove it
          - Cannot validate antecedent reconstruction!

Result: Consequent prediction works (✓), antecedent validation fails (✗)
        Rules A -> C can be extracted, but lower quality (no antecedent filtering)

This baseline serves to justify the need for custom tabular foundation models
specifically designed for unsupervised rule discovery.
"""
import time
import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo
from tabicl import TabICLClassifier

from src.shared.aerial import prepare_categorical_data
from src.shared.aerial.data_prep import add_gaussian_noise
from src.shared.aerial.test_matrix import generate_aerial_test_matrix
from src.shared.aerial.rule_extraction import extract_rules_from_reconstruction
from src.shared.rule_quality import calculate_rule_metrics


def adapt_tabicl_for_reconstruction(tabicl_model, context_table, query_matrix,
                                    feature_value_indices, n_samples=None, noise_factor=0.5):
    """
    Adapt TabICL for unsupervised rule learning following Aerial's ALL-AT-ONCE reconstruction logic.

    Aerial's Approach:
    - For a query with marked features A, pass it through the autoencoder ONCE
    - Get reconstruction probabilities for ALL features (both A and F/A) simultaneously
    - Check if A reconstructs well (antecedent validation)
    - Check which features in F/A reconstruct well (consequent extraction)

    PyAerial marks features A and looks at reconstruction of F/A. If reconstruction is successful
    based on both antecedent and consequent similarity thresholds for features C, then A -> C.

    Problem: TabICL is supervised (needs y) and predicts ONE label at a time,
    while Aerial reconstructs ALL features at once.

    Solution:
    - Train one model per feature to reconstruct that feature from all other features
    - For each query, predict ALL features to simulate "all-at-once" reconstruction
    - This mimics Aerial's behavior: given marked features A, what are probabilities for all features?

    Args:
        tabicl_model: Pretrained TabICL model
        context_table: The dataset to use as context (n_rows, n_features)
        query_matrix: Test matrix with antecedent patterns (n_queries, n_features)
                     Each row has marked features (A) set to 1 in their class position,
                     and unmarked features (F/A) set to uniform probabilities
        feature_value_indices: Feature range information
        n_samples: Number of context samples to use (None = all)
        noise_factor: Gaussian noise standard deviation to add to context (default=0.5, matching PyAerial)

    Returns:
        reconstruction_probs: Probability matrix (n_queries, n_features)
                             For each query, contains predicted probabilities for ALL features,
                             allowing us to check both antecedent reconstruction and consequent prediction
    """

    # Limit context size if needed (TabICL has context length limits)
    if n_samples and len(context_table) > n_samples:
        context_table = context_table[:n_samples]

    # Add Gaussian noise to context table, matching PyAerial's training approach
    # This makes TabICL see values between 0 and 1 (not just one-hot 0s and 1s),
    # which makes query patterns with equal probabilities [0.5, 0.5] more natural
    print(f"    Adding Gaussian noise (factor={noise_factor}) to context...")
    noisy_context = add_gaussian_noise(context_table, noise_factor=noise_factor)

    n_queries = query_matrix.shape[0]
    n_features_total = query_matrix.shape[1]

    # Initialize reconstruction matrix - will contain ALL feature predictions for ALL queries
    reconstruction_probs = np.zeros((n_queries, n_features_total))

    # For each feature, train TabICL to predict it and reconstruct for ALL queries
    # This simulates Aerial's all-at-once reconstruction behavior
    print(f"    Reconstructing all features for {n_queries} queries...")
    print(f"    Training {len(feature_value_indices)} feature predictors...")

    for feat_idx, feat_info in enumerate(feature_value_indices):
        start_idx = feat_info['start']
        end_idx = feat_info['end']
        n_classes = end_idx - start_idx

        print(f"        Feature {feat_idx}: classes={n_classes}, range=[{start_idx}:{end_idx}]")

        # Prepare context: X = all OTHER features (excluding target), y = current feature
        # This properly teaches TabICL: given all other features, predict this feature
        # This avoids data leakage and matches autoencoder reconstruction behavior
        # Use noisy_context to match PyAerial's training with noise
        x_context = np.delete(noisy_context, range(start_idx, end_idx), axis=1)
        y_context_onehot = context_table[:, start_idx:end_idx]  # Target is clean (no noise)
        y_context = np.argmax(y_context_onehot, axis=1)

        # Fit TabICL to predict this feature from all OTHER features
        tabicl_model.fit(x_context, y_context)

        # Prepare query matrix: remove the target feature columns from query
        # We must match the training input shape (context without target feature)
        # However, we want to keep OTHER marked features (antecedents) in the query
        x_query = np.delete(query_matrix, range(start_idx, end_idx), axis=1)

        if hasattr(tabicl_model, 'predict_proba'):
            probs = tabicl_model.predict_proba(x_query)
            # probs shape: (n_queries, n_classes_seen_in_training)
            # May not match n_classes if some classes weren't in context

            if probs.shape[1] != n_classes:
                print(f"        WARNING: predict_proba returned {probs.shape[1]} classes, expected {n_classes}")
                # Create proper sized array and fill with available probabilities
                proper_probs = np.zeros((n_queries, n_classes))
                # Get the classes TabICL actually learned
                if hasattr(tabicl_model, 'classes_'):
                    for i, cls in enumerate(tabicl_model.classes_):
                        if cls < n_classes:
                            proper_probs[:, cls] = probs[:, i]
                else:
                    # Assume sequential classes starting from 0
                    min_cols = min(probs.shape[1], n_classes)
                    proper_probs[:, :min_cols] = probs[:, :min_cols]
                reconstruction_probs[:, start_idx:end_idx] = proper_probs
            else:
                reconstruction_probs[:, start_idx:end_idx] = probs

    return reconstruction_probs


def tabicl_rule_learning(dataset, max_antecedents=2, context_samples=100,
                         ant_similarity=0.5, cons_similarity=0.8):
    """
    End-to-end unsupervised rule learning using TabICL.

    Args:
        dataset: DataFrame with categorical features
        max_antecedents: Maximum antecedents per rule
        context_samples: Number of samples to use as context
        ant_similarity: Antecedent threshold
        cons_similarity: Consequent threshold

    Returns:
        rules: List of extracted association rules
    """

    # Prepare data
    encoded_data, classes_per_feature, feature_names, encoder = prepare_categorical_data(dataset)

    print(f"Dataset shape: {encoded_data.shape}")
    print(f"Number of features: {len(classes_per_feature)}")
    print(f"Classes per feature: {classes_per_feature}")

    # Generate test matrix (query patterns)
    # Use equal probabilities for unmarked features (NOT zeros)
    # Since we add noise to the context, TabICL will see values between 0 and 1,
    # making [0.33, 0.33, 0.33] patterns more natural and consistent with the training distribution
    test_matrix, test_descriptions, feature_value_indices = generate_aerial_test_matrix(
        n_features=len(classes_per_feature),
        classes_per_feature=classes_per_feature,
        max_antecedents=max_antecedents,
        use_zeros_for_unmarked=False  # Equal probabilities work better with noisy context
    )

    print(f"\nTest matrix shape: {test_matrix.shape}")
    print(f"Number of test vectors: {len(test_descriptions)}")

    # Initialize TabICL
    tabicl_model = TabICLClassifier()

    # Adapt TabICL for reconstruction
    print(f"\nUsing TabICL for pattern reconstruction...")
    reconstruction_probs = adapt_tabicl_for_reconstruction(
        tabicl_model=tabicl_model,
        context_table=encoded_data,
        query_matrix=test_matrix,
        feature_value_indices=feature_value_indices,
        n_samples=context_samples
    )

    print(f"Reconstruction shape: {reconstruction_probs.shape}")

    # Extract rules using PyAerial logic
    print(f"\nExtracting rules...")
    rules = extract_rules_from_reconstruction(
        prob_matrix=reconstruction_probs,
        test_descriptions=test_descriptions,
        feature_value_indices=feature_value_indices,
        ant_similarity=ant_similarity,
        cons_similarity=cons_similarity,
        feature_names=feature_names,
        encoder=encoder  # Pass encoder to map class indices to actual values
    )

    print(f"{len(rules)} rules found!")

    return rules, feature_names


# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("TabICL Baseline for Rule Learning")
    print("=" * 80)

    # Load dataset
    print("\n" + "=" * 80)
    print("Loading breast cancer dataset...")
    breast_cancer = fetch_ucirepo(id=14)
    # congressional_voting_records = fetch_ucirepo(id=105)
    X_features = breast_cancer.data.features
    y_target = breast_cancer.data.targets

    # Combine features and target into single table
    full_data = pd.concat([X_features, y_target], axis=1)

    print(f"Features shape: {X_features.shape}")
    print(f"Target shape: {y_target.shape}")
    print(f"Combined data shape: {full_data.shape}")

    # Extract rules with timing
    print("\n" + "=" * 80)
    print("Starting rule extraction...")
    start_time = time.time()

    extracted_rules, feature_names = tabicl_rule_learning(
        dataset=full_data,
        max_antecedents=2,
        context_samples=X_features.shape[0],
        ant_similarity=0.0,  # Disabled: TabICL cannot validate antecedents
        cons_similarity=0.8
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Display results
    print(f"\n{'=' * 80}")
    print(f"RESULTS")
    print(f"{'=' * 80}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Extracted {len(extracted_rules)} rules")

    if len(extracted_rules) > 0:
        # Calculate rule quality metrics
        print("\n" + "=" * 80)
        print("CALCULATING RULE QUALITY METRICS...")
        print("=" * 80)

        # Convert original data to proper format for metrics calculation
        original_data_array = full_data.values

        # Calculate metrics for all rules
        rules_with_metrics, avg_metrics = calculate_rule_metrics(
            rules=extracted_rules,
            data=original_data_array,
            feature_names=feature_names,
            metric_names=['support', 'confidence', 'rule_coverage', 'zhangs_metric']
        )

        # Display first 10 rules with their metrics
        print("\nFirst 10 rules with metrics:")
        for idx, rule in enumerate(rules_with_metrics[:10], 1):
            antecedents_str = " AND ".join([f"{a['feature']}={a['value']}" for a in rule['antecedents']])
            consequent_str = f"{rule['consequent']['feature']}={rule['consequent']['value']}"
            print(f"{idx}. {antecedents_str} => {consequent_str}")
            print(f"    Support: {rule['support']:.4f}, Confidence: {rule['confidence']:.4f}, "
                  f"Coverage: {rule['rule_coverage']:.4f}, Zhang: {rule['zhangs_metric']:.4f}")

        # Display average metrics
        print("\n" + "=" * 80)
        print("RULE QUALITY METRICS (Averages)")
        print("=" * 80)

        for metric_name, value in avg_metrics.items():
            print(f"{metric_name}: {value:.4f}")

    else:
        print("\nWARNING: No rules extracted! ")
