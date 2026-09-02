"""
Shared probing orchestration for TabProbe rule extraction across model backends
(TabPFN, TabICL, TabDPT, XGBoost, RandomForest).
"""

from src.utils.test_matrix import generate_test_matrix
from src.utils.rule_extraction import extract_rules_from_reconstruction, get_significant_single_items


def probe_and_extract_rules(classes_per_feature, max_antecedents, adapt_fn,
                            ant_similarity, cons_similarity, feature_names, encoder,
                            use_zeros_for_unmarked=False):
    """
    Runs the probe -> reconstruct -> extract pipeline, with pruning for antecedent
    lengths beyond 2.

    Antecedent lengths 1-2 are always enumerated exhaustively (identical to the
    non-pruned implementation). For max_antecedents > 2, lengths 3+ are restricted to
    items that individually passed ant_similarity at length 1 (Apriori-style candidate
    pruning) -- this only trims the length-3+ search space, it does not change
    length-1/2 results.

    Args:
        classes_per_feature: List of number of classes for each feature
        max_antecedents: Maximum antecedents per rule
        adapt_fn: callable(query_matrix, feature_value_indices) -> reconstruction_probs.
            Wraps a specific backend's fit/predict (TabPFN/TabICL/TabDPT/XGBoost/
            RandomForest); the model is only ever fit once per feature inside adapt_fn,
            called once for the base pass and (if needed) once more for the pruned pass.
        ant_similarity, cons_similarity, feature_names, encoder: forwarded to
            extract_rules_from_reconstruction
        use_zeros_for_unmarked: forwarded to generate_test_matrix

    Returns:
        rules: List of extracted association rules
        feature_value_indices: List of dicts with 'start', 'end', 'feature' for each feature
    """
    n_features = len(classes_per_feature)
    base_max = min(max_antecedents, 2)

    test_matrix, test_descriptions, feature_value_indices = generate_test_matrix(
        n_features=n_features,
        classes_per_feature=classes_per_feature,
        max_antecedents=base_max,
        use_zeros_for_unmarked=use_zeros_for_unmarked
    )
    print(f"\nTest matrix shape (antecedents 1-{base_max}): {test_matrix.shape}")

    reconstruction_probs = adapt_fn(test_matrix, feature_value_indices)
    print(f"Reconstruction shape: {reconstruction_probs.shape}")

    rules = extract_rules_from_reconstruction(
        prob_matrix=reconstruction_probs,
        test_descriptions=test_descriptions,
        feature_value_indices=feature_value_indices,
        ant_similarity=ant_similarity,
        cons_similarity=cons_similarity,
        feature_names=feature_names,
        encoder=encoder
    )

    if max_antecedents > 2:
        significant_items = get_significant_single_items(
            reconstruction_probs, test_descriptions, feature_value_indices, ant_similarity
        )
        print(f"\n{len(significant_items)} single items passed ant_similarity>={ant_similarity}; "
              f"pruning antecedents 3-{max_antecedents} to combinations of these only...")

        pruned_matrix, pruned_descriptions, _ = generate_test_matrix(
            n_features=n_features,
            classes_per_feature=classes_per_feature,
            max_antecedents=max_antecedents,
            min_antecedents=3,
            use_zeros_for_unmarked=use_zeros_for_unmarked,
            allowed_items=significant_items
        )
        print(f"Pruned test matrix shape (antecedents 3-{max_antecedents}): {pruned_matrix.shape}")

        if len(pruned_descriptions) > 0:
            pruned_probs = adapt_fn(pruned_matrix, feature_value_indices)
            pruned_rules = extract_rules_from_reconstruction(
                prob_matrix=pruned_probs,
                test_descriptions=pruned_descriptions,
                feature_value_indices=feature_value_indices,
                ant_similarity=ant_similarity,
                cons_similarity=cons_similarity,
                feature_names=feature_names,
                encoder=encoder
            )
            rules += pruned_rules

    print(f"{len(rules)} rules found!")
    return rules, feature_value_indices
