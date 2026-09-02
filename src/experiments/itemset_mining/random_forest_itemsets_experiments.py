"""
RandomForest-based Frequent Itemset Mining

Adapts RandomForest for frequent itemset discovery using TabProbe
"""

from src.utils.data_prep import prepare_categorical_data
from src.utils.test_matrix import generate_test_matrix
from src.utils.rule_extraction import extract_frequent_itemsets_from_reconstruction
from src.experiments.rule_mining.random_forest_experiments import adapt_rf_for_reconstruction


def rf_itemset_learning(dataset, max_itemset_length=2, context_samples=None, similarity=0.5,
                        n_estimators=100, max_depth=None, min_samples_leaf=1, random_state=42,
                        max_workers=1, query_batch_size=512):
    """Frequent itemset mining using RandomForest."""
    encoded_data, classes_per_feature, feature_names, encoder = prepare_categorical_data(dataset)

    test_matrix, test_descriptions, feature_value_indices = generate_test_matrix(
        n_features=len(classes_per_feature),
        classes_per_feature=classes_per_feature,
        max_antecedents=max_itemset_length,
        use_zeros_for_unmarked=False
    )

    reconstruction_probs = adapt_rf_for_reconstruction(
        context_table=encoded_data,
        query_matrix=test_matrix,
        feature_value_indices=feature_value_indices,
        n_samples=context_samples,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        max_workers=max_workers,
        query_batch_size=query_batch_size,
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