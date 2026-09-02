"""
XGBoost-based Frequent Itemset Mining

Adapts XGBoost for frequent itemset discovery using TabProbe
"""

from src.utils.data_prep import prepare_categorical_data
from src.utils.test_matrix import generate_test_matrix
from src.utils.rule_extraction import extract_frequent_itemsets_from_reconstruction
from src.experiments.rule_mining.xgboost_experiments import (
    adapt_xgb_for_reconstruction,
    _resolve_calibration,
)


def xgb_itemset_learning(dataset, max_itemset_length=2, context_samples=None, similarity=0.5,
                         n_estimators=100, max_depth=3, learning_rate=0.3, random_state=42,
                         max_workers=1, query_batch_size=512, device='cpu',
                         auto_calibrate=True):
    """Frequent itemset mining using XGBoost.

    auto_calibrate applies the same per-dataset calibration policy as xgb_rule_learning
    (see xgboost_experiments.CALIBRATE_DATASETS), based on dataset.attrs['name'].
    """
    encoded_data, classes_per_feature, feature_names, encoder = prepare_categorical_data(dataset)

    calibrate, calibration_method = (
        _resolve_calibration(dataset.attrs.get('name')) if auto_calibrate else (False, 'sigmoid')
    )

    test_matrix, test_descriptions, feature_value_indices = generate_test_matrix(
        n_features=len(classes_per_feature),
        classes_per_feature=classes_per_feature,
        max_antecedents=max_itemset_length,
        use_zeros_for_unmarked=False
    )

    reconstruction_probs = adapt_xgb_for_reconstruction(
        context_table=encoded_data,
        query_matrix=test_matrix,
        feature_value_indices=feature_value_indices,
        n_samples=context_samples,
        calibrate=calibrate,
        calibration_method=calibration_method,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        max_workers=max_workers,
        query_batch_size=query_batch_size,
        device=device,
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