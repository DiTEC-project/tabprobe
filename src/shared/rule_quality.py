"""
Rule Quality Metrics Module

This module provides efficient computation of rule quality metrics for
association rules. Designed to work with PyAerial's rule output structure.

Supported Metrics:
- Support: Frequency of the complete rule (antecedent + consequent) in the dataset
- Confidence: P(consequent | antecedent)
- Zhang's Metric: Correlation measure between antecedent and consequent
- Rule Coverage (LHS Support): Frequency of the antecedent alone

Rule Format (PyAerial compatible):
{
    "antecedents": [{"feature": "feature_name", "value": "category_value"}, ...],
    "consequent": {"feature": "feature_name", "value": "category_value"},
    ...
}
"""

import numpy as np
from typing import List, Dict, Optional


def _get_feature_index(feature, feature_names: Optional[List[str]]) -> int:
    """
    Get the column index for a feature.

    Args:
        feature: Feature name or index
        feature_names: Optional list of feature names

    Returns:
        Feature index
    """
    if isinstance(feature, int):
        return feature
    elif feature_names and feature in feature_names:
        return feature_names.index(feature)
    else:
        raise ValueError(f"Unknown feature: {feature}")


def _get_mask_for_conditions(conditions: List[Dict], data: np.ndarray,
                             feature_names: Optional[List[str]] = None) -> np.ndarray:
    """
    Create boolean mask for rows matching all conditions.

    Args:
        conditions: List of condition dictionaries with 'feature' and 'value'
        data: Dataset array
        feature_names: Optional list of feature names

    Returns:
        Boolean array of shape (n_samples,)
    """
    mask = np.ones(data.shape[0], dtype=bool)

    for condition in conditions:
        feature_idx = _get_feature_index(condition['feature'], feature_names)
        value = condition['value']
        mask &= (data[:, feature_idx] == value)

    return mask


def calculate_support(rule: Dict, data: np.ndarray,
                      feature_names: Optional[List[str]] = None) -> float:
    """
    Calculate support: P(antecedent AND consequent).

    Support is the proportion of transactions containing both the antecedent
    and consequent.

    Args:
        rule: Rule dictionary with 'antecedents' and 'consequent'
        data: Dataset as numpy array (n_samples, n_features)
        feature_names: Optional list of feature names

    Returns:
        Support value in [0, 1]
    """
    all_conditions = rule['antecedents'] + [rule['consequent']]
    mask = _get_mask_for_conditions(all_conditions, data, feature_names)
    return float(np.sum(mask)) / data.shape[0]


def calculate_confidence(rule: Dict, data: np.ndarray,
                         feature_names: Optional[List[str]] = None) -> float:
    """
    Calculate confidence: P(consequent | antecedent).

    Confidence is the proportion of transactions with the antecedent
    that also contain the consequent.

    Args:
        rule: Rule dictionary with 'antecedents' and 'consequent'
        data: Dataset as numpy array (n_samples, n_features)
        feature_names: Optional list of feature names

    Returns:
        Confidence value in [0, 1]
    """
    antecedent_mask = _get_mask_for_conditions(rule['antecedents'], data, feature_names)
    antecedent_count = np.sum(antecedent_mask)

    if antecedent_count == 0:
        return 0.0

    all_conditions = rule['antecedents'] + [rule['consequent']]
    full_mask = _get_mask_for_conditions(all_conditions, data, feature_names)
    full_count = np.sum(full_mask)

    return float(full_count) / float(antecedent_count)


def calculate_rule_coverage(rule: Dict, data: np.ndarray,
                            feature_names: Optional[List[str]] = None) -> float:
    """
    Calculate rule coverage (LHS support): P(antecedent).

    Rule coverage is the proportion of transactions containing the antecedent.

    Args:
        rule: Rule dictionary with 'antecedents'
        data: Dataset as numpy array (n_samples, n_features)
        feature_names: Optional list of feature names

    Returns:
        Rule coverage value in [0, 1]
    """
    antecedent_mask = _get_mask_for_conditions(rule['antecedents'], data, feature_names)
    return float(np.sum(antecedent_mask)) / data.shape[0]


def calculate_zhangs_metric(rule: Dict, data: np.ndarray,
                            feature_names: Optional[List[str]] = None) -> float:
    """
    Calculate Zhang's metric: correlation measure between antecedent and consequent.

    Zhang's metric is defined as:
    Z = (confidence - expected_confidence) / max(confidence, expected_confidence) * (1 - expected_confidence)

    where expected_confidence = P(consequent)

    Range: [-1, 1]
    - Z > 0: positive correlation
    - Z = 0: independence
    - Z < 0: negative correlation

    Args:
        rule: Rule dictionary with 'antecedents' and 'consequent'
        data: Dataset as numpy array (n_samples, n_features)
        feature_names: Optional list of feature names

    Returns:
        Zhang's metric value in [-1, 1]
    """
    confidence = calculate_confidence(rule, data, feature_names)
    consequent_mask = _get_mask_for_conditions([rule['consequent']], data, feature_names)
    p_consequent = float(np.sum(consequent_mask)) / data.shape[0]

    if p_consequent == 0 or p_consequent == 1:
        return 0.0

    numerator = confidence - p_consequent

    if numerator == 0:
        return 0.0

    if confidence > p_consequent:
        denominator = max(confidence, p_consequent) * (1 - p_consequent)
    else:
        denominator = max(confidence, p_consequent) * (1 - confidence)

    if denominator == 0:
        return 0.0

    return numerator / denominator


def calculate_rule_metrics(rules: List[Dict], data: np.ndarray,
                           feature_names: Optional[List[str]] = None,
                           metric_names: Optional[List[str]] = None) -> tuple:
    """
    Calculate specified metrics for each rule and compute averages.

    This function:
    1. Calculates the specified metrics for each rule
    2. Extends each rule dictionary with the calculated metric values
    3. Computes average values across all rules for each metric

    Args:
        rules: List of rule dictionaries (PyAerial format)
        data: Dataset as numpy array (n_samples, n_features)
        feature_names: Optional list of feature names
        metric_names: List of metric names to calculate. If None, calculates all metrics.
                     Available: 'support', 'confidence', 'rule_coverage', 'zhangs_metric'

    Returns:
        Tuple of (rules_with_metrics, average_metrics):
        - rules_with_metrics: List of rules with metric values added
        - average_metrics: Dictionary with average values for each metric plus 'num_rules'
    """
    # Mapping of metric names to their calculation functions
    METRIC_FUNCTIONS = {
        'support': calculate_support,
        'confidence': calculate_confidence,
        'rule_coverage': calculate_rule_coverage,
        'zhangs_metric': calculate_zhangs_metric,
    }

    if metric_names is None:
        metric_names = ['support', 'confidence', 'rule_coverage', 'zhangs_metric']

    # Validate metric names
    for metric_name in metric_names:
        if metric_name not in METRIC_FUNCTIONS:
            raise ValueError(f"Unknown metric: {metric_name}. Available: {list(METRIC_FUNCTIONS.keys())}")

    # If no rules, return empty results
    if not rules:
        average_metrics = {metric: 0.0 for metric in metric_names}
        average_metrics['num_rules'] = 0
        return [], average_metrics

    # Calculate metrics for each rule
    rules_with_metrics = []
    metric_sums = {metric: 0.0 for metric in metric_names}

    for rule in rules:
        # Copy rule to avoid modifying original
        rule_with_metrics = rule.copy()

        # Calculate each requested metric
        for metric_name in metric_names:
            metric_func = METRIC_FUNCTIONS[metric_name]
            metric_value = metric_func(rule, data, feature_names)
            rule_with_metrics[metric_name] = metric_value
            metric_sums[metric_name] += metric_value

        rules_with_metrics.append(rule_with_metrics)

    # Compute averages
    n_rules = len(rules)
    average_metrics = {metric: metric_sums[metric] / n_rules for metric in metric_names}
    average_metrics['num_rules'] = n_rules

    return rules_with_metrics, average_metrics
