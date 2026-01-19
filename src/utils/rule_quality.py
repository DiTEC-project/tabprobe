"""
Rule Quality Metrics Module

This module provides efficient computation of rule quality metrics for
association rules.

Supported Metrics:
- Support: Frequency of the complete rule (antecedent + consequent) in the dataset
- Confidence: P(consequent | antecedent)
- Zhang's Metric: Correlation measure between antecedent and consequent
- Rule Coverage (LHS Support): Frequency of the antecedent alone
- Interestingness: Confidence * (support / rhs_support) * (1 - (support / input_length))

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
    """Get the column index for a feature."""
    if isinstance(feature, int):
        return feature
    elif feature_names and feature in feature_names:
        return feature_names.index(feature)
    else:
        raise ValueError(f"Unknown feature: {feature}")


def _get_mask_for_conditions(conditions: List[Dict], data: np.ndarray,
                             feature_names: Optional[List[str]] = None) -> np.ndarray:
    """Create boolean mask for rows matching all conditions."""
    mask = np.ones(data.shape[0], dtype=bool)
    for condition in conditions:
        feature_idx = _get_feature_index(condition['feature'], feature_names)
        value = condition['value']
        mask &= (data[:, feature_idx] == value)
    return mask


def _calculate_base_metrics(rule: Dict, data: np.ndarray,
                            feature_names: Optional[List[str]] = None) -> tuple:
    """
    Calculate base metrics that other metrics depend on.

    Returns tuple of (metrics_dict, antecedent_mask)
    """
    n_samples = data.shape[0]

    antecedent_mask = _get_mask_for_conditions(rule['antecedents'], data, feature_names)
    consequent_mask = _get_mask_for_conditions([rule['consequent']], data, feature_names)
    full_mask = antecedent_mask & consequent_mask

    antecedent_count = np.sum(antecedent_mask)
    consequent_count = np.sum(consequent_mask)
    full_count = np.sum(full_mask)

    support = float(full_count) / n_samples
    rule_coverage = float(antecedent_count) / n_samples
    rhs_support = float(consequent_count) / n_samples
    confidence = float(full_count) / float(antecedent_count) if antecedent_count > 0 else 0.0

    metrics = {
        'support': support,
        'confidence': confidence,
        'rule_coverage': rule_coverage,
        'rhs_support': rhs_support,
    }

    return metrics, antecedent_mask


def _calculate_zhangs_metric(base_metrics: Dict[str, float]) -> float:
    """Calculate Zhang's metric from pre-calculated base metrics."""
    confidence = base_metrics['confidence']
    p_consequent = base_metrics['rhs_support']

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


def _calculate_interestingness(base_metrics: Dict[str, float], n_samples: int) -> float:
    """Calculate interestingness from pre-calculated base metrics."""
    confidence = base_metrics['confidence']
    support = base_metrics['support']
    rhs_support = base_metrics['rhs_support']

    if rhs_support == 0:
        return 0.0

    return confidence * (support / rhs_support) * (1 - (support / n_samples))


def calculate_rule_metrics(rules: List[Dict], data: np.ndarray,
                           feature_names: Optional[List[str]] = None,
                           metric_names: Optional[List[str]] = None) -> tuple:
    """
    Calculate specified metrics for each rule and compute averages.

    Efficiently calculates metrics by computing base metrics once and reusing them
    for derived metrics.

    Args:
        rules: List of rule dictionaries (PyAerial format)
        data: Dataset as numpy array (n_samples, n_features)
        feature_names: Optional list of feature names
        metric_names: List of metric names to calculate. If None, calculates all metrics.
                     Available: 'support', 'confidence', 'rule_coverage', 'zhangs_metric', 'interestingness'

    Returns:
        Tuple of (rules_with_metrics, average_metrics):
        - rules_with_metrics: List of rules with metric values added
        - average_metrics: Dictionary with average values for each metric plus 'num_rules'
    """
    available_metrics = ['support', 'confidence', 'rule_coverage', 'zhangs_metric', 'interestingness']

    if metric_names is None:
        metric_names = available_metrics

    # Validate metric names
    for metric_name in metric_names:
        if metric_name not in available_metrics:
            raise ValueError(f"Unknown metric: {metric_name}. Available: {available_metrics}")

    # If no rules, return empty results
    if not rules:
        average_metrics = {metric: 0.0 for metric in metric_names}
        average_metrics['num_rules'] = 0
        return [], average_metrics

    n_samples = data.shape[0]
    rules_with_metrics = []
    metric_sums = {metric: 0.0 for metric in metric_names}
    dataset_coverage = np.zeros(n_samples, dtype=bool)

    for rule in rules:
        rule_with_metrics = rule.copy()

        # Calculate base metrics once
        base_metrics, antecedent_mask = _calculate_base_metrics(rule, data, feature_names)

        # Track dataset coverage
        dataset_coverage |= antecedent_mask

        # Add requested base metrics
        for metric in ['support', 'confidence', 'rule_coverage']:
            if metric in metric_names:
                rule_with_metrics[metric] = base_metrics[metric]
                metric_sums[metric] += base_metrics[metric]

        # Calculate derived metrics using base metrics
        if 'zhangs_metric' in metric_names:
            zhangs = _calculate_zhangs_metric(base_metrics)
            rule_with_metrics['zhangs_metric'] = zhangs
            metric_sums['zhangs_metric'] += zhangs

        if 'interestingness' in metric_names:
            interestingness = _calculate_interestingness(base_metrics, n_samples)
            rule_with_metrics['interestingness'] = interestingness
            metric_sums['interestingness'] += interestingness

        rules_with_metrics.append(rule_with_metrics)

    # Compute averages
    n_rules = len(rules)
    average_metrics = {metric: metric_sums[metric] / n_rules for metric in metric_names}
    average_metrics['num_rules'] = n_rules
    average_metrics['data_coverage'] = float(np.sum(dataset_coverage)) / n_samples

    return rules_with_metrics, average_metrics
