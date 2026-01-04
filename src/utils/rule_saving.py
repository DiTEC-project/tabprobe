"""
Rule Saving and FP-Growth Calibration Utilities

Common utilities for saving rules across all rule mining methods and
calculating FP-Growth calibration thresholds.
"""
import os
import json
import numpy as np
from typing import List, Dict, Any


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy and pandas types."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.str_):
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        return super().default(obj)


def strip_rule_to_essentials(rule: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strip rule down to essential fields needed for CBA/CORELS.

    Keeps only: antecedents, consequent, support, confidence
    Drops: zhangs_metric, interestingness, rule_coverage, data_coverage

    Args:
        rule: Rule dict with all metrics

    Returns:
        Rule dict with only essential fields
    """
    essential_fields = ['antecedents', 'consequent', 'support', 'confidence']
    return {k: v for k, v in rule.items() if k in essential_fields}


def convert_metrics_to_stats(avg_metrics: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert calculate_rule_metrics output to stats format for save_rules().

    This standardizes the metric format across all methods.

    Args:
        avg_metrics: Dictionary from calculate_rule_metrics

    Returns:
        Dictionary in stats format with standardized keys
    """
    return {
        'rule_count': avg_metrics['num_rules'],
        'average_support': avg_metrics['support'],
        'average_confidence': avg_metrics['confidence'],
        'average_zhangs_metric': avg_metrics['zhangs_metric'],
        'average_interestingness': avg_metrics['interestingness'],
        'average_coverage': avg_metrics['rule_coverage'],
        'data_coverage': avg_metrics['data_coverage']
    }


def save_rules(rules: List[Dict], stats: Dict[str, float], dataset_name: str,
               method_name: str, seed: Any = None, output_dir: str = "out/rules") -> str:
    """
    Save rules and metrics to JSON file.

    Automatically strips each rule to essential fields (antecedents, consequent, support, confidence)
    to reduce file size while keeping everything needed for CBA/CORELS.

    Args:
        rules: List of rules in PyAerial format (with individual metrics attached)
        stats: Dictionary with aggregated rule quality metrics
        dataset_name: Name of dataset
        method_name: Name of method (aerial, tabpfn, tabicl, tabdpt, fpgrowth)
        seed: Random seed used for this run (None for deterministic methods like FP-Growth)
        output_dir: Base output directory (default: "out/rules")

    Returns:
        Path to saved file
    """
    # Create directory structure: out/rules/{method}/{dataset}/
    method_dir = os.path.join(output_dir, method_name, dataset_name)
    os.makedirs(method_dir, exist_ok=True)

    # Save to seed-specific file (or rules.json for deterministic methods)
    if seed is None:
        output_file = os.path.join(method_dir, "rules.json")
    else:
        output_file = os.path.join(method_dir, f"seed_{seed}.json")

    # Strip rules to essential fields only (antecedents, consequent, support, confidence)
    essential_rules = [strip_rule_to_essentials(rule) for rule in rules]

    data = {
        'dataset': dataset_name,
        'method': method_name,
        'seed': seed,
        'rules': essential_rules,
        'rule_metrics': stats
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)

    return output_file


def load_rules(dataset_name: str, method_name: str, seed: int = None,
               output_dir: str = "out/rules") -> Dict[str, Any]:
    """
    Load rules from JSON file.

    Args:
        dataset_name: Name of dataset
        method_name: Name of method
        seed: Random seed (None for deterministic methods like FP-Growth)
        output_dir: Base output directory

    Returns:
        Dictionary with rules and metrics
    """
    if seed is None:
        rule_file = os.path.join(output_dir, method_name, dataset_name, "rules.json")
    else:
        rule_file = os.path.join(output_dir, method_name, dataset_name, f"seed_{seed}.json")

    with open(rule_file, 'r') as f:
        return json.load(f)


def calculate_fpgrowth_calibration_threshold(dataset_name: str, reference_method: str = "aerial",
                                             coverage_percentage: float = 0.9,
                                             output_dir: str = "out/rules") -> float:
    """
    Calculate minimum support threshold for FP-Growth that covers a percentage
    of rules discovered by the reference method.

    Strategy:
    1. Load all rules from reference method (all seeds)
    2. Calculate support for each unique rule
    3. Find threshold where FP-Growth would discover >= coverage_percentage of rules

    Args:
        dataset_name: Name of dataset
        reference_method: Reference method to calibrate against (default: "aerial")
        coverage_percentage: Target coverage (default: 0.9 for 90%)
        output_dir: Base output directory

    Returns:
        Minimum support threshold for FP-Growth
    """
    method_dir = os.path.join(output_dir, reference_method, dataset_name)

    if not os.path.exists(method_dir):
        raise ValueError(f"No rules found for {reference_method} on {dataset_name}. "
                         f"Run {reference_method} experiments first.")

    # Load all rule files for this dataset
    all_rules = []
    rule_files = [f for f in os.listdir(method_dir) if f.startswith("seed_") and f.endswith(".json")]

    if len(rule_files) == 0:
        raise ValueError(f"No rule files found in {method_dir}")

    for rule_file in rule_files:
        file_path = os.path.join(method_dir, rule_file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Extract rules with their support values
            for rule in data['rules']:
                if 'support' in rule:
                    all_rules.append(rule['support'])

    if len(all_rules) == 0:
        raise ValueError(f"No rules with support values found for {reference_method}")

    # Sort supports in descending order
    supports = sorted(all_rules, reverse=True)

    # Find support threshold that covers coverage_percentage of rules
    target_count = int(len(supports) * coverage_percentage)
    if target_count == 0:
        target_count = 1

    # The threshold is the support of the rule at the coverage_percentage percentile
    # We want FP-Growth to find at least this many rules, so we use slightly lower threshold
    threshold_support = supports[min(target_count - 1, len(supports) - 1)]

    # Use slightly lower threshold to ensure we get at least coverage_percentage
    # FP-Growth's min_support is inclusive, so we reduce by a small margin
    calibrated_threshold = max(threshold_support * 0.95, 0.01)

    return calibrated_threshold


def save_fpgrowth_calibration(dataset_name: str, threshold: float,
                              reference_method: str = "aerial",
                              coverage_percentage: float = 0.9,
                              output_dir: str = "out/rules") -> str:
    """
    Save FP-Growth calibration threshold to file.

    Args:
        dataset_name: Name of dataset
        threshold: Calculated minimum support threshold
        reference_method: Reference method used for calibration
        coverage_percentage: Target coverage percentage
        output_dir: Base output directory

    Returns:
        Path to saved calibration file
    """
    method_dir = os.path.join(output_dir, reference_method, dataset_name)
    os.makedirs(method_dir, exist_ok=True)

    calibration_file = os.path.join(method_dir, "fpgrowth_calibration.json")

    calibration_data = {
        'dataset': dataset_name,
        'reference_method': reference_method,
        'coverage_percentage': coverage_percentage,
        'min_support_threshold': threshold,
        'description': f'Minimum support threshold for FP-Growth to cover {coverage_percentage * 100:.0f}% of {reference_method} rules'
    }

    with open(calibration_file, 'w') as f:
        json.dump(calibration_data, f, indent=2, cls=NumpyEncoder)

    return calibration_file


def load_fpgrowth_calibration(dataset_name: str, reference_method: str = "aerial",
                              output_dir: str = "out/rules") -> float:
    """
    Load FP-Growth calibration threshold from file.

    Args:
        dataset_name: Name of dataset
        reference_method: Reference method used for calibration
        output_dir: Base output directory

    Returns:
        Minimum support threshold
    """
    calibration_file = os.path.join(output_dir, reference_method, dataset_name,
                                    "fpgrowth_calibration.json")

    if not os.path.exists(calibration_file):
        raise ValueError(f"Calibration file not found: {calibration_file}. "
                         f"Run {reference_method} experiments first and calculate calibration.")

    with open(calibration_file, 'r') as f:
        data = json.load(f)

    return data['min_support_threshold']


def calculate_and_save_all_calibrations(datasets: List[str], reference_method: str = "aerial",
                                        coverage_percentage: float = 0.9,
                                        output_dir: str = "out/rules") -> Dict[str, float]:
    """
    Calculate and save FP-Growth calibration thresholds for all datasets.

    Args:
        datasets: List of dataset names
        reference_method: Reference method for calibration
        coverage_percentage: Target coverage percentage
        output_dir: Base output directory

    Returns:
        Dictionary mapping dataset names to thresholds
    """
    thresholds = {}

    for dataset_name in datasets:
        try:
            threshold = calculate_fpgrowth_calibration_threshold(
                dataset_name=dataset_name,
                reference_method=reference_method,
                coverage_percentage=coverage_percentage,
                output_dir=output_dir
            )

            save_fpgrowth_calibration(
                dataset_name=dataset_name,
                threshold=threshold,
                reference_method=reference_method,
                coverage_percentage=coverage_percentage,
                output_dir=output_dir
            )

            thresholds[dataset_name] = threshold
            print(f"  {dataset_name}: min_support = {threshold:.4f}")

        except ValueError as e:
            print(f"  {dataset_name}: ERROR - {e}")
            thresholds[dataset_name] = None

    return thresholds
