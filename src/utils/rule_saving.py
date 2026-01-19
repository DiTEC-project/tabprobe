"""
Save association rules into json files

Common utilities for saving rules across all rule mining methods
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
