"""
Itemset Saving and FP-Growth Calibration Utilities

Common utilities for saving frequent itemsets across all mining methods and
calculating FP-Growth calibration thresholds.
"""
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional


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


def save_itemsets(itemsets: List[Dict], stats: Dict[str, float], dataset_name: str,
                  method_name: str, seed: Any = None, output_dir: str = "out/frequent_itemsets") -> str:
    """
    Save frequent itemsets and metrics to JSON file.

    Args:
        itemsets: List of itemsets in PyAerial format
            [{'itemset': [{'feature': 'age', 'value': '30-39'}, ...], 'support': 0.5}, ...]
        stats: Dictionary with aggregated itemset statistics
        dataset_name: Name of dataset
        method_name: Name of method (aerial, tabpfn, tabicl, tabdpt, fpgrowth)
        seed: Random seed used for this run (None for deterministic methods like FP-Growth)
        output_dir: Base output directory (default: "out/frequent_itemsets")

    Returns:
        Path to saved file
    """
    # Create directory structure: out/frequent_itemsets/{method}/{dataset}/
    method_dir = os.path.join(output_dir, method_name, dataset_name)
    os.makedirs(method_dir, exist_ok=True)

    # Save to seed-specific file (or itemsets.json for deterministic methods)
    if seed is None:
        output_file = os.path.join(method_dir, "itemsets.json")
    else:
        output_file = os.path.join(method_dir, f"seed_{seed}.json")

    # Prepare data structure
    data = {
        'dataset': dataset_name,
        'method': method_name,
        'seed': seed,
        'itemsets': itemsets,
        'statistics': stats
    }

    # Write to JSON
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)

    return output_file


def load_itemsets(dataset_name: str, method_name: str, seed: Any = None,
                  output_dir: str = "out/frequent_itemsets") -> Dict:
    """
    Load frequent itemsets from JSON file.

    Args:
        dataset_name: Name of dataset
        method_name: Name of method (aerial, tabpfn, tabicl, tabdpt, fpgrowth)
        seed: Random seed (None for deterministic methods)
        output_dir: Base output directory (default: "out/frequent_itemsets")

    Returns:
        Dictionary with 'dataset', 'method', 'seed', 'itemsets', 'statistics'
    """
    method_dir = os.path.join(output_dir, method_name, dataset_name)

    if seed is None:
        input_file = os.path.join(method_dir, "itemsets.json")
    else:
        input_file = os.path.join(method_dir, f"seed_{seed}.json")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Itemsets file not found: {input_file}")

    with open(input_file, 'r') as f:
        data = json.load(f)

    return data
