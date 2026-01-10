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


def calculate_fpgrowth_itemset_calibration_threshold(dataset_name: str, reference_method: str = "aerial",
                                                      coverage_percentage: float = 0.9,
                                                      output_dir: str = "out/frequent_itemsets") -> float:
    """
    Calculate minimum support threshold for FP-Growth to discover at least
    coverage_percentage of reference method's itemsets.

    This calibration ensures fair comparison between FP-Growth and learning-based methods
    by adjusting FP-Growth's min_support to match the reference method's itemset discovery.

    Args:
        dataset_name: Name of dataset
        reference_method: Reference method to calibrate against (default: "aerial")
        coverage_percentage: Target coverage percentage (default: 0.9 for 90%)
        output_dir: Base output directory

    Returns:
        min_support: Calibrated minimum support threshold for FP-Growth
    """
    # Load all reference itemsets (across all seeds)
    reference_dir = os.path.join(output_dir, reference_method, dataset_name)

    if not os.path.exists(reference_dir):
        raise FileNotFoundError(f"Reference itemsets not found: {reference_dir}")

    # Collect all itemsets from all seeds
    all_reference_itemsets = []
    for filename in os.listdir(reference_dir):
        if filename.startswith("seed_") and filename.endswith(".json"):
            filepath = os.path.join(reference_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                all_reference_itemsets.extend(data['itemsets'])

    if len(all_reference_itemsets) == 0:
        raise ValueError(f"No reference itemsets found for {dataset_name}")

    # Get all support values
    support_values = [itemset['support'] for itemset in all_reference_itemsets]

    # Calculate threshold at coverage_percentage percentile
    # Use lower percentile to ensure we capture at least coverage_percentage of itemsets
    target_percentile = (1.0 - coverage_percentage) * 100
    min_support = float(np.percentile(support_values, target_percentile))

    # Ensure min_support is reasonable (not too low)
    min_support = max(min_support, 0.01)  # At least 1% support

    return min_support


def save_fpgrowth_itemset_calibration(dataset_name: str, min_support: float,
                                       reference_method: str = "aerial",
                                       coverage_percentage: float = 0.9,
                                       output_dir: str = "out/frequent_itemsets"):
    """
    Save FP-Growth calibration threshold to JSON file.

    Saved in the reference method's directory for traceability:
    out/frequent_itemsets/{reference_method}/{dataset}/fpgrowth_calibration.json

    Args:
        dataset_name: Name of dataset
        min_support: Calibrated minimum support threshold
        reference_method: Reference method used for calibration
        coverage_percentage: Target coverage percentage
        output_dir: Base output directory
    """
    reference_dir = os.path.join(output_dir, reference_method, dataset_name)
    os.makedirs(reference_dir, exist_ok=True)

    calibration_file = os.path.join(reference_dir, "fpgrowth_calibration.json")

    data = {
        'dataset': dataset_name,
        'reference_method': reference_method,
        'coverage_percentage': coverage_percentage,
        'min_support': min_support
    }

    with open(calibration_file, 'w') as f:
        json.dump(data, f, indent=2)


def load_fpgrowth_itemset_calibration(dataset_name: str, reference_method: str = "aerial",
                                       output_dir: str = "out/frequent_itemsets") -> Dict:
    """
    Load FP-Growth calibration threshold from JSON file.

    Args:
        dataset_name: Name of dataset
        reference_method: Reference method used for calibration
        output_dir: Base output directory

    Returns:
        Dictionary with calibration data including 'min_support'
    """
    reference_dir = os.path.join(output_dir, reference_method, dataset_name)
    calibration_file = os.path.join(reference_dir, "fpgrowth_calibration.json")

    if not os.path.exists(calibration_file):
        raise FileNotFoundError(f"Calibration file not found: {calibration_file}")

    with open(calibration_file, 'r') as f:
        data = json.load(f)

    return data


def calculate_and_save_all_itemset_calibrations(dataset_names: Optional[List[str]] = None,
                                                 reference_method: str = "aerial",
                                                 coverage_percentage: float = 0.9,
                                                 output_dir: str = "out/frequent_itemsets"):
    """
    Calculate and save FP-Growth calibration thresholds for all datasets.

    Args:
        dataset_names: List of dataset names (if None, auto-detect from reference method directory)
        reference_method: Reference method for calibration
        coverage_percentage: Target coverage percentage
        output_dir: Base output directory
    """
    reference_base = os.path.join(output_dir, reference_method)

    if dataset_names is None:
        # Auto-detect datasets from reference method directory
        if not os.path.exists(reference_base):
            raise FileNotFoundError(f"Reference method directory not found: {reference_base}")

        dataset_names = [d for d in os.listdir(reference_base)
                         if os.path.isdir(os.path.join(reference_base, d))]

    print(f"Calculating FP-Growth calibration for {len(dataset_names)} datasets...")

    for dataset_name in dataset_names:
        try:
            min_support = calculate_fpgrowth_itemset_calibration_threshold(
                dataset_name=dataset_name,
                reference_method=reference_method,
                coverage_percentage=coverage_percentage,
                output_dir=output_dir
            )

            save_fpgrowth_itemset_calibration(
                dataset_name=dataset_name,
                min_support=min_support,
                reference_method=reference_method,
                coverage_percentage=coverage_percentage,
                output_dir=output_dir
            )

            print(f"  {dataset_name}: min_support={min_support:.4f}")

        except Exception as e:
            print(f"  {dataset_name}: ERROR - {str(e)}")

    print("Calibration complete!")