"""
Shared utilities for the rulepfn project.

This package contains modularized, reusable components that can be used
across different experiments and approaches.

Modules:
- data_loading: Dataset loading utilities
- data_prep: Data preparation and encoding utilities
- discretization: Feature discretization utilities
- rule_extraction: Rule and itemset extraction utilities
- rule_quality: Rule quality metrics
- rule_saving: Rule saving and FP-Growth calibration utilities
- seed_utils: Seed management for reproducibility
- test_matrix: Test matrix generation utilities
"""

from .data_loading import get_ucimlrepo_datasets
from .data_prep import prepare_categorical_data, add_gaussian_noise
from .discretization import discretize_numerical_features
from .rule_extraction import (
    extract_rules_from_reconstruction,
    extract_frequent_itemsets_from_reconstruction,
    calculate_itemset_support
)
from .rule_quality import calculate_rule_metrics
from .rule_saving import (
    save_rules,
    load_rules,
    strip_rule_to_essentials,
    convert_metrics_to_stats,
    calculate_fpgrowth_calibration_threshold,
    save_fpgrowth_calibration,
    load_fpgrowth_calibration,
    calculate_and_save_all_calibrations,
)
from .seed_utils import set_seed, generate_seed_sequence
from .test_matrix import generate_test_matrix

__all__ = [
    # Data loading and preparation
    'get_ucimlrepo_datasets',
    'prepare_categorical_data',
    'add_gaussian_noise',
    'discretize_numerical_features',
    # Rule extraction
    'extract_rules_from_reconstruction',
    'extract_frequent_itemsets_from_reconstruction',
    'calculate_itemset_support',
    # Rule quality and saving
    'calculate_rule_metrics',
    'save_rules',
    'load_rules',
    'strip_rule_to_essentials',
    'convert_metrics_to_stats',
    'calculate_fpgrowth_calibration_threshold',
    'save_fpgrowth_calibration',
    'load_fpgrowth_calibration',
    'calculate_and_save_all_calibrations',
    # Seed management
    'set_seed',
    'generate_seed_sequence',
    # Test matrix generation
    'generate_test_matrix',
]
