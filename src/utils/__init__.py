"""
Shared utilities for the rulepfn project.

This package contains modularized, reusable components that can be used
across different experiments and approaches.

Submodules:
- aerial: Aerial-style rule extraction utilities
- data_loading: Dataset loading utilities
- discretization: Feature discretization utilities
- rule_quality: Rule quality metrics
- seed_utils: Seed management for reproducibility
- rule_saving: Rule saving and FP-Growth calibration utilities
"""

from . import aerial
from .data_loading import (
    get_ucimlrepo_datasets,
    get_gene_expression_datasets,
    list_available_datasets,
)
from .discretization import discretize_numerical_features
from .rule_quality import calculate_rule_metrics
from .seed_utils import set_seed, generate_seed_sequence
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

__all__ = [
    'aerial',
    'get_ucimlrepo_datasets',
    'get_gene_expression_datasets',
    'list_available_datasets',
    'discretize_numerical_features',
    'calculate_rule_metrics',
    'set_seed',
    'generate_seed_sequence',
    'save_rules',
    'load_rules',
    'strip_rule_to_essentials',
    'convert_metrics_to_stats',
    'calculate_fpgrowth_calibration_threshold',
    'save_fpgrowth_calibration',
    'load_fpgrowth_calibration',
    'calculate_and_save_all_calibrations',
]
