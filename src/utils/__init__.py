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

__all__ = [
    'aerial',
    'get_ucimlrepo_datasets',
    'get_gene_expression_datasets',
    'list_available_datasets',
    'discretize_numerical_features',
    'calculate_rule_metrics',
    'set_seed',
    'generate_seed_sequence',
]
