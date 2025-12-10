"""
Shared Aerial utilities for rule extraction and data preparation.

This module provides reusable components for working with the Aerial approach
to association rule learning from tabular data.
"""

from .test_matrix import generate_aerial_test_matrix
from .rule_extraction import extract_rules_from_reconstruction
from .data_prep import prepare_categorical_data

__all__ = [
    'generate_aerial_test_matrix',
    'extract_rules_from_reconstruction',
    'prepare_categorical_data',
]