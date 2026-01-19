"""
Seed Management Utilities for Reproducible Experiments

This module provides utilities for managing random seeds across experiments to ensure reproducibility

Key Features:
- Generate consistent seed sequences for multiple experimental runs
- Set seeds for all major random number generators (Python, NumPy, PyTorch)
- Save and load seed configurations
- Create deterministic experimental setups
"""

import random
import numpy as np
from typing import List, Optional


def set_seed(seed: int) -> None:
    """
    Set random seed for all major libraries to ensure reproducibility.

    This function sets seeds for:
    - Python's built-in random module
    - NumPy
    - PyTorch (if available)

    Args:
        seed: Integer seed value

    Example:
        >>> set_seed(42)
        >>> # All subsequent random operations will be deterministic
    """
    random.seed(seed)
    np.random.seed(seed)

    # Set PyTorch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For deterministic behavior in CUDA operations
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not installed, skip


def generate_seed_sequence(base_seed: int, n_runs: int) -> List[int]:
    """
    Generate a reproducible sequence of seeds for multiple experimental runs.

    Uses a deterministic approach to generate N different seeds from a base seed.
    This ensures that given the same base_seed, you always get the same sequence.

    Args:
        base_seed: The base seed to generate sequence from
        n_runs: Number of seeds to generate

    Returns:
        List of n_runs seed values

    Example:
        >>> seeds = generate_seed_sequence(42, 5)
        >>> print(seeds)
        [42, 12345, 67890, 24680, 13579]  # Deterministic sequence

    Note:
        The first seed in the sequence is always the base_seed itself.
    """
    # Set seed for reproducible generation
    rng = np.random.RandomState(base_seed)

    # Generate n_runs unique seeds
    # First seed is the base_seed, rest are randomly generated
    seeds = [base_seed]

    # Generate additional seeds (avoid duplicates)
    while len(seeds) < n_runs:
        new_seed = int(rng.randint(0, 2**31 - 1))
        if new_seed not in seeds:
            seeds.append(new_seed)

    return seeds
