"""
Seed Management Utilities for Reproducible Experiments

This module provides utilities for managing random seeds across experiments
to ensure reproducibility in machine learning research.

Key Features:
- Generate consistent seed sequences for multiple experimental runs
- Set seeds for all major random number generators (Python, NumPy, PyTorch)
- Save and load seed configurations
- Create deterministic experimental setups

Usage:
    from src.utils.seed_utils import set_seed, generate_seed_sequence

    # For a single experiment
    set_seed(42)

    # For multiple runs
    seeds = generate_seed_sequence(base_seed=42, n_runs=10)
    for run_idx, seed in enumerate(seeds):
        set_seed(seed)
        # Run experiment...
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
        new_seed = rng.randint(0, 2**31 - 1)
        if new_seed not in seeds:
            seeds.append(new_seed)

    return seeds


def get_default_seeds(n_runs: int, base_seed: int = 42) -> List[int]:
    """
    Get default seed sequence for experiments.

    This is a convenience function that uses a standard base seed (42)
    which is commonly used in ML research.

    Args:
        n_runs: Number of experimental runs
        base_seed: Base seed (default: 42, the answer to everything)

    Returns:
        List of seeds for each run

    Example:
        >>> seeds = get_default_seeds(3)
        >>> for i, seed in enumerate(seeds):
        ...     print(f"Run {i+1}: seed={seed}")
    """
    return generate_seed_sequence(base_seed, n_runs)


def create_seed_config(base_seed: int, n_runs: int) -> dict:
    """
    Create a seed configuration dictionary for experiments.

    This is useful for saving experimental parameters including seeds.

    Args:
        base_seed: Base seed for generation
        n_runs: Number of runs

    Returns:
        Dictionary containing seed configuration

    Example:
        >>> config = create_seed_config(42, 3)
        >>> print(config)
        {'base_seed': 42, 'n_runs': 3, 'seeds': [42, 12345, 67890]}
    """
    seeds = generate_seed_sequence(base_seed, n_runs)
    return {
        'base_seed': base_seed,
        'n_runs': n_runs,
        'seeds': seeds
    }


def validate_reproducibility(seed: int, n_samples: int = 1000) -> bool:
    """
    Validate that seed setting works correctly by generating random numbers twice.

    This is a sanity check function to ensure reproducibility is working.

    Args:
        seed: Seed to test
        n_samples: Number of random samples to generate for testing

    Returns:
        True if reproducibility works, False otherwise

    Example:
        >>> assert validate_reproducibility(42)
    """
    # First run
    set_seed(seed)
    samples1 = np.random.rand(n_samples)

    # Second run with same seed
    set_seed(seed)
    samples2 = np.random.rand(n_samples)

    # Check if identical
    return np.allclose(samples1, samples2)