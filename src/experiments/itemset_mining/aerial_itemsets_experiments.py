"""
PyAerial Frequent Itemset Mining Experiments

This module runs frequent itemset mining experiments using PyAerial.

PyAerial learns to reconstruct categorical data through a neural autoencoder,
then extracts frequent itemsets using the generate_frequent_itemsets() function.
"""
import time
import os
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from aerial import model, rule_extraction

from src.utils import (
    get_ucimlrepo_datasets,
    set_seed,
    generate_seed_sequence,
    save_itemsets,
    calculate_and_save_all_itemset_calibrations,
)


def aerial_itemset_learning(dataset, max_length=3, similarity=0.5, batch_size=None,
                             layer_dims=None, epochs=2, random_state=42):
    """
    End-to-end unsupervised frequent itemset learning using PyAerial.

    Args:
        dataset: DataFrame with categorical features
        max_length: Maximum itemset length (default: 3)
        similarity: Similarity threshold for itemset validation (default: 0.5)
        layer_dims: Hidden layer dimensions for autoencoder (default: auto)
        batch_size: Batch size for training (default: auto)
        epochs: Number of training epochs
        random_state: Random seed for reproducibility

    Returns:
        itemsets: List of extracted frequent itemsets
        stats: Statistics from pyaerial (itemset_count, average_support)
    """
    # Set seed for PyTorch (aerial uses PyTorch internally)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_state)

    feature_names = list(dataset.columns)

    print(f"Dataset shape: {dataset.shape}")
    print(f"Number of features: {len(feature_names)}")

    # Train the autoencoder
    print(f"\nTraining Aerial autoencoder...")
    print(f"  epochs={epochs}, layer_dims={layer_dims}, batch_size={batch_size}")
    trained_autoencoder = model.train(
        dataset,
        layer_dims=layer_dims,
        batch_size=batch_size,
        epochs=epochs
    )

    # Extract frequent itemsets using Aerial's native function
    print(f"\nExtracting frequent itemsets...")
    print(f"  max_length={max_length}")
    print(f"  similarity={similarity}")

    result = rule_extraction.generate_frequent_itemsets(
        trained_autoencoder,
        similarity=similarity,
        max_length=max_length
    )

    itemsets = result['itemsets']
    stats = result['statistics']

    print(f"\n{len(itemsets)} frequent itemsets found!")
    print(f"  PyAerial stats: avg_support={stats.get('average_support', 0):.4f}")

    return itemsets, stats


def get_dataset_parameters(dataset_name, dataset_size):
    """
    Get dataset-specific training parameters.

    Args:
        dataset_name: Name of the dataset
        dataset_size: Size category ('normal' or 'small')

    Returns:
        dict: Parameters (batch_size, layer_dims, epochs)
    """
    # Dataset-specific parameter overrides
    DATASET_PARAMS = {
        'breast_cancer': {
            'batch_size': 2,
            'layer_dims': [4],
            'epochs': 2
        },
        'congressional_voting': {
            'batch_size': 4,
            'layer_dims': [2],
            'epochs': 2
        }
    }

    # Default parameters by dataset size
    DEFAULT_PARAMS = {
        'normal': {
            'batch_size': 64,
            'layer_dims': [4],
            'epochs': 2
        },
        'small': {
            'batch_size': 2,
            'layer_dims': [4],
            'epochs': 10
        }
    }

    # Check for dataset-specific override first
    if dataset_name in DATASET_PARAMS:
        return DATASET_PARAMS[dataset_name]

    # Otherwise use default for the dataset size
    return DEFAULT_PARAMS.get(dataset_size, DEFAULT_PARAMS['normal'])


# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("PyAerial Frequent Itemset Mining Experiments")
    print("=" * 80)

    # Common parameters
    n_runs = 10
    max_length = 3  # Maximum itemset length
    similarity = 0.5  # Similarity threshold
    base_seed = 42  # Base seed for reproducibility

    # Dataset size: 'normal' or 'small'
    dataset_size = "small"

    # Generate seed sequence for all runs
    print(f"\nGenerating seed sequence from base_seed={base_seed}...")
    seed_sequence = generate_seed_sequence(base_seed, n_runs)
    print(f"Seeds for {n_runs} runs: {seed_sequence}")

    # Load datasets
    print("\nLoading datasets...")
    datasets = get_ucimlrepo_datasets(size=dataset_size)

    # Create output directory
    os.makedirs("out/frequent_itemsets", exist_ok=True)

    # Results storage - store all individual runs
    all_individual_results = []
    all_average_results = []
    all_dataset_params = []  # Track parameters used for each dataset

    # Run experiments for each dataset
    for dataset_info in datasets:
        dataset_name = dataset_info['name']
        dataset = dataset_info['data']

        print("\n" + "=" * 80)
        print(f"Dataset: {dataset_name}")
        print("=" * 80)
        print(f"Shape: {dataset.shape}")

        # Get dataset-specific parameters
        params = get_dataset_parameters(dataset_name, dataset_size)
        batch_size = params['batch_size']
        layer_dims = params['layer_dims']
        epochs = params['epochs']

        print(f"Using dataset-specific parameters:")
        print(f"  batch_size={batch_size}, layer_dims={layer_dims}, epochs={epochs}")

        # Store parameters for this dataset
        all_dataset_params.append({
            'dataset': dataset_name,
            'batch_size': batch_size,
            'layer_dims': str(layer_dims),
            'epochs': epochs,
            'max_length': max_length,
            'similarity': similarity
        })

        # Storage for this dataset's runs
        dataset_runs = []

        # Run N times
        for run_idx in range(n_runs):
            run_seed = seed_sequence[run_idx]
            print(f"\n--- Run {run_idx + 1}/{n_runs} (seed={run_seed}) ---")

            # Set global seed for this run
            set_seed(run_seed)

            # Track peak GPU memory
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            start_time = time.time()

            extracted_itemsets, stats = aerial_itemset_learning(
                dataset=dataset,
                max_length=max_length,
                similarity=similarity,
                layer_dims=layer_dims,
                epochs=epochs,
                random_state=run_seed,
                batch_size=batch_size
            )

            end_time = time.time()
            elapsed_time = end_time - start_time

            # Get peak GPU memory usage
            peak_gpu_memory_mb = 0.0
            if torch.cuda.is_available():
                peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 ** 2

            print(f"\nRun {run_idx + 1} completed in {elapsed_time:.2f} seconds")
            print(f"Extracted {len(extracted_itemsets)} frequent itemsets")
            if torch.cuda.is_available():
                print(f"Peak GPU Memory: {peak_gpu_memory_mb:.2f} MB")

            # Save itemsets to JSON (for CORELS classification later)
            itemsets_file = save_itemsets(
                itemsets=extracted_itemsets,
                stats=stats,
                dataset_name=dataset_name,
                method_name="aerial",
                seed=run_seed
            )
            print(f"Itemsets saved to {itemsets_file}")

            # Use metrics from pyaerial stats
            if len(extracted_itemsets) > 0:
                print(f"  Support: {stats.get('average_support', 0):.4f}")

                # Store results for this run
                result = {
                    'dataset': dataset_name,
                    'run': run_idx + 1,
                    'seed': run_seed,
                    'num_itemsets': stats.get('itemset_count', len(extracted_itemsets)),
                    'avg_support': stats.get('average_support', 0),
                    'execution_time': elapsed_time,
                    'peak_gpu_memory_mb': peak_gpu_memory_mb
                }
            else:
                print("  WARNING: No itemsets extracted!")
                result = {
                    'dataset': dataset_name,
                    'run': run_idx + 1,
                    'seed': run_seed,
                    'num_itemsets': 0,
                    'avg_support': 0.0,
                    'execution_time': elapsed_time,
                    'peak_gpu_memory_mb': peak_gpu_memory_mb
                }

            dataset_runs.append(result)
            all_individual_results.append(result)

        # Calculate averages across runs for this dataset
        runs_with_itemsets = [r for r in dataset_runs if r['num_itemsets'] > 0]
        n_runs_with_itemsets = len(runs_with_itemsets)

        if n_runs_with_itemsets > 0:
            avg_result = {
                'dataset': dataset_name,
                'num_itemsets': np.mean([r['num_itemsets'] for r in runs_with_itemsets]),
                'avg_support': np.mean([r['avg_support'] for r in runs_with_itemsets]),
                'execution_time': np.mean([r['execution_time'] for r in dataset_runs]),
                'peak_gpu_memory_mb': np.mean([r['peak_gpu_memory_mb'] for r in dataset_runs])
            }
        else:
            avg_result = {
                'dataset': dataset_name,
                'num_itemsets': 0,
                'avg_support': 0.0,
                'execution_time': np.mean([r['execution_time'] for r in dataset_runs]),
                'peak_gpu_memory_mb': np.mean([r['peak_gpu_memory_mb'] for r in dataset_runs])
            }
        all_average_results.append(avg_result)

        print(f"\n=== Average Results for {dataset_name} ({n_runs_with_itemsets}/{n_runs} runs with itemsets) ===")
        print(f"  Itemsets: {avg_result['num_itemsets']:.1f}")
        print(f"  Support: {avg_result['avg_support']:.4f}")
        print(f"  Avg Time: {avg_result['execution_time']:.2f}s")
        print(f"  Avg Peak GPU Memory: {avg_result['peak_gpu_memory_mb']:.2f} MB")

    # Save results to Excel with multiple sheets
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"out/aerial_itemsets_{timestamp}.xlsx"

    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        # Sheet 1: Individual run results
        individual_df = pd.DataFrame(all_individual_results)
        individual_df.to_excel(writer, sheet_name='Individual Results', index=False)

        # Sheet 2: Average results per dataset
        average_df = pd.DataFrame(all_average_results)
        average_df.to_excel(writer, sheet_name='Average Results', index=False)

        # Sheet 3: Dataset-specific parameters
        dataset_params_df = pd.DataFrame(all_dataset_params)
        dataset_params_df.to_excel(writer, sheet_name='Dataset Parameters', index=False)

        # Sheet 4: Common parameters
        common_params_df = pd.DataFrame([{
            'n_runs': n_runs,
            'base_seed': base_seed,
            'dataset_size': dataset_size
        }])
        common_params_df.to_excel(writer, sheet_name='Common Parameters', index=False)

        # Sheet 5: Seed Sequence (for reproducibility)
        seeds_df = pd.DataFrame({
            'run': list(range(1, n_runs + 1)),
            'seed': seed_sequence
        })
        seeds_df.to_excel(writer, sheet_name='Seeds', index=False)

    print("\n" + "=" * 80)
    print(f"Results saved to {output_filename}")
    print(f"  - Sheet 1: Individual Results (all {n_runs * len(datasets)} runs)")
    print(f"  - Sheet 2: Average Results (per dataset)")
    print(f"  - Sheet 3: Dataset Parameters (batch_size, layer_dims, epochs per dataset)")
    print(f"  - Sheet 4: Common Parameters (n_runs, base_seed, dataset_size)")
    print(f"  - Sheet 5: Seeds (seed sequence for reproducibility)")
    print("=" * 80)

    # Calculate and save FP-Growth calibration thresholds for itemsets
    print("\n" + "=" * 80)
    print("Calculating FP-Growth Calibration Thresholds for Itemsets")
    print("=" * 80)
    print("Finding minimum support thresholds for FP-Growth to cover 90% of Aerial itemsets...")

    dataset_names = [d['name'] for d in datasets]
    calculate_and_save_all_itemset_calibrations(
        dataset_names=dataset_names,
        reference_method="aerial",
        coverage_percentage=0.9
    )

    print("\nFP-Growth calibration thresholds saved.")
    print("=" * 80)