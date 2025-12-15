"""
PyAerial Rule Learning Experiments

This module runs association rule mining experiments using PyAerial.

PyAerial learns to reconstruct categorical data through a neural autoencoder,
then extracts rules by testing which feature combinations lead to successful
reconstruction of other features.
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
)


def aerial_rule_learning(dataset, max_antecedents=2, ant_similarity=0.5, batch_size=None,
                         cons_similarity=0.8, layer_dims=None, epochs=2, random_state=42):
    """
    End-to-end unsupervised rule learning using PyAerial.

    Args:
        dataset: DataFrame with categorical features
        max_antecedents: Maximum antecedents per rule
        ant_similarity: Antecedent similarity threshold
        cons_similarity: Consequent similarity threshold
        layer_dims: Hidden layer dimensions for autoencoder (default: auto)
        epochs: Number of training epochs
        random_state: Random seed for reproducibility

    Returns:
        rules: List of extracted association rules
        stats: Statistics from pyaerial (average_support, average_confidence, etc.)
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
    print(f"  epochs={epochs}, layer_dims={layer_dims}")
    trained_autoencoder = model.train(
        dataset,
        layer_dims=layer_dims,
        batch_size=batch_size,
        epochs=epochs
    )

    # Extract rules
    print(f"\nExtracting rules...")
    print(f"  max_antecedents={max_antecedents}")
    print(f"  ant_similarity={ant_similarity}")
    print(f"  cons_similarity={cons_similarity}")

    result = rule_extraction.generate_rules(
        trained_autoencoder,
        ant_similarity=ant_similarity,
        cons_similarity=cons_similarity,
        max_antecedents=max_antecedents,
        quality_metrics=['support', 'confidence', 'zhangs_metric', 'interestingness']
    )

    rules = result['rules']
    stats = result['statistics']

    print(f"\n{len(rules)} rules found!")
    print(f"  PyAerial stats: avg_support={stats.get('average_support', 0):.4f}, "
          f"avg_confidence={stats.get('average_confidence', 0):.4f}")

    return rules, stats


# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("PyAerial Rule Learning Experiments")
    print("=" * 80)

    # Parameters
    n_runs = 10
    max_antecedents = 2
    ant_similarity = 0.5
    cons_similarity = 0.8
    epochs = 3
    layer_dims = [2]
    batch_size = 2
    base_seed = 42  # Base seed for reproducibility

    # Parameter settings
    # "normal_size_tables": 64 batch_size, 4 layer_dims, 2 epochs --> all except breast cancer and congressional voting
    #   breast_cancer: 2 batch_size, 2 layer_dims, 3 epochs
    #   congressional_voting: 4 batch_size, 2 layer_dims, 2 epochs
    # "small_size_tables": 64 batch_size, 4 layer_dims, 2 epochs

    # Generate seed sequence for all runs
    print(f"\nGenerating seed sequence from base_seed={base_seed}...")
    seed_sequence = generate_seed_sequence(base_seed, n_runs)
    print(f"Seeds for {n_runs} runs: {seed_sequence}")

    # Load datasets
    print("\nLoading datasets...")
    datasets = get_ucimlrepo_datasets(size="normal", names=[
        'breast_cancer',
    ])

    # Create output directory
    os.makedirs("out", exist_ok=True)

    # Results storage - store all individual runs
    all_individual_results = []
    all_average_results = []

    # Run experiments for each dataset
    for dataset_info in datasets:
        dataset_name = dataset_info['name']
        dataset = dataset_info['data']

        print("\n" + "=" * 80)
        print(f"Dataset: {dataset_name}")
        print("=" * 80)
        print(f"Shape: {dataset.shape}")

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

            extracted_rules, stats = aerial_rule_learning(
                dataset=dataset,
                max_antecedents=max_antecedents,
                ant_similarity=ant_similarity,
                cons_similarity=cons_similarity,
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
            print(f"Extracted {len(extracted_rules)} rules")
            if torch.cuda.is_available():
                print(f"Peak GPU Memory: {peak_gpu_memory_mb:.2f} MB")

            # Use metrics from pyaerial stats
            if len(extracted_rules) > 0:
                print(f"  Support: {stats.get('average_support', 0):.4f}")
                print(f"  Confidence: {stats.get('average_confidence', 0):.4f}")
                print(f"  Zhang's Metric: {stats.get('average_zhangs_metric', 0):.4f}")
                print(f"  Interestingness: {stats.get('average_interestingness', 0):.4f}")
                print(f"  Rule coverage: {stats.get('average_coverage', 0):.4f}")
                print(f"  Data coverage: {stats.get('data_coverage', 0):.4f}")

                # Store results for this run
                result = {
                    'dataset': dataset_name,
                    'run': run_idx + 1,
                    'seed': run_seed,
                    'num_rules': stats.get('rule_count', len(extracted_rules)),
                    'avg_support': stats.get('average_support', 0),
                    'avg_confidence': stats.get('average_confidence', 0),
                    'avg_zhangs_metric': stats.get('average_zhangs_metric', 0),
                    'avg_interestingness': stats.get('average_interestingness', 0),
                    'avg_rule_coverage': stats.get('average_coverage', 0),
                    'data_coverage': stats.get('data_coverage', 0),
                    'execution_time': elapsed_time,
                    'peak_gpu_memory_mb': peak_gpu_memory_mb
                }
            else:
                print("  WARNING: No rules extracted!")
                result = {
                    'dataset': dataset_name,
                    'run': run_idx + 1,
                    'seed': run_seed,
                    'num_rules': 0,
                    'avg_support': 0.0,
                    'avg_confidence': 0.0,
                    'avg_zhangs_metric': 0.0,
                    'avg_interestingness': 0.0,
                    'avg_rule_coverage': 0.0,
                    'data_coverage': 0.0,
                    'execution_time': elapsed_time,
                    'peak_gpu_memory_mb': peak_gpu_memory_mb
                }

            dataset_runs.append(result)
            all_individual_results.append(result)

        # Calculate averages across runs for this dataset
        # Only average rule metrics over runs that produced rules (>0 rules)
        runs_with_rules = [r for r in dataset_runs if r['num_rules'] > 0]
        n_runs_with_rules = len(runs_with_rules)

        if n_runs_with_rules > 0:
            avg_result = {
                'dataset': dataset_name,
                'num_rules': np.mean([r['num_rules'] for r in runs_with_rules]),
                'avg_support': np.mean([r['avg_support'] for r in runs_with_rules]),
                'avg_confidence': np.mean([r['avg_confidence'] for r in runs_with_rules]),
                'avg_zhangs_metric': np.mean([r['avg_zhangs_metric'] for r in runs_with_rules]),
                'avg_interestingness': np.mean([r['avg_interestingness'] for r in runs_with_rules]),
                'avg_rule_coverage': np.mean([r['avg_rule_coverage'] for r in runs_with_rules]),
                'data_coverage': np.mean([r['data_coverage'] for r in runs_with_rules]),
                'execution_time': np.mean([r['execution_time'] for r in dataset_runs]),
                'peak_gpu_memory_mb': np.mean([r['peak_gpu_memory_mb'] for r in dataset_runs])
            }
        else:
            avg_result = {
                'dataset': dataset_name,
                'num_rules': 0,
                'avg_support': 0.0,
                'avg_confidence': 0.0,
                'avg_zhangs_metric': 0.0,
                'avg_interestingness': 0.0,
                'avg_rule_coverage': 0.0,
                'data_coverage': 0.0,
                'execution_time': np.mean([r['execution_time'] for r in dataset_runs]),
                'peak_gpu_memory_mb': np.mean([r['peak_gpu_memory_mb'] for r in dataset_runs])
            }
        all_average_results.append(avg_result)

        print(f"\n=== Average Results for {dataset_name} ({n_runs_with_rules}/{n_runs} runs with rules) ===")
        print(f"  Rules: {avg_result['num_rules']:.1f}")
        print(f"  Support: {avg_result['avg_support']:.4f}")
        print(f"  Confidence: {avg_result['avg_confidence']:.4f}")
        print(f"  Zhang's Metric: {avg_result['avg_zhangs_metric']:.4f}")
        print(f"  Interestingness: {avg_result['avg_interestingness']:.4f}")
        print(f"  Rule Coverage: {avg_result['avg_rule_coverage']:.4f}")
        print(f"  Data Coverage: {avg_result['data_coverage']:.4f}")
        print(f"  Avg Time: {avg_result['execution_time']:.2f}s")
        print(f"  Avg Peak GPU Memory: {avg_result['peak_gpu_memory_mb']:.2f} MB")

    # Save results to Excel with multiple sheets
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"out/aerial_{timestamp}.xlsx"

    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        # Sheet 1: Individual run results
        individual_df = pd.DataFrame(all_individual_results)
        individual_df.to_excel(writer, sheet_name='Individual Results', index=False)

        # Sheet 2: Average results per dataset
        average_df = pd.DataFrame(all_average_results)
        average_df.to_excel(writer, sheet_name='Average Results', index=False)

        # Sheet 3: Parameters
        params_df = pd.DataFrame([{
            'n_runs': n_runs,
            'base_seed': base_seed,
            'max_antecedents': max_antecedents,
            'ant_similarity': ant_similarity,
            'cons_similarity': cons_similarity,
            'epochs': epochs,
            'layer_dims': str(layer_dims)
        }])
        params_df.to_excel(writer, sheet_name='Parameters', index=False)

        # Sheet 4: Seed Sequence (for reproducibility)
        seeds_df = pd.DataFrame({
            'run': list(range(1, n_runs + 1)),
            'seed': seed_sequence
        })
        seeds_df.to_excel(writer, sheet_name='Seeds', index=False)

    print("\n" + "=" * 80)
    print(f"Results saved to {output_filename}")
    print(f"  - Sheet 1: Individual Results (all {n_runs * len(datasets)} runs)")
    print(f"  - Sheet 2: Average Results (per dataset)")
    print(f"  - Sheet 3: Parameters (including base_seed={base_seed})")
    print(f"  - Sheet 4: Seeds (seed sequence for reproducibility)")
    print("=" * 80)
