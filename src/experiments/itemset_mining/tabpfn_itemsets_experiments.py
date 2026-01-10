"""
TabPFN-based Frequent Itemset Mining

Adapts TabPFN for frequent itemset discovery using reconstruction-based approach.
Similar to tabpfn_experiments.py but extracts itemsets instead of association rules.
"""
import time
import os
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from tabpfn import TabPFNClassifier
from tabpfn_extensions.many_class import ManyClassClassifier

from src.utils.data_prep import prepare_categorical_data, add_gaussian_noise
from src.utils.test_matrix import generate_test_matrix
from src.utils.rule_extraction import extract_frequent_itemsets_from_reconstruction
from src.utils import (
    get_ucimlrepo_datasets,
    set_seed,
    generate_seed_sequence,
    save_itemsets,
)


def adapt_tabpfn_for_reconstruction(tabpfn_model, context_table, query_matrix,
                                    feature_value_indices, n_samples=None, noise_factor=0.5):
    """
    Adapt TabPFN for frequent itemset mining using reconstruction logic.

    For each feature, trains TabPFN to predict that feature from all others,
    then reconstructs ALL features for each query to get itemset patterns.
    """
    # Limit context size if needed
    if n_samples and len(context_table) > n_samples:
        context_table = context_table[:n_samples]

    # Add Gaussian noise to context
    print(f"    Adding Gaussian noise (factor={noise_factor}) to context...")
    noisy_context = add_gaussian_noise(context_table, noise_factor=noise_factor)

    n_queries = query_matrix.shape[0]
    n_features_total = query_matrix.shape[1]

    # Initialize reconstruction matrix
    reconstruction_probs = np.zeros((n_queries, n_features_total))

    print(f"    Reconstructing all features for {n_queries} queries...")
    print(f"    Training {len(feature_value_indices)} feature predictors...")

    for feat_idx, feat_info in enumerate(feature_value_indices):
        start_idx = feat_info['start']
        end_idx = feat_info['end']
        n_classes = end_idx - start_idx

        print(f"        Feature {feat_idx}: classes={n_classes}, range=[{start_idx}:{end_idx}]")

        # Prepare context: X = all OTHER features, y = current feature
        x_context = np.delete(noisy_context, range(start_idx, end_idx), axis=1)
        y_context_onehot = context_table[:, start_idx:end_idx]
        y_context = np.argmax(y_context_onehot, axis=1)

        # Use ManyClassClassifier for features with >10 classes
        if n_classes > 10:
            print(f"        Using ManyClassClassifier wrapper (classes={n_classes} > 10)")
            model_to_use = ManyClassClassifier(
                estimator=tabpfn_model,
                alphabet_size=10,
                random_state=tabpfn_model.random_state if hasattr(tabpfn_model, 'random_state') else None
            )
        else:
            model_to_use = tabpfn_model

        # Fit model
        model_to_use.fit(x_context, y_context)

        # Prepare query matrix
        x_query = np.delete(query_matrix, range(start_idx, end_idx), axis=1)

        if hasattr(model_to_use, 'predict_proba'):
            probs = model_to_use.predict_proba(x_query)

            if probs.shape[1] != n_classes:
                print(f"        WARNING: predict_proba returned {probs.shape[1]} classes, expected {n_classes}")
                proper_probs = np.zeros((n_queries, n_classes))
                if hasattr(model_to_use, 'classes_'):
                    for i, cls in enumerate(model_to_use.classes_):
                        if cls < n_classes:
                            proper_probs[:, cls] = probs[:, i]
                else:
                    min_cols = min(probs.shape[1], n_classes)
                    proper_probs[:, :min_cols] = probs[:, :min_cols]
                reconstruction_probs[:, start_idx:end_idx] = proper_probs
            else:
                reconstruction_probs[:, start_idx:end_idx] = probs

    return reconstruction_probs


def tabpfn_itemset_learning(dataset, max_itemset_length=3, context_samples=100,
                             similarity=0.5, random_state=42):
    """
    End-to-end frequent itemset mining using TabPFN.

    Args:
        dataset: DataFrame with categorical features
        max_itemset_length: Maximum itemset length (default: 3)
        context_samples: Number of samples to use as context
        similarity: Similarity threshold for itemset validation (default: 0.5)
        random_state: Random seed for TabPFN model

    Returns:
        itemsets: List of extracted frequent itemsets with support
        stats: Statistics dictionary
    """
    # Prepare data
    encoded_data, classes_per_feature, feature_names, encoder = prepare_categorical_data(dataset)

    print(f"Dataset shape: {encoded_data.shape}")
    print(f"Number of features: {len(classes_per_feature)}")
    print(f"Classes per feature: {classes_per_feature}")

    # Generate test matrix (itemset patterns)
    test_matrix, test_descriptions, feature_value_indices = generate_test_matrix(
        n_features=len(classes_per_feature),
        classes_per_feature=classes_per_feature,
        max_antecedents=max_itemset_length,
        use_zeros_for_unmarked=False
    )

    print(f"\nTest matrix shape: {test_matrix.shape}")
    print(f"Number of test vectors: {len(test_descriptions)}")

    # Initialize TabPFN
    tabpfn_model = TabPFNClassifier(
        n_estimators=8,
        random_state=random_state,
        average_before_softmax=True,
        inference_precision='auto'
    )

    # Adapt TabPFN for reconstruction
    print(f"\nUsing TabPFN for pattern reconstruction...")
    reconstruction_probs = adapt_tabpfn_for_reconstruction(
        tabpfn_model=tabpfn_model,
        context_table=encoded_data,
        query_matrix=test_matrix,
        feature_value_indices=feature_value_indices,
        n_samples=context_samples
    )

    print(f"Reconstruction shape: {reconstruction_probs.shape}")

    # Extract frequent itemsets
    print(f"\nExtracting frequent itemsets...")
    result = extract_frequent_itemsets_from_reconstruction(
        prob_matrix=reconstruction_probs,
        test_descriptions=test_descriptions,
        feature_value_indices=feature_value_indices,
        data=dataset,  # Original data for support calculation
        similarity=similarity,
        feature_names=feature_names,
        encoder=encoder
    )

    itemsets = result['itemsets']
    stats = result['statistics']

    print(f"{len(itemsets)} frequent itemsets found!")

    return itemsets, stats, feature_names, dataset.values


# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("TabPFN Frequent Itemset Mining")
    print("=" * 80)

    # Parameters
    n_runs = 10
    max_itemset_length = 3
    similarity = 0.5
    context_samples = None  # Use all samples
    base_seed = 42

    # Generate seed sequence
    print(f"\nGenerating seed sequence from base_seed={base_seed}...")
    seed_sequence = generate_seed_sequence(base_seed, n_runs)
    print(f"Seeds for {n_runs} runs: {seed_sequence}")

    # Load datasets
    print("\nLoading datasets...")
    datasets = get_ucimlrepo_datasets(size="normal")

    # Create output directory
    os.makedirs("out/frequent_itemsets", exist_ok=True)

    # Results storage
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

            # Set global seed
            set_seed(run_seed)

            # Track GPU memory
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                initial_gpu_memory = torch.cuda.memory_allocated() / 1024 ** 2

            start_time = time.time()

            extracted_itemsets, stats, feature_names, original_data = tabpfn_itemset_learning(
                dataset=dataset,
                max_itemset_length=max_itemset_length,
                context_samples=context_samples if context_samples else dataset.shape[0],
                similarity=similarity,
                random_state=run_seed
            )

            end_time = time.time()
            elapsed_time = end_time - start_time

            # Get peak GPU memory
            peak_gpu_memory_mb = 0.0
            if torch.cuda.is_available():
                peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 ** 2

            print(f"\nRun {run_idx + 1} completed in {elapsed_time:.2f} seconds")
            print(f"Extracted {len(extracted_itemsets)} frequent itemsets")
            if torch.cuda.is_available():
                print(f"Peak GPU Memory: {peak_gpu_memory_mb:.2f} MB")

            # Save itemsets
            if len(extracted_itemsets) > 0:
                itemsets_file = save_itemsets(
                    itemsets=extracted_itemsets,
                    stats=stats,
                    dataset_name=dataset_name,
                    method_name="tabpfn",
                    seed=run_seed
                )
                print(f"Itemsets saved to {itemsets_file}")

                print(f"  Itemset count: {stats.get('itemset_count', len(extracted_itemsets))}")
                print(f"  Average support: {stats.get('average_support', 0):.4f}")

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

        # Calculate averages
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

    # Save results to Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"out/tabpfn_itemsets_{timestamp}.xlsx"

    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        # Sheet 1: Individual results
        pd.DataFrame(all_individual_results).to_excel(writer, sheet_name='Individual Results', index=False)

        # Sheet 2: Average results
        pd.DataFrame(all_average_results).to_excel(writer, sheet_name='Average Results', index=False)

        # Sheet 3: Parameters
        params_df = pd.DataFrame([{
            'n_runs': n_runs,
            'base_seed': base_seed,
            'max_itemset_length': max_itemset_length,
            'similarity': similarity,
            'context_samples': context_samples if context_samples else 'all'
        }])
        params_df.to_excel(writer, sheet_name='Parameters', index=False)

        # Sheet 4: Seed sequence
        seeds_df = pd.DataFrame({'run': range(1, n_runs + 1), 'seed': seed_sequence})
        seeds_df.to_excel(writer, sheet_name='Seed Sequence', index=False)

    print("\n" + "=" * 80)
    print(f"Results saved to {output_filename}")
    print("=" * 80)