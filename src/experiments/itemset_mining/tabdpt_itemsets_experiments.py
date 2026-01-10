"""
TabDPT-based Frequent Itemset Mining

Adapts TabDPT for frequent itemset discovery using reconstruction-based approach.
"""
import time
import os
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from tabdpt import TabDPTClassifier

from src.utils.data_prep import prepare_categorical_data, add_gaussian_noise
from src.utils.test_matrix import generate_test_matrix
from src.utils.rule_extraction import extract_frequent_itemsets_from_reconstruction
from src.utils import (
    get_ucimlrepo_datasets,
    set_seed,
    generate_seed_sequence,
    save_itemsets,
)


def adapt_tabdpt_for_reconstruction(context_table, query_matrix, feature_value_indices, n_samples=None, noise_factor=0.5):
    """Adapt TabDPT for itemset mining using reconstruction logic."""
    if n_samples and len(context_table) > n_samples:
        context_table = context_table[:n_samples]

    print(f"    Adding Gaussian noise (factor={noise_factor}) to context...")
    noisy_context = add_gaussian_noise(context_table, noise_factor=noise_factor)

    n_queries = query_matrix.shape[0]
    n_features_total = query_matrix.shape[1]
    reconstruction_probs = np.zeros((n_queries, n_features_total))

    print(f"    Reconstructing all features for {n_queries} queries...")

    for feat_idx, feat_info in enumerate(feature_value_indices):
        start_idx = feat_info['start']
        end_idx = feat_info['end']
        n_classes = end_idx - start_idx

        print(f"        Feature {feat_idx}: classes={n_classes}")

        x_context = np.delete(noisy_context, range(start_idx, end_idx), axis=1)
        y_context = np.argmax(context_table[:, start_idx:end_idx], axis=1)

        tabdpt_model = TabDPTClassifier()
        tabdpt_model.fit(x_context, y_context)

        x_query = np.delete(query_matrix, range(start_idx, end_idx), axis=1)
        
        # Process queries in batches for memory efficiency
        batch_size = 512
        all_probs = []
        for batch_start in range(0, len(x_query), batch_size):
            batch_end = min(batch_start + batch_size, len(x_query))
            batch_probs = tabdpt_model.predict_proba(x_query[batch_start:batch_end])
            all_probs.append(batch_probs)
            
        probs = np.vstack(all_probs)

        if probs.shape[1] != n_classes:
            proper_probs = np.zeros((n_queries, n_classes))
            if hasattr(tabdpt_model, 'classes_'):
                for i, cls in enumerate(tabdpt_model.classes_):
                    if cls < n_classes:
                        proper_probs[:, cls] = probs[:, i]
            reconstruction_probs[:, start_idx:end_idx] = proper_probs
        else:
            reconstruction_probs[:, start_idx:end_idx] = probs

        # Clear GPU cache after each feature
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return reconstruction_probs


def tabdpt_itemset_learning(dataset, max_itemset_length=2, context_samples=100, similarity=0.5):
    """Frequent itemset mining using TabDPT."""
    encoded_data, classes_per_feature, feature_names, encoder = prepare_categorical_data(dataset)

    test_matrix, test_descriptions, feature_value_indices = generate_test_matrix(
        n_features=len(classes_per_feature),
        classes_per_feature=classes_per_feature,
        max_antecedents=max_itemset_length,
        use_zeros_for_unmarked=False
    )

    reconstruction_probs = adapt_tabdpt_for_reconstruction(
        context_table=encoded_data,
        query_matrix=test_matrix,
        feature_value_indices=feature_value_indices,
        n_samples=context_samples
    )

    result = extract_frequent_itemsets_from_reconstruction(
        prob_matrix=reconstruction_probs,
        test_descriptions=test_descriptions,
        feature_value_indices=feature_value_indices,
        data=dataset,
        similarity=similarity,
        feature_names=feature_names,
        encoder=encoder
    )

    return result['itemsets'], result['statistics'], feature_names, dataset.values


if __name__ == "__main__":
    print("=" * 80)
    print("TabDPT Frequent Itemset Mining")
    print("=" * 80)

    n_runs = 10
    max_itemset_length = 3
    similarity = 0.5
    context_samples = None
    base_seed = 42

    seed_sequence = generate_seed_sequence(base_seed, n_runs)
    datasets = get_ucimlrepo_datasets(size="small")
    os.makedirs("out/frequent_itemsets", exist_ok=True)

    all_individual_results = []
    all_average_results = []

    for dataset_info in datasets:
        dataset_name = dataset_info['name']
        dataset = dataset_info['data']
        dataset_runs = []

        for run_idx in range(n_runs):
            run_seed = seed_sequence[run_idx]
            set_seed(run_seed)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            start_time = time.time()

            extracted_itemsets, stats, _, _ = tabdpt_itemset_learning(
                dataset=dataset,
                max_itemset_length=max_itemset_length,
                context_samples=context_samples if context_samples else dataset.shape[0],
                similarity=similarity
            )

            elapsed_time = time.time() - start_time
            peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0

            if len(extracted_itemsets) > 0:
                save_itemsets(extracted_itemsets, stats, dataset_name, "tabdpt", run_seed)
                result = {
                    'dataset': dataset_name,
                    'run': run_idx + 1,
                    'seed': run_seed,
                    'num_itemsets': len(extracted_itemsets),
                    'avg_support': stats.get('average_support', 0),
                    'execution_time': elapsed_time,
                    'peak_gpu_memory_mb': peak_gpu_memory_mb
                }
            else:
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

        runs_with_itemsets = [r for r in dataset_runs if r['num_itemsets'] > 0]
        if len(runs_with_itemsets) > 0:
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with pd.ExcelWriter(f"out/tabdpt_itemsets_{timestamp}.xlsx", engine='openpyxl') as writer:
        pd.DataFrame(all_individual_results).to_excel(writer, sheet_name='Individual Results', index=False)
        pd.DataFrame(all_average_results).to_excel(writer, sheet_name='Average Results', index=False)
        pd.DataFrame([{'n_runs': n_runs, 'base_seed': base_seed}]).to_excel(writer, sheet_name='Parameters', index=False)
        pd.DataFrame({'run': range(1, n_runs + 1), 'seed': seed_sequence}).to_excel(writer, sheet_name='Seed Sequence', index=False)
