"""
ContextTab-based Rule Extraction

Adapts SAP-RPT-OSS for association rule mining using Aerial-style reconstruction logic.
For each feature, fits ContextTab to predict that feature from all others.
"""
import time
import os
import gc
import numpy as np
import pandas as pd
import torch
from datetime import datetime

from sap_rpt_oss import SAP_RPT_OSS_Classifier

from src.utils.data_prep import prepare_categorical_data, add_gaussian_noise
from src.utils.test_matrix import generate_test_matrix
from src.utils.rule_extraction import extract_rules_from_reconstruction
from src.utils import (
    get_ucimlrepo_datasets,
    calculate_rule_metrics,
    set_seed,
    generate_seed_sequence,
    save_rules,
    convert_metrics_to_stats,
)


def adapt_contexttab_for_reconstruction(context_table, query_matrix,
                                        feature_value_indices, noise_factor=0.5, bagging=8,
                                        query_batch_size=512):
    """Adapt ContextTab for reconstruction by fitting one model per feature.

    Args:
        context_table: The dataset to use as context
        query_matrix: Test matrix with antecedent patterns
        feature_value_indices: Feature range information
        noise_factor: Gaussian noise factor for context
        bagging: Number of bagging iterations for ContextTab
        query_batch_size: Batch size for processing queries (to avoid OOM)
    """

    print(f"    Adding Gaussian noise (factor={noise_factor}) to context...")
    noisy_context = add_gaussian_noise(context_table, noise_factor=noise_factor)

    n_queries = query_matrix.shape[0]
    n_features_total = query_matrix.shape[1]
    reconstruction_probs = np.zeros((n_queries, n_features_total))

    # Use entire table size as max_context_size for fair comparison
    max_context_size = context_table.shape[0]

    # Calculate number of batches
    n_batches = (n_queries + query_batch_size - 1) // query_batch_size

    print(f"    Reconstructing all features for {n_queries} queries...")
    print(f"    Fitting {len(feature_value_indices)} feature predictors...")
    print(f"    Using max_context_size={max_context_size} (full dataset)")
    if n_batches > 1:
        print(f"    Processing queries in {n_batches} batches of {query_batch_size}")

    for feat_idx, feat_info in enumerate(feature_value_indices):
        start_idx = feat_info['start']
        end_idx = feat_info['end']
        n_classes = end_idx - start_idx

        print(f"        Feature {feat_idx}: classes={n_classes}, range=[{start_idx}:{end_idx}]")

        # X = all OTHER features, y = current feature
        x_context = np.delete(noisy_context, range(start_idx, end_idx), axis=1)
        y_context_onehot = context_table[:, start_idx:end_idx]
        y_context = np.argmax(y_context_onehot, axis=1)

        contexttab_model = SAP_RPT_OSS_Classifier(
            max_context_size=max_context_size,
            bagging=bagging
        )

        contexttab_model.fit(x_context, y_context)
        x_query_full = np.delete(query_matrix, range(start_idx, end_idx), axis=1)

        if hasattr(contexttab_model, 'predict_proba'):
            # Process queries in batches to avoid OOM
            all_probs = []
            for batch_idx in range(n_batches):
                batch_start = batch_idx * query_batch_size
                batch_end = min((batch_idx + 1) * query_batch_size, n_queries)
                x_query_batch = x_query_full[batch_start:batch_end]

                batch_probs = contexttab_model.predict_proba(x_query_batch)
                all_probs.append(batch_probs)

                # Clear GPU cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            probs = np.vstack(all_probs)

            if probs.shape[1] != n_classes:
                print(f"        WARNING: predict_proba returned {probs.shape[1]} classes, expected {n_classes}")
                proper_probs = np.zeros((n_queries, n_classes))
                if hasattr(contexttab_model, 'classes_'):
                    for i, cls in enumerate(contexttab_model.classes_):
                        if cls < n_classes:
                            proper_probs[:, cls] = probs[:, i]
                else:
                    min_cols = min(probs.shape[1], n_classes)
                    proper_probs[:, :min_cols] = probs[:, :min_cols]
                reconstruction_probs[:, start_idx:end_idx] = proper_probs
            else:
                reconstruction_probs[:, start_idx:end_idx] = probs

        del contexttab_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return reconstruction_probs


def contexttab_rule_learning(dataset, max_antecedents=2, ant_similarity=0.5,
                             cons_similarity=0.8, bagging=8):
    """End-to-end rule learning using ContextTab."""

    encoded_data, classes_per_feature, feature_names, encoder = prepare_categorical_data(dataset)

    print(f"Dataset shape: {encoded_data.shape}")
    print(f"Number of features: {len(classes_per_feature)}")
    print(f"Classes per feature: {classes_per_feature}")

    test_matrix, test_descriptions, feature_value_indices = generate_test_matrix(
        n_features=len(classes_per_feature),
        classes_per_feature=classes_per_feature,
        max_antecedents=max_antecedents,
        use_zeros_for_unmarked=False
    )

    print(f"\nTest matrix shape: {test_matrix.shape}")
    print(f"Number of test vectors: {len(test_descriptions)}")

    print(f"\nUsing ContextTab for pattern reconstruction...")
    reconstruction_probs = adapt_contexttab_for_reconstruction(
        context_table=encoded_data,
        query_matrix=test_matrix,
        feature_value_indices=feature_value_indices,
        bagging=bagging
    )

    print(f"Reconstruction shape: {reconstruction_probs.shape}")

    print(f"\nExtracting rules...")
    rules = extract_rules_from_reconstruction(
        prob_matrix=reconstruction_probs,
        test_descriptions=test_descriptions,
        feature_value_indices=feature_value_indices,
        ant_similarity=ant_similarity,
        cons_similarity=cons_similarity,
        feature_names=feature_names,
        encoder=encoder
    )

    print(f"{len(rules)} rules found!")

    return rules, feature_names, dataset.values


if __name__ == "__main__":
    print("=" * 80)
    print("ContextTab (SAP-RPT-OSS) Rule Learning")
    print("=" * 80)

    # Parameters
    n_runs = 10
    max_antecedents = 2
    ant_similarity = 0.5
    cons_similarity = 0.8
    bagging = 8
    base_seed = 42

    # Generate seed sequence for all runs
    print(f"\nGenerating seed sequence from base_seed={base_seed}...")
    seed_sequence = generate_seed_sequence(base_seed, n_runs)
    print(f"Seeds for {n_runs} runs: {seed_sequence}")

    # Load datasets
    print("\nLoading datasets...")
    datasets = get_ucimlrepo_datasets(names=[
        'breast_cancer',
        'congressional_voting',
        'mushroom',
        'chess_king_rook_vs_king_pawn',
        'spambase',
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
                initial_gpu_memory = torch.cuda.memory_allocated() / 1024 ** 2

            start_time = time.time()

            extracted_rules, feature_names, original_data = contexttab_rule_learning(
                dataset=dataset,
                max_antecedents=max_antecedents,
                ant_similarity=ant_similarity,
                cons_similarity=cons_similarity,
                bagging=bagging
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

            # Calculate metrics and save rules
            if len(extracted_rules) > 0:
                rules_with_metrics, avg_metrics = calculate_rule_metrics(
                    rules=extracted_rules,
                    data=original_data,
                    feature_names=feature_names
                )

                # Convert to stats format
                stats = convert_metrics_to_stats(avg_metrics)

                # Save rules to JSON (for CBA/CORELS classification later)
                rules_file = save_rules(
                    rules=rules_with_metrics,
                    stats=stats,
                    dataset_name=dataset_name,
                    method_name="contexttab",
                    seed=run_seed
                )
                print(f"Rules saved to {rules_file}")

                print(f"  Support: {avg_metrics['support']:.4f}")
                print(f"  Confidence: {avg_metrics['confidence']:.4f}")
                print(f"  Zhang's Metric: {avg_metrics['zhangs_metric']:.4f}")
                print(f"  Interestingness: {avg_metrics['interestingness']:.4f}")
                print(f"  Rule coverage: {avg_metrics['rule_coverage']:.4f}")
                print(f"  Data coverage: {avg_metrics['data_coverage']:.4f}")

                # Store results for this run
                result = {
                    'dataset': dataset_name,
                    'run': run_idx + 1,
                    'seed': run_seed,
                    'num_rules': avg_metrics['num_rules'],
                    'avg_support': avg_metrics['support'],
                    'avg_confidence': avg_metrics['confidence'],
                    'avg_zhangs_metric': avg_metrics['zhangs_metric'],
                    'avg_interestingness': avg_metrics['interestingness'],
                    'avg_rule_coverage': avg_metrics['rule_coverage'],
                    'data_coverage': avg_metrics['data_coverage'],
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
    output_filename = f"out/contexttab_{timestamp}.xlsx"

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
            'bagging': bagging,
            'max_context_size': "full table"
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
