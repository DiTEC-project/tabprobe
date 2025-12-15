"""
TabPFN-based Rule Extraction Baseline

This is a BASELINE implementation/adaptation of the tabular foundation model TabPFN to do association
rule mining from tabular data using a Aerial-like logic (association by reconstruction/prediction success)

Adaptation Strategy:
- For each feature: Train TabPFN to predict that feature from all others
- Add Gaussian noise to context (matching PyAerial's denoising approach)
- Apply to test vectors (antecedent patterns) to get reconstructions
- Extract rules using Aerial's rule extraction logic

Limitations of this approach:
1. TabPFN is supervised (needs discrete labels), while rule learning is unsupervised
2. We must predict each feature independently, losing holistic pattern reconstruction
3. Much slower than Aerial (must retrain/fit for each feature)
4. Quality expected to be inferior to specialized rule learning models
5. CRITICAL: Antecedent validation is fundamentally broken (see below)

FUNDAMENTAL INCOMPATIBILITY:
PyAerial: "Given marked features A, reconstruct ALL features"
          - Antecedents A are in the query
          - Can validate A reconstructs well

TabPFN:   "Given all features EXCEPT i, predict feature i"
          - When i ∈ A (i is antecedent), we MUST remove it
          - Cannot validate antecedent reconstruction!

Result: Consequent prediction works (✓), antecedent validation fails (✗)
        Rules A -> C can be extracted, but lower quality (no antecedent filtering)

This baseline serves to justify the need for custom tabular foundation models
specifically designed for unsupervised rule discovery.
"""
import time
import os
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from tabpfn import TabPFNClassifier
from tabpfn_extensions.many_class import ManyClassClassifier

from src.utils.aerial import prepare_categorical_data
from src.utils.aerial.data_prep import add_gaussian_noise
from src.utils.aerial.test_matrix import generate_aerial_test_matrix
from src.utils.aerial.rule_extraction import extract_rules_from_reconstruction
from src.utils import (
    get_ucimlrepo_datasets,
    get_gene_expression_datasets,
    calculate_rule_metrics,
    set_seed,
    generate_seed_sequence,
)


def adapt_tabpfn_for_reconstruction(tabpfn_model, context_table, query_matrix,
                                    feature_value_indices, n_samples=None, noise_factor=0.5):
    """
    Adapt TabPFN for unsupervised rule learning following Aerial's ALL-AT-ONCE reconstruction logic.

    Aerial's Approach:
    - For a query with marked features A, pass it through the autoencoder ONCE
    - Get reconstruction probabilities for ALL features (both A and F/A) simultaneously
    - Check if A reconstructs well (antecedent validation)
    - Check which features in F/A reconstruct well (consequent extraction)

    PyAerial marks features A and looks at reconstruction of F/A. If reconstruction is successful
    based on both antecedent and consequent similarity thresholds for features C, then A -> C.

    Problem: TabPFN is supervised (needs y) and predicts ONE label at a time,
    while Aerial reconstructs ALL features at once.

    Solution:
    - Train one model per feature to reconstruct that feature from all other features
    - For each query, predict ALL features to simulate "all-at-once" reconstruction
    - This mimics Aerial's behavior: given marked features A, what are probabilities for all features?

    Args:
        tabpfn_model: Pretrained TabPFN model
        context_table: The dataset to use as context (n_rows, n_features)
        query_matrix: Test matrix with antecedent patterns (n_queries, n_features)
                     Each row has marked features (A) set to 1 in their class position,
                     and unmarked features (F/A) set to uniform probabilities
        feature_value_indices: Feature range information
        n_samples: Number of context samples to use (None = all)
        noise_factor: Gaussian noise standard deviation to add to context (default=0.5, matching PyAerial)

    Returns:
        reconstruction_probs: Probability matrix (n_queries, n_features)
                             For each query, contains predicted probabilities for ALL features,
                             allowing us to check both antecedent reconstruction and consequent prediction
    """

    # Limit context size if needed (TabPFN has context length limits)
    if n_samples and len(context_table) > n_samples:
        context_table = context_table[:n_samples]

    # Add Gaussian noise to context table, matching PyAerial's training approach
    # This makes TabPFN see values between 0 and 1 (not just one-hot 0s and 1s),
    # which makes query patterns with equal probabilities [0.5, 0.5] more natural
    print(f"    Adding Gaussian noise (factor={noise_factor}) to context...")
    noisy_context = add_gaussian_noise(context_table, noise_factor=noise_factor)

    n_queries = query_matrix.shape[0]
    n_features_total = query_matrix.shape[1]

    # Initialize reconstruction matrix - will contain ALL feature predictions for ALL queries
    reconstruction_probs = np.zeros((n_queries, n_features_total))

    # For each feature, train TabPFN to predict it and reconstruct for ALL queries
    # This simulates Aerial's all-at-once reconstruction behavior
    print(f"    Reconstructing all features for {n_queries} queries...")
    print(f"    Training {len(feature_value_indices)} feature predictors...")

    for feat_idx, feat_info in enumerate(feature_value_indices):
        start_idx = feat_info['start']
        end_idx = feat_info['end']
        n_classes = end_idx - start_idx

        print(f"        Feature {feat_idx}: classes={n_classes}, range=[{start_idx}:{end_idx}]")

        # Prepare context: X = all OTHER features (excluding target), y = current feature
        # This properly teaches TabPFN: given all other features, predict this feature
        # This avoids data leakage and matches autoencoder reconstruction behavior
        # Use noisy_context to match PyAerial's training with noise
        x_context = np.delete(noisy_context, range(start_idx, end_idx), axis=1)
        y_context_onehot = context_table[:, start_idx:end_idx]  # Target is clean (no noise)
        y_context = np.argmax(y_context_onehot, axis=1)

        # If feature has more than 10 classes, wrap TabPFN with ManyClassClassifier
        if n_classes > 10:
            print(f"        Using ManyClassClassifier wrapper (classes={n_classes} > 10)")
            model_to_use = ManyClassClassifier(
                estimator=tabpfn_model,
                alphabet_size=10,
                random_state=tabpfn_model.random_state if hasattr(tabpfn_model, 'random_state') else None
            )
        else:
            model_to_use = tabpfn_model

        # Fit model to predict this feature from all OTHER features
        model_to_use.fit(x_context, y_context)

        # Prepare query matrix: remove the target feature columns from query
        # We must match the training input shape (context without target feature)
        # However, we want to keep OTHER marked features (antecedents) in the query
        x_query = np.delete(query_matrix, range(start_idx, end_idx), axis=1)

        if hasattr(model_to_use, 'predict_proba'):
            probs = model_to_use.predict_proba(x_query)
            # probs shape: (n_queries, n_classes_seen_in_training)
            # May not match n_classes if some classes weren't in context

            if probs.shape[1] != n_classes:
                print(f"        WARNING: predict_proba returned {probs.shape[1]} classes, expected {n_classes}")
                # Create proper sized array and fill with available probabilities
                proper_probs = np.zeros((n_queries, n_classes))
                # Get the classes the model actually learned
                if hasattr(model_to_use, 'classes_'):
                    for i, cls in enumerate(model_to_use.classes_):
                        if cls < n_classes:
                            proper_probs[:, cls] = probs[:, i]
                else:
                    # Assume sequential classes starting from 0
                    min_cols = min(probs.shape[1], n_classes)
                    proper_probs[:, :min_cols] = probs[:, :min_cols]
                reconstruction_probs[:, start_idx:end_idx] = proper_probs
            else:
                reconstruction_probs[:, start_idx:end_idx] = probs

    return reconstruction_probs


def tabpfn_rule_learning(dataset, max_antecedents=2, context_samples=100,
                         ant_similarity=0.5, cons_similarity=0.8, random_state=42):
    """
    End-to-end unsupervised rule learning using TabPFN.

    Args:
        dataset: DataFrame with categorical features
        max_antecedents: Maximum antecedents per rule
        context_samples: Number of samples to use as context
        ant_similarity: Antecedent threshold
        cons_similarity: Consequent threshold
        random_state: Random seed for TabPFN model (for reproducibility)

    Returns:
        rules: List of extracted association rules
    """

    # Prepare data
    encoded_data, classes_per_feature, feature_names, encoder = prepare_categorical_data(dataset)

    print(f"Dataset shape: {encoded_data.shape}")
    print(f"Number of features: {len(classes_per_feature)}")
    print(f"Classes per feature: {classes_per_feature}")

    # Generate test matrix (query patterns)
    # Use equal probabilities for unmarked features (NOT zeros)
    # Since we add noise to the context, TabPFN will see values between 0 and 1,
    # making [0.33, 0.33, 0.33] patterns more natural and consistent with the training distribution
    test_matrix, test_descriptions, feature_value_indices = generate_aerial_test_matrix(
        n_features=len(classes_per_feature),
        classes_per_feature=classes_per_feature,
        max_antecedents=max_antecedents,
        use_zeros_for_unmarked=False  # Equal probabilities work better with noisy context
    )

    print(f"\nTest matrix shape: {test_matrix.shape}")
    print(f"Number of test vectors: {len(test_descriptions)}")

    # Initialize TabPFN with random_state for reproducibility
    # IMPORTANT: Parameters aligned with TabICL for fair comparison:
    # - n_estimators=32 (matches TabICL default, was 8)
    # - average_before_softmax=True (matches TabICL's average_logits=True)
    # - inference_precision='auto' (closest to TabICL's use_amp=True)
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

    # Extract rules using PyAerial logic
    print(f"\nExtracting rules...")
    rules = extract_rules_from_reconstruction(
        prob_matrix=reconstruction_probs,
        test_descriptions=test_descriptions,
        feature_value_indices=feature_value_indices,
        ant_similarity=ant_similarity,
        cons_similarity=cons_similarity,
        feature_names=feature_names,
        encoder=encoder  # Pass encoder to map class indices to actual values
    )

    print(f"{len(rules)} rules found!")

    return rules, feature_names, dataset.values


# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("TabPFN Baseline for Rule Learning")
    print("=" * 80)

    # Parameters
    n_runs = 10
    max_antecedents = 2
    ant_similarity = 0.5
    cons_similarity = 0.8
    context_samples = None
    base_seed = 42  # Base seed for reproducibility

    # Generate seed sequence for all runs
    print(f"\nGenerating seed sequence from base_seed={base_seed}...")
    seed_sequence = generate_seed_sequence(base_seed, n_runs)
    print(f"Seeds for {n_runs} runs: {seed_sequence}")

    # Load datasets
    print("\nLoading datasets...")
    datasets = get_ucimlrepo_datasets(size="small")

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

            extracted_rules, feature_names, original_data = tabpfn_rule_learning(
                dataset=dataset,
                max_antecedents=max_antecedents,
                context_samples=context_samples if context_samples else dataset.shape[0],
                ant_similarity=ant_similarity,
                cons_similarity=cons_similarity,
                random_state=run_seed  # Pass seed to TabPFN for reproducibility
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

            # Calculate metrics
            if len(extracted_rules) > 0:
                rules_with_metrics, avg_metrics = calculate_rule_metrics(
                    rules=extracted_rules,
                    data=original_data,
                    feature_names=feature_names
                )

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
    output_filename = f"out/tabpfn_{timestamp}.xlsx"

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
            'cons_similarity': cons_similarity
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
