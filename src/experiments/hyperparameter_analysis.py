"""
Hyperparameter Analysis for TabPFN, TabICL, and TabDPT

Analyzes the effect of ant_similarity and cons_similarity thresholds on rule quality
across all three tabular foundation models.
"""
import time
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# Redirect stderr to suppress deprecation warnings from TabDPT
@contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output."""
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


from tabpfn import TabPFNClassifier
from tabicl import TabICLClassifier

from src.utils.data_prep import prepare_categorical_data
from src.utils.test_matrix import generate_test_matrix
from src.utils.rule_extraction import extract_rules_from_reconstruction
# Import from rule_mining
from src.experiments.rule_mining.tabpfn_experiments import adapt_tabpfn_for_reconstruction
from src.experiments.rule_mining.tabicl_experiments import adapt_tabicl_for_reconstruction
from src.experiments.rule_mining.tabdpt_experiments import adapt_tabdpt_for_reconstruction
from src.utils import (
    get_ucimlrepo_datasets,
    calculate_rule_metrics,
    set_seed,
    generate_seed_sequence,
)


def generate_reconstruction_cache(datasets, seed_sequence, methods, max_antecedents=2):
    """
    Generate and cache reconstruction probability matrices for all datasets, seeds, and methods.
    This avoids recomputing the expensive reconstruction step for each hyperparameter combination.

    Each method is handled individually with its specific initialization requirements:
    - TabPFN: requires TabPFNClassifier instance
    - TabICL: requires TabICLClassifier instance
    - TabDPT: creates models internally (no pre-initialization needed)

    Returns:
        cache: Dict with keys (method_name, dataset_name, run_seed) -> reconstruction data
    """
    cache = {}

    print("\n" + "=" * 80)
    print("CACHING RECONSTRUCTION PROBABILITIES")
    print("=" * 80)

    for method_name in methods:
        print(f"\n{method_name}:")

        for dataset_info in datasets:
            dataset_name = dataset_info['name']
            dataset = dataset_info['data']
            print(f"  {dataset_name}:", end=" ")

            for run_idx, run_seed in enumerate(seed_sequence):
                set_seed(run_seed)

                try:
                    # Prepare data
                    encoded_data, classes_per_feature, feature_names, encoder = prepare_categorical_data(dataset)

                    # Generate test matrix
                    test_matrix, test_descriptions, feature_value_indices = generate_test_matrix(
                        n_features=len(classes_per_feature),
                        classes_per_feature=classes_per_feature,
                        max_antecedents=max_antecedents,
                        use_zeros_for_unmarked=False
                    )

                    # Generate reconstruction probabilities (method-specific)
                    if method_name == 'TabPFN':
                        # Initialize TabPFN model
                        tabpfn_model = TabPFNClassifier(
                            n_estimators=8,
                            random_state=run_seed,
                            average_before_softmax=True,
                            inference_precision='auto'
                        )
                        reconstruction_probs = adapt_tabpfn_for_reconstruction(
                            tabpfn_model=tabpfn_model,
                            context_table=encoded_data,
                            query_matrix=test_matrix,
                            feature_value_indices=feature_value_indices,
                            n_samples=None  # Use full dataset
                        )
                    elif method_name == 'TabICL':
                        tabicl_model = TabICLClassifier(random_state=run_seed, n_estimators=8)
                        reconstruction_probs = adapt_tabicl_for_reconstruction(
                            tabicl_model=tabicl_model,
                            context_table=encoded_data,
                            query_matrix=test_matrix,
                            feature_value_indices=feature_value_indices,
                            n_samples=None  # Use full dataset
                        )
                    elif method_name == 'TabDPT':
                        # Suppress stderr to avoid TabDPT warnings
                        with suppress_stderr():
                            reconstruction_probs = adapt_tabdpt_for_reconstruction(
                                context_table=encoded_data,
                                query_matrix=test_matrix,
                                feature_value_indices=feature_value_indices,
                                n_ensembles=8,
                                random_state=run_seed
                            )
                    else:
                        raise ValueError(f"Unknown method: {method_name}")

                    # Cache everything needed for rule extraction
                    cache_key = (method_name, dataset_name, run_seed)
                    cache[cache_key] = {
                        'reconstruction_probs': reconstruction_probs,
                        'test_descriptions': test_descriptions,
                        'feature_value_indices': feature_value_indices,
                        'feature_names': feature_names,
                        'encoder': encoder,
                        'original_data': dataset.values
                    }

                    if run_idx == 0:
                        print(f"run 1/{len(seed_sequence)}", end="")
                    elif run_idx == len(seed_sequence) - 1:
                        print(f"...{run_idx + 1} ✓")

                except Exception as e:
                    print(f"\n    ERROR on run {run_idx + 1}: {e}")
                    cache_key = (method_name, dataset_name, run_seed)
                    cache[cache_key] = None

    print("\n" + "=" * 80)
    return cache


def run_hyperparam_experiment(method_name, datasets, seed_sequence, reconstruction_cache,
                              ant_sim, cons_sim):
    """
    Run hyperparameter experiment using cached reconstruction probabilities.
    Only performs rule extraction (fast), not reconstruction (slow).
    """
    results = []

    for dataset_info in datasets:
        dataset_name = dataset_info['name']
        dataset_run_results = []

        for run_idx, run_seed in enumerate(seed_sequence):
            cache_key = (method_name, dataset_name, run_seed)
            cached_data = reconstruction_cache.get(cache_key)

            if cached_data is None:
                dataset_run_results.append({
                    'num_rules': 0, 'support': 0.0, 'confidence': 0.0,
                    'zhangs_metric': 0.0, 'interestingness': 0.0,
                    'rule_coverage': 0.0, 'data_coverage': 0.0, 'time': 0.0
                })
                continue

            start_time = time.time()

            try:
                # Extract rules using cached reconstruction probabilities
                extracted_rules = extract_rules_from_reconstruction(
                    prob_matrix=cached_data['reconstruction_probs'],
                    test_descriptions=cached_data['test_descriptions'],
                    feature_value_indices=cached_data['feature_value_indices'],
                    ant_similarity=ant_sim,
                    cons_similarity=cons_sim,
                    feature_names=cached_data['feature_names'],
                    encoder=cached_data['encoder']
                )

                elapsed_time = time.time() - start_time

                if len(extracted_rules) > 0:
                    rules_with_metrics, avg_metrics = calculate_rule_metrics(
                        rules=extracted_rules,
                        data=cached_data['original_data'],
                        feature_names=cached_data['feature_names']
                    )
                    result = {
                        'num_rules': avg_metrics['num_rules'],
                        'support': avg_metrics['support'],
                        'confidence': avg_metrics['confidence'],
                        'zhangs_metric': avg_metrics['zhangs_metric'],
                        'interestingness': avg_metrics['interestingness'],
                        'rule_coverage': avg_metrics['rule_coverage'],
                        'data_coverage': avg_metrics['data_coverage'],
                        'time': elapsed_time
                    }
                else:
                    result = {
                        'num_rules': 0, 'support': 0.0, 'confidence': 0.0,
                        'zhangs_metric': 0.0, 'interestingness': 0.0,
                        'rule_coverage': 0.0, 'data_coverage': 0.0, 'time': elapsed_time
                    }
                dataset_run_results.append(result)

            except Exception as e:
                print(f"\n  ERROR on {dataset_name} run {run_idx + 1}: {e}")
                dataset_run_results.append({
                    'num_rules': 0, 'support': 0.0, 'confidence': 0.0,
                    'zhangs_metric': 0.0, 'interestingness': 0.0,
                    'rule_coverage': 0.0, 'data_coverage': 0.0, 'time': 0.0
                })

        # Average across runs
        runs_with_rules = [r for r in dataset_run_results if r['num_rules'] > 0]
        if runs_with_rules:
            avg_result = {
                'method': method_name,
                'dataset': dataset_name,
                'ant_similarity': ant_sim,
                'cons_similarity': cons_sim,
                'num_rules': np.mean([r['num_rules'] for r in runs_with_rules]),
                'support': np.mean([r['support'] for r in runs_with_rules]),
                'confidence': np.mean([r['confidence'] for r in runs_with_rules]),
                'zhangs_metric': np.mean([r['zhangs_metric'] for r in runs_with_rules]),
                'interestingness': np.mean([r['interestingness'] for r in runs_with_rules]),
                'rule_coverage': np.mean([r['rule_coverage'] for r in runs_with_rules]),
                'data_coverage': np.mean([r['data_coverage'] for r in runs_with_rules]),
                'avg_time': np.mean([r['time'] for r in dataset_run_results]),
                'runs_with_rules': len(runs_with_rules)
            }
        else:
            avg_result = {
                'method': method_name,
                'dataset': dataset_name,
                'ant_similarity': ant_sim,
                'cons_similarity': cons_sim,
                'num_rules': 0, 'support': 0.0, 'confidence': 0.0,
                'zhangs_metric': 0.0, 'interestingness': 0.0,
                'rule_coverage': 0.0, 'data_coverage': 0.0,
                'avg_time': np.mean([r['time'] for r in dataset_run_results]),
                'runs_with_rules': 0
            }

        results.append(avg_result)

    return results


if __name__ == "__main__":
    print("=" * 80)
    print("Hyperparameter Analysis: TabPFN, TabICL, TabDPT")
    print("=" * 80)

    # Parameters
    n_runs = 10
    max_antecedents = 2
    base_seed = 42

    # Hyperparameter grids
    cons_similarity_values = [0.5, 0.6, 0.7, 0.8, 0.9]
    ant_similarity_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    fixed_ant_similarity = 0.5
    fixed_cons_similarity = 0.8

    # Methods to test
    methods = ['TabPFN', 'TabICL', 'TabDPT']

    # Generate seed sequence
    print(f"\nGenerating seed sequence from base_seed={base_seed}...")
    seed_sequence = generate_seed_sequence(base_seed, n_runs)

    # Load datasets: 2 small + 2 large
    print("\nLoading datasets...")
    small_datasets = get_ucimlrepo_datasets(size="small")
    large_datasets = get_ucimlrepo_datasets(size="normal")
    datasets = small_datasets + large_datasets
    print(f"Datasets: {', '.join([d['name'] for d in datasets])}")

    # Create output directory
    os.makedirs("out", exist_ok=True)

    # Generate and cache reconstruction probabilities ONCE for all methods, datasets, and seeds
    reconstruction_cache = generate_reconstruction_cache(
        datasets=datasets,
        seed_sequence=seed_sequence,
        methods=methods,
        max_antecedents=max_antecedents
    )

    # Results storage
    all_results = []

    # Phase 1: Vary cons_similarity (ant_similarity fixed at 0.5)
    print("\n" + "=" * 80)
    print("PHASE 1: Varying cons_similarity (ant_similarity=0.5)")
    print("=" * 80)

    for cons_sim in cons_similarity_values:
        print(f"\ncons_similarity={cons_sim}")
        for method_name in methods:
            print(f"  {method_name}:", end=" ")
            results = run_hyperparam_experiment(
                method_name=method_name,
                datasets=datasets,
                seed_sequence=seed_sequence,
                reconstruction_cache=reconstruction_cache,
                ant_sim=fixed_ant_similarity,
                cons_sim=cons_sim
            )
            all_results.extend(results)
            print(f"{sum(r['num_rules'] for r in results) / len(results):.1f} avg rules")

    # Phase 2: Vary ant_similarity (cons_similarity fixed at 0.8)
    print("\n" + "=" * 80)
    print("PHASE 2: Varying ant_similarity (cons_similarity=0.8)")
    print("=" * 80)

    for ant_sim in ant_similarity_values:
        print(f"\nant_similarity={ant_sim}")
        for method_name in methods:
            print(f"  {method_name}:", end=" ")
            results = run_hyperparam_experiment(
                method_name=method_name,
                datasets=datasets,
                seed_sequence=seed_sequence,
                reconstruction_cache=reconstruction_cache,
                ant_sim=ant_sim,
                cons_sim=fixed_cons_similarity
            )
            all_results.extend(results)
            print(f"{sum(r['num_rules'] for r in results) / len(results):.1f} avg rules")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"out/hyperparameter_analysis_{timestamp}.xlsx"

    results_df = pd.DataFrame(all_results)

    # Calculate aggregate statistics across all datasets for each method and hyperparameter combination
    agg_stats = []
    for method_name in methods:
        for ant_sim in [fixed_ant_similarity] + ant_similarity_values:
            for cons_sim in cons_similarity_values + [fixed_cons_similarity]:
                subset = results_df[
                    (results_df['method'] == method_name) &
                    (results_df['ant_similarity'] == ant_sim) &
                    (results_df['cons_similarity'] == cons_sim)
                    ]

                if len(subset) > 0:
                    subset_with_rules = subset[subset['num_rules'] > 0]

                    if len(subset_with_rules) > 0:
                        agg_stats.append({
                            'method': method_name,
                            'ant_similarity': ant_sim,
                            'cons_similarity': cons_sim,
                            'avg_num_rules': subset_with_rules['num_rules'].mean(),
                            'avg_support': subset_with_rules['support'].mean(),
                            'avg_confidence': subset_with_rules['confidence'].mean(),
                            'avg_zhangs_metric': subset_with_rules['zhangs_metric'].mean(),
                            'avg_interestingness': subset_with_rules['interestingness'].mean(),
                            'avg_rule_coverage': subset_with_rules['rule_coverage'].mean(),
                            'avg_data_coverage': subset_with_rules['data_coverage'].mean()
                        })

    agg_df = pd.DataFrame(agg_stats)

    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Per Dataset Results', index=False)
        agg_df.to_excel(writer, sheet_name='Aggregated Results', index=False)

        params_df = pd.DataFrame([{
            'n_runs': n_runs,
            'base_seed': base_seed,
            'max_antecedents': max_antecedents,
            'methods': ', '.join(methods),
            'datasets': ', '.join([d['name'] for d in datasets])
        }])
        params_df.to_excel(writer, sheet_name='Parameters', index=False)

    print("\n" + "=" * 80)
    print(f"Results saved to {output_filename}")
    print(f"  - Sheet 1: Per Dataset Results")
    print(f"  - Sheet 2: Aggregated Results (across all datasets)")
    print(f"  - Sheet 3: Parameters")
    print("=" * 80)
