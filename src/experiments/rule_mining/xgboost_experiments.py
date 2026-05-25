"""
XGBoost-based Rule Extraction using TabProbe
"""

import time
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from concurrent.futures import ThreadPoolExecutor, as_completed
from xgboost import XGBClassifier

from src.utils.data_prep import prepare_categorical_data
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

try:
    import torch as _torch

    _CUDA_AVAILABLE = _torch.cuda.is_available()
except ImportError:
    _CUDA_AVAILABLE = False


def _process_feature_xgb(feat_info, context_table, query_matrix,
                         n_queries, n_estimators, max_depth, learning_rate,
                         random_state, query_batch_size, device):
    start_idx, end_idx = feat_info['start'], feat_info['end']
    n_classes = end_idx - start_idx

    x_context = np.delete(context_table, range(start_idx, end_idx), axis=1)
    y_context = np.argmax(context_table[:, start_idx:end_idx], axis=1)

    if len(np.unique(y_context)) < 2:
        return start_idx, end_idx, np.full((n_queries, n_classes), 1.0 / n_classes)

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        verbosity=0,
        eval_metric='mlogloss',
        device=device,
    )
    model.fit(x_context, y_context)

    x_query = np.delete(query_matrix, range(start_idx, end_idx), axis=1)
    batches = [model.predict_proba(x_query[b:b + query_batch_size])
               for b in range(0, n_queries, query_batch_size)]
    probs = np.vstack(batches)

    if probs.shape[1] != n_classes:
        aligned = np.zeros((n_queries, n_classes))
        for i, cls in enumerate(model.classes_):
            if int(cls) < n_classes:
                aligned[:, int(cls)] = probs[:, i]
        probs = aligned

    return start_idx, end_idx, probs


def adapt_xgb_for_reconstruction(context_table, query_matrix, feature_value_indices,
                                 n_samples=None, n_estimators=100, max_depth=3,
                                 learning_rate=0.3, random_state=42,
                                 max_workers=1, query_batch_size=512, device='cpu'):
    if n_samples and len(context_table) > n_samples:
        context_table = context_table[:n_samples]

    n_queries = query_matrix.shape[0]
    n_features_total = query_matrix.shape[1]
    reconstruction_probs = np.zeros((n_queries, n_features_total))

    print(f"    Reconstructing {len(feature_value_indices)} features for {n_queries} queries "
          f"(workers={max_workers}, batch_size={query_batch_size}, device={device})...")

    shared = (context_table, query_matrix, n_queries,
              n_estimators, max_depth, learning_rate, random_state, query_batch_size, device)

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_feature_xgb, fi, *shared): fi
                       for fi in feature_value_indices}
            for future in as_completed(futures):
                s, e, probs = future.result()
                reconstruction_probs[:, s:e] = probs
    else:
        for fi in feature_value_indices:
            s, e, probs = _process_feature_xgb(fi, *shared)
            reconstruction_probs[:, s:e] = probs

    return reconstruction_probs


def xgb_rule_learning(dataset, max_antecedents=2, context_samples=None,
                      n_estimators=100, max_depth=3, learning_rate=0.3,
                      ant_similarity=0.5, cons_similarity=0.8, random_state=42,
                      max_workers=1, query_batch_size=512, device='cpu'):
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

    print(f"\nUsing XGBoost for pattern reconstruction...")
    reconstruction_probs = adapt_xgb_for_reconstruction(
        context_table=encoded_data,
        query_matrix=test_matrix,
        feature_value_indices=feature_value_indices,
        n_samples=context_samples,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        max_workers=max_workers,
        query_batch_size=query_batch_size,
        device=device,
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
    print("XGBoost Baseline for Rule Learning")
    print("=" * 80)

    n_runs = 1
    max_antecedents = 2
    ant_similarity = 0.5
    cons_similarity = 0.8
    base_seed = 42
    # Defaults used when no tuned params exist for a dataset
    default_n_estimators = 100
    default_max_depth = 3
    default_learning_rate = 0.3
    max_workers = 4
    query_batch_size = 4096
    device = 'cuda' if _CUDA_AVAILABLE else 'cpu'

    print(f"Device: {device}")

    params_json = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "xgboost_best_parameters.json")
    )
    if os.path.exists(params_json):
        with open(params_json) as f:
            best_params_by_dataset = json.load(f)
        print(f"Loaded tuned parameters from {params_json}")
    else:
        best_params_by_dataset = {}
        print(f"No tuned parameters found at {params_json}, using defaults for all datasets.")

    print(f"\nGenerating seed sequence from base_seed={base_seed}...")
    seed_sequence = generate_seed_sequence(base_seed, n_runs)
    print(f"Seeds for {n_runs} runs: {seed_sequence}")

    print("\nLoading datasets...")
    datasets = get_ucimlrepo_datasets(size="normal", names=["breast_cancer"])

    os.makedirs("out", exist_ok=True)

    all_individual_results = []
    all_average_results = []

    for dataset_info in datasets:
        dataset_name = dataset_info['name']
        dataset = dataset_info['data']

        print("\n" + "=" * 80)
        print(f"Dataset: {dataset_name}")
        print("=" * 80)
        print(f"Shape: {dataset.shape}")

        tuned = best_params_by_dataset.get(dataset_name, {})
        n_estimators = tuned.get('n_estimators', default_n_estimators)
        max_depth = tuned.get('max_depth', default_max_depth)
        learning_rate = tuned.get('learning_rate', default_learning_rate)
        if tuned:
            print(f"Using tuned params: n_estimators={n_estimators}, "
                  f"max_depth={max_depth}, learning_rate={learning_rate:.4f}")
        else:
            print(f"Using default params: n_estimators={n_estimators}, "
                  f"max_depth={max_depth}, learning_rate={learning_rate}")

        dataset_runs = []

        for run_idx in range(n_runs):
            run_seed = seed_sequence[run_idx]
            print(f"\n--- Run {run_idx + 1}/{n_runs} (seed={run_seed}) ---")

            set_seed(run_seed)

            start_time = time.time()

            extracted_rules, feature_names, original_data = xgb_rule_learning(
                dataset=dataset,
                max_antecedents=max_antecedents,
                ant_similarity=ant_similarity,
                cons_similarity=cons_similarity,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=run_seed,
                max_workers=max_workers,
                query_batch_size=query_batch_size,
                device=device,
            )

            end_time = time.time()
            elapsed_time = end_time - start_time

            print(f"\nRun {run_idx + 1} completed in {elapsed_time:.2f} seconds")
            print(f"Extracted {len(extracted_rules)} rules")

            if len(extracted_rules) > 0:
                rules_with_metrics, avg_metrics = calculate_rule_metrics(
                    rules=extracted_rules,
                    data=original_data,
                    feature_names=feature_names
                )

                stats = convert_metrics_to_stats(avg_metrics)

                rules_file = save_rules(
                    rules=rules_with_metrics,
                    stats=stats,
                    dataset_name=dataset_name,
                    method_name="xgboost",
                    seed=run_seed
                )
                print(f"Rules saved to {rules_file}")

                print(f"  Support: {avg_metrics['support']:.4f}")
                print(f"  Confidence: {avg_metrics['confidence']:.4f}")
                print(f"  Zhang's Metric: {avg_metrics['zhangs_metric']:.4f}")
                print(f"  |Zhang's Metric|: {abs(avg_metrics['zhangs_metric']):.4f}")
                print(f"  Interestingness: {avg_metrics['interestingness']:.4f}")
                print(f"  Rule coverage: {avg_metrics['rule_coverage']:.4f}")
                print(f"  Data coverage: {avg_metrics['data_coverage']:.4f}")

                result = {
                    'dataset': dataset_name,
                    'run': run_idx + 1,
                    'seed': run_seed,
                    'num_rules': avg_metrics['num_rules'],
                    'avg_support': avg_metrics['support'],
                    'avg_confidence': avg_metrics['confidence'],
                    'avg_zhangs_metric': avg_metrics['zhangs_metric'],
                    'avg_abs_zhangs_metric': abs(avg_metrics['zhangs_metric']),
                    'avg_interestingness': avg_metrics['interestingness'],
                    'avg_rule_coverage': avg_metrics['rule_coverage'],
                    'data_coverage': avg_metrics['data_coverage'],
                    'execution_time': elapsed_time,
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
                    'avg_abs_zhangs_metric': 0.0,
                    'avg_interestingness': 0.0,
                    'avg_rule_coverage': 0.0,
                    'data_coverage': 0.0,
                    'execution_time': elapsed_time,
                }

            dataset_runs.append(result)
            all_individual_results.append(result)

        runs_with_rules = [r for r in dataset_runs if r['num_rules'] > 0]
        n_runs_with_rules = len(runs_with_rules)

        if n_runs_with_rules > 0:
            avg_result = {
                'dataset': dataset_name,
                'num_rules': np.mean([r['num_rules'] for r in runs_with_rules]),
                'avg_support': np.mean([r['avg_support'] for r in runs_with_rules]),
                'avg_confidence': np.mean([r['avg_confidence'] for r in runs_with_rules]),
                'avg_zhangs_metric': np.mean([r['avg_zhangs_metric'] for r in runs_with_rules]),
                'avg_abs_zhangs_metric': np.mean([r['avg_abs_zhangs_metric'] for r in runs_with_rules]),
                'avg_interestingness': np.mean([r['avg_interestingness'] for r in runs_with_rules]),
                'avg_rule_coverage': np.mean([r['avg_rule_coverage'] for r in runs_with_rules]),
                'data_coverage': np.mean([r['data_coverage'] for r in runs_with_rules]),
                'execution_time': np.mean([r['execution_time'] for r in dataset_runs]),
            }
        else:
            avg_result = {
                'dataset': dataset_name,
                'num_rules': 0,
                'avg_support': 0.0,
                'avg_confidence': 0.0,
                'avg_zhangs_metric': 0.0,
                'avg_abs_zhangs_metric': 0.0,
                'avg_interestingness': 0.0,
                'avg_rule_coverage': 0.0,
                'data_coverage': 0.0,
                'execution_time': np.mean([r['execution_time'] for r in dataset_runs]),
            }
        all_average_results.append(avg_result)

        print(f"\n=== Average Results for {dataset_name} ({n_runs_with_rules}/{n_runs} runs with rules) ===")
        print(f"  Rules: {avg_result['num_rules']:.1f}")
        print(f"  Support: {avg_result['avg_support']:.4f}")
        print(f"  Confidence: {avg_result['avg_confidence']:.4f}")
        print(f"  Zhang's Metric: {avg_result['avg_zhangs_metric']:.4f}")
        print(f"  |Zhang's Metric|: {avg_result['avg_abs_zhangs_metric']:.4f}")
        print(f"  Interestingness: {avg_result['avg_interestingness']:.4f}")
        print(f"  Rule Coverage: {avg_result['avg_rule_coverage']:.4f}")
        print(f"  Data Coverage: {avg_result['data_coverage']:.4f}")
        print(f"  Avg Time: {avg_result['execution_time']:.2f}s")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"out/xgboost_{timestamp}.xlsx"

    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        individual_df = pd.DataFrame(all_individual_results)
        individual_df.to_excel(writer, sheet_name='Individual Results', index=False)

        average_df = pd.DataFrame(all_average_results)
        average_df.to_excel(writer, sheet_name='Average Results', index=False)

        params_df = pd.DataFrame([{
            'n_runs': n_runs,
            'base_seed': base_seed,
            'max_antecedents': max_antecedents,
            'ant_similarity': ant_similarity,
            'cons_similarity': cons_similarity,
            'default_n_estimators': default_n_estimators,
            'default_max_depth': default_max_depth,
            'default_learning_rate': default_learning_rate,
            'device': device,
            'max_workers': max_workers,
            'query_batch_size': query_batch_size,
        }])
        params_df.to_excel(writer, sheet_name='Parameters', index=False)

        seeds_df = pd.DataFrame({
            'run': list(range(1, n_runs + 1)),
            'seed': seed_sequence
        })
        seeds_df.to_excel(writer, sheet_name='Seeds', index=False)

    print("\n" + "=" * 80)
    print(f"Results saved to {output_filename}")
    print(f"  - Sheet 1: Individual Results (all {n_runs * len(datasets)} runs)")
    print(f"  - Sheet 2: Average Results (per dataset)")
    print(f"  - Sheet 3: Parameters")
    print(f"  - Sheet 4: Seeds (seed sequence for reproducibility)")
    print("=" * 80)
