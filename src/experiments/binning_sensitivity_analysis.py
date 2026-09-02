"""
Discretization Bin-Count Sensitivity Analysis

Reruns rule mining across all instantiations (Aerial+, 3 TFMs, XGBoost, RandomForest,
FP-Growth at 3 support thresholds) with alternative discretization bin counts (3, 5)
alongside the paper's default (10). Reports rule quality only, matching the rule mining
experiments.

Restricted to DATASETS_TO_DISCRETIZE: bin count only affects datasets with numerical
columns, so purely categorical datasets would produce identical results at every bin
count and are skipped.
"""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

from src.utils import (
    get_ucimlrepo_datasets,
    calculate_rule_metrics,
    set_seed,
    generate_seed_sequence,
    convert_metrics_to_stats,
    load_tuned_params,
)
from src.utils.data_loading import DATASETS_TO_DISCRETIZE, UCIMLREPO_SMALL_DATASETS
from src.utils.discretization import discretize_numerical_features

from src.experiments.rule_mining.aerial_experiments import (
    aerial_rule_learning,
    get_dataset_parameters as get_aerial_dataset_parameters,
)
from src.experiments.rule_mining.tabpfn_experiments import tabpfn_rule_learning
from src.experiments.rule_mining.tabicl_experiments import tabicl_rule_learning
from src.experiments.rule_mining.tabdpt_experiments import tabdpt_rule_learning, filter_single_value_columns
from src.experiments.rule_mining.xgboost_experiments import (
    xgb_rule_learning, _CUDA_AVAILABLE, CALIBRATE_DATASETS,
)
from src.experiments.rule_mining.random_forest_experiments import rf_rule_learning
from src.experiments.rule_mining.fpgrowth_experiments import fpgrowth_rule_learning

_DIR = os.path.dirname(__file__)
XGBOOST_PARAMS_JSON = os.path.join(_DIR, "xgboost_best_parameters.json")
RANDOM_FOREST_PARAMS_JSON = os.path.join(_DIR, "random_forest_best_parameters.json")

ZERO_STATS = {
    'rule_count': 0, 'average_support': 0.0, 'average_confidence': 0.0,
    'average_zhangs_metric': 0.0, 'average_interestingness': 0.0,
    'average_coverage': 0.0, 'data_coverage': 0.0,
}


def _load_target_datasets():
    """Raw (undiscretized) versions of the datasets with numerical columns that XGBoost
    calibrates (DATASETS_TO_DISCRETIZE ∩ CALIBRATE_DATASETS)."""
    target_names = DATASETS_TO_DISCRETIZE & CALIBRATE_DATASETS
    small_names = [n for n in target_names if n in UCIMLREPO_SMALL_DATASETS]
    normal_names = [n for n in target_names if n not in UCIMLREPO_SMALL_DATASETS]

    loaded = {}
    for size, names in (('small', small_names), ('normal', normal_names)):
        if not names:
            continue
        for info in get_ucimlrepo_datasets(size=size, names=names, discretize=False):
            loaded[info['name']] = (info['data'], size)
    return loaded


def _run_seeded_method(method, dataset, dataset_name, dataset_size, run_seed,
                       tuned_xgb, tuned_rf):
    """Run one seeded (non-FP-Growth) method, returning a stats dict or None if no rules."""
    if method == 'tabdpt':
        dataset = filter_single_value_columns(dataset)

    if method == 'aerial':
        params = get_aerial_dataset_parameters(dataset_name, dataset_size)
        rules, stats = aerial_rule_learning(
            dataset,
            max_antecedents=MAX_ANTECEDENTS,
            ant_similarity=ANT_SIMILARITY,
            cons_similarity=CONS_SIMILARITY,
            batch_size=params['batch_size'],
            layer_dims=params['layer_dims'],
            epochs=params['epochs'],
            random_state=run_seed,
        )
        return stats if stats.get('rule_count', 0) > 0 else None

    if method == 'tabpfn':
        rules, feature_names, data = tabpfn_rule_learning(
            dataset,
            max_antecedents=MAX_ANTECEDENTS,
            context_samples=None,
            n_estimators=TFM_N_ESTIMATORS,
            ant_similarity=ANT_SIMILARITY,
            cons_similarity=CONS_SIMILARITY,
            random_state=run_seed,
            max_workers=MAX_WORKERS,
            query_batch_size=QUERY_BATCH_SIZE,
        )
    elif method == 'tabicl':
        rules, feature_names, data = tabicl_rule_learning(
            dataset,
            max_antecedents=MAX_ANTECEDENTS,
            context_samples=None,
            ant_similarity=ANT_SIMILARITY,
            cons_similarity=CONS_SIMILARITY,
            random_state=run_seed,
            n_estimators=TFM_N_ESTIMATORS,
            max_workers=MAX_WORKERS,
            query_batch_size=QUERY_BATCH_SIZE,
        )
    elif method == 'tabdpt':
        rules, feature_names, data = tabdpt_rule_learning(
            dataset,
            max_antecedents=MAX_ANTECEDENTS,
            ant_similarity=ANT_SIMILARITY,
            cons_similarity=CONS_SIMILARITY,
            n_ensembles=TFM_N_ESTIMATORS,
            random_state=run_seed,
            max_workers=1,  # torch.compile/dynamo is not thread-safe
            query_batch_size=QUERY_BATCH_SIZE,
        )
    elif method == 'xgboost':
        rules, feature_names, data = xgb_rule_learning(
            dataset,
            max_antecedents=MAX_ANTECEDENTS,
            context_samples=None,
            n_estimators=tuned_xgb.get('n_estimators', 100),
            max_depth=tuned_xgb.get('max_depth', 3),
            learning_rate=tuned_xgb.get('learning_rate', 0.3),
            ant_similarity=ANT_SIMILARITY,
            cons_similarity=CONS_SIMILARITY,
            random_state=run_seed,
            max_workers=MAX_WORKERS,
            query_batch_size=QUERY_BATCH_SIZE,
            device='cuda' if _CUDA_AVAILABLE else 'cpu',
        )
    elif method == 'random_forest':
        rules, feature_names, data = rf_rule_learning(
            dataset,
            max_antecedents=MAX_ANTECEDENTS,
            context_samples=None,
            n_estimators=tuned_rf.get('n_estimators', 100),
            max_depth=tuned_rf.get('max_depth', None),
            min_samples_leaf=tuned_rf.get('min_samples_leaf', 1),
            ant_similarity=ANT_SIMILARITY,
            cons_similarity=CONS_SIMILARITY,
            random_state=run_seed,
            max_workers=MAX_WORKERS,
            query_batch_size=QUERY_BATCH_SIZE,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    if len(rules) == 0:
        return None
    _, avg_metrics = calculate_rule_metrics(rules=rules, data=data, feature_names=feature_names)
    return convert_metrics_to_stats(avg_metrics)


def _stats_to_result(method, dataset_name, n_bins, run, seed, stats, elapsed_time):
    if stats is None:
        stats = ZERO_STATS
    return {
        'method': method,
        'dataset': dataset_name,
        'n_bins': n_bins,
        'run': run,
        'seed': seed,
        'num_rules': stats['rule_count'],
        'avg_support': stats['average_support'],
        'avg_confidence': stats['average_confidence'],
        'avg_zhangs_metric': stats['average_zhangs_metric'],
        'avg_abs_zhangs_metric': abs(stats['average_zhangs_metric']),
        'avg_interestingness': stats['average_interestingness'],
        'avg_rule_coverage': stats['average_coverage'],
        'data_coverage': stats['data_coverage'],
        'execution_time': elapsed_time,
    }


if __name__ == "__main__":
    print("=" * 80)
    print("Discretization Bin-Count Sensitivity Analysis")
    print("=" * 80)

    # Parameters (matching the individual rule mining experiments)
    N_RUNS = 10
    BASE_SEED = 42
    MAX_ANTECEDENTS = 2
    ANT_SIMILARITY = 0.5
    CONS_SIMILARITY = 0.8
    TFM_N_ESTIMATORS = 8
    MAX_WORKERS = 10
    QUERY_BATCH_SIZE = 4096
    BIN_COUNTS = [3, 5]
    FPGROWTH_SUPPORTS = [0.5, 0.3, 0.2, 0.1]
    FPGROWTH_MIN_CONFIDENCE = 0.8

    # SEEDED_METHODS = ['aerial', 'tabpfn', 'tabicl', 'tabdpt', 'xgboost', 'random_forest']
    SEEDED_METHODS = ['xgboost']
    # FPGROWTH_METHODS = [f'fpgrowth_{s}' for s in FPGROWTH_SUPPORTS]
    FPGROWTH_METHODS = []
    ALL_METHODS = SEEDED_METHODS + FPGROWTH_METHODS

    seed_sequence = generate_seed_sequence(BASE_SEED, N_RUNS)
    print(f"Seeds for {N_RUNS} runs: {seed_sequence}")
    print(f"Bin counts: {BIN_COUNTS}")
    print(f"Methods: {ALL_METHODS}")

    print("\nLoading raw (undiscretized) datasets...")
    raw_datasets = _load_target_datasets()
    print(f"Datasets with numerical columns ({len(raw_datasets)}): {sorted(raw_datasets)}")

    os.makedirs("out", exist_ok=True)
    all_individual_results = []

    for n_bins in BIN_COUNTS:
        for dataset_name, (raw_df, dataset_size) in raw_datasets.items():
            print("\n" + "=" * 80)
            print(f"Dataset: {dataset_name} | n_bins={n_bins}")
            print("=" * 80)

            dataset = discretize_numerical_features(raw_df, n_bins=n_bins)
            print(f"Shape after discretization: {dataset.shape}")

            tuned_xgb = load_tuned_params(XGBOOST_PARAMS_JSON, dataset_name)
            tuned_rf = load_tuned_params(RANDOM_FOREST_PARAMS_JSON, dataset_name)

            for method in ALL_METHODS:
                print(f"\n--- Method: {method} ---")

                if method.startswith('fpgrowth'):
                    support = float(method.split('_', 1)[1])
                    start_time = time.time()
                    try:
                        rules, stats = fpgrowth_rule_learning(
                            dataset,
                            min_support=support,
                            min_confidence=FPGROWTH_MIN_CONFIDENCE,
                            max_len=MAX_ANTECEDENTS,
                            compute_stats=True,
                        )
                        stats = stats if len(rules) > 0 else None
                    except Exception as e:
                        print(f"  Error: {e}")
                        stats = None
                    elapsed_time = time.time() - start_time

                    result = _stats_to_result(method, dataset_name, n_bins, 1, None,
                                              stats, elapsed_time)
                    print(f"  rules={result['num_rules']} confidence={result['avg_confidence']:.4f} "
                          f"({elapsed_time:.2f}s)")
                    all_individual_results.append(result)
                    continue

                for run_idx in range(N_RUNS):
                    run_seed = seed_sequence[run_idx]
                    set_seed(run_seed)

                    start_time = time.time()
                    try:
                        stats = _run_seeded_method(method, dataset, dataset_name, dataset_size,
                                                   run_seed, tuned_xgb, tuned_rf)
                    except Exception as e:
                        print(f"  Run {run_idx + 1} error: {e}")
                        stats = None
                    elapsed_time = time.time() - start_time

                    result = _stats_to_result(method, dataset_name, n_bins, run_idx + 1,
                                              run_seed, stats, elapsed_time)
                    all_individual_results.append(result)

                run_results = [r for r in all_individual_results
                               if r['method'] == method and r['dataset'] == dataset_name
                               and r['n_bins'] == n_bins]
                mean_conf = np.mean([r['avg_confidence'] for r in run_results if r['num_rules'] > 0] or [0.0])
                mean_rules = np.mean([r['num_rules'] for r in run_results])
                print(f"  avg rules={mean_rules:.1f} avg confidence={mean_conf:.4f} "
                      f"({elapsed_time:.2f}s last run)")

    # ---- Summarise ----

    individual_df = pd.DataFrame(all_individual_results)

    average_rows = []
    for (method, dataset_name, n_bins), group in individual_df.groupby(['method', 'dataset', 'n_bins']):
        with_rules = group[group['num_rules'] > 0]
        n_with_rules = len(with_rules)
        source = with_rules if n_with_rules > 0 else group
        average_rows.append({
            'method': method,
            'dataset': dataset_name,
            'n_bins': n_bins,
            'n_runs': len(group),
            'n_runs_with_rules': n_with_rules,
            'num_rules': source['num_rules'].mean(),
            'avg_support': source['avg_support'].mean(),
            'avg_confidence': source['avg_confidence'].mean(),
            'avg_zhangs_metric': source['avg_zhangs_metric'].mean(),
            'avg_abs_zhangs_metric': source['avg_abs_zhangs_metric'].mean(),
            'avg_interestingness': source['avg_interestingness'].mean(),
            'avg_rule_coverage': source['avg_rule_coverage'].mean(),
            'data_coverage': source['data_coverage'].mean(),
            'execution_time': group['execution_time'].mean(),
        })
    average_df = pd.DataFrame(average_rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"out/binning_sensitivity_{timestamp}.xlsx"

    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        individual_df.to_excel(writer, sheet_name='Individual Results', index=False)
        average_df.to_excel(writer, sheet_name='Average Results', index=False)

        params_df = pd.DataFrame([{
            'n_runs': N_RUNS,
            'base_seed': BASE_SEED,
            'max_antecedents': MAX_ANTECEDENTS,
            'ant_similarity': ANT_SIMILARITY,
            'cons_similarity': CONS_SIMILARITY,
            'tfm_n_estimators': TFM_N_ESTIMATORS,
            'max_workers': MAX_WORKERS,
            'query_batch_size': QUERY_BATCH_SIZE,
            'bin_counts': str(BIN_COUNTS),
            'fpgrowth_supports': str(FPGROWTH_SUPPORTS),
            'fpgrowth_min_confidence': FPGROWTH_MIN_CONFIDENCE,
            'datasets': str(sorted(raw_datasets)),
        }])
        params_df.to_excel(writer, sheet_name='Parameters', index=False)

        seeds_df = pd.DataFrame({'run': list(range(1, N_RUNS + 1)), 'seed': seed_sequence})
        seeds_df.to_excel(writer, sheet_name='Seeds', index=False)

    print("\n" + "=" * 80)
    print(f"Results saved to {output_filename}")
    print(f"  - Sheet 1: Individual Results (every run)")
    print(f"  - Sheet 2: Average Results (per method/dataset/n_bins)")
    print(f"  - Sheet 3: Parameters")
    print(f"  - Sheet 4: Seeds")
    print("=" * 80)
