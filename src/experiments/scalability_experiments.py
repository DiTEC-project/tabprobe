"""
Scalability Experiment: Execution Time vs. Number of Columns

This experiment evaluates how Aerial, TabICL, TabPFN, TabDPT, and FP-Growth scale with
increasing feature count (measured as one-hot encoded columns) in terms of execution time
"""
import time
import os
import gc
import io
import contextlib
import psutil
import pandas as pd
import numpy as np
from datetime import datetime
import torch
import warnings

from src.experiments.rule_mining.aerial_experiments import aerial_rule_learning
from src.experiments.rule_mining.tabicl_experiments import tabicl_rule_learning
from src.experiments.rule_mining.tabpfn_experiments import tabpfn_rule_learning
from src.experiments.rule_mining.tabdpt_experiments import tabdpt_rule_learning, filter_single_value_columns
from src.experiments.rule_mining.fpgrowth_experiments import fpgrowth_rule_learning

from src.utils import set_seed, generate_seed_sequence
from src.utils.data_prep import prepare_categorical_data


# ========== T10I4D100K Loader ==========

def load_t10i4d100k_as_table(path, n_rows=10000, seed=42):
    """
    Read T10I4D100K transaction file and return a binary DataFrame.

    Each row in the output corresponds to one transaction; each column
    corresponds to one item (named item_<id>), with value 1 if the item
    appears in that transaction and 0 otherwise.

    n_rows transactions are sampled without replacement using a fixed seed
    so results are reproducible.
    """
    transactions = []
    with open(path, 'r') as f:
        for line in f:
            items = [int(x) for x in line.strip().split(',') if x.strip()]
            if items:
                transactions.append(items)

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(transactions), size=min(n_rows, len(transactions)), replace=False)
    sampled = [transactions[i] for i in sorted(idx)]

    all_items = sorted({item for t in sampled for item in t})
    item_sets = [set(t) for t in sampled]

    data = {
        f'item_{item}': np.array([1 if item in s else 0 for s in item_sets], dtype=np.int8)
        for item in all_items
    }
    return pd.DataFrame(data)


# ========== Feature Selection ==========

def select_features_for_target_columns(dataset, classes_per_feature, target_n_columns):
    """
    Select the first N features whose total OHE column count is closest to
    (but does not exceed, if possible) target_n_columns.
    """
    cumulative_cols = 0
    n_features_selected = 0

    for n_classes in classes_per_feature:
        if cumulative_cols + n_classes > target_n_columns and cumulative_cols > 0:
            break
        cumulative_cols += n_classes
        n_features_selected += 1

    if n_features_selected == 0:
        n_features_selected = 1
        cumulative_cols = classes_per_feature[0]

    subset_df = dataset.iloc[:, :n_features_selected]
    return subset_df, cumulative_cols, n_features_selected


# ========== Measurement ==========

def measure_time_and_memory(method_func, dataset, method_params, method_name):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    process = psutil.Process()
    initial_memory_mb = process.memory_info().rss / 1024 ** 2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        start_time = time.time()
        success = True
        error_msg = None
        num_rules = 0

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = method_func(dataset, **method_params)
            rules = result[0] if isinstance(result, tuple) else result
            num_rules = len(rules) if rules else 0
        except Exception as e:
            success = False
            error_msg = str(e)

        end_time = time.time()

    current_memory_mb = process.memory_info().rss / 1024 ** 2
    peak_cpu_memory_mb = current_memory_mb

    peak_gpu_memory_mb = 0.0
    if torch.cuda.is_available():
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 ** 2

    return {
        'execution_time_sec': end_time - start_time,
        'peak_cpu_memory_mb': peak_cpu_memory_mb,
        'peak_gpu_memory_mb': peak_gpu_memory_mb,
        'success': success,
        'error_msg': error_msg,
        'num_rules': num_rules
    }


# ========== Main ==========

if __name__ == "__main__":
    # ---- Configuration ----

    T10I4D100K_PATH = "data/T10I4D100K.csv"
    N_ROWS = 5000
    n_runs_per_config = 5
    base_seed = 42

    # Target OHE column counts. Each binary item encodes to 2 OHE columns,
    # so target 200 → selects ~100 items. Adjust upper bound to taste;
    # the experiment handles failures gracefully if a run times out.
    target_column_counts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]

    # Number of parallel workers for TFM feature-predictor loop.
    # On GPU: 4 keeps memory manageable while exploiting CUDA stream concurrency.
    # On CPU: raise to os.cpu_count() for full utilisation.
    PARALLEL_WORKERS = 1
    QUERY_BATCH_SIZE = 4096

    # ---- Method Parameters ----

    aerial_params = {
        'max_antecedents': 2,
        'ant_similarity': 0.5,
        'cons_similarity': 0.8,
        'batch_size': 64,
        'layer_dims': [4],
        'epochs': 2,
        'quality_metrics': []
    }

    tabicl_params = {
        'max_antecedents': 2,
        'ant_similarity': 0.5,
        'cons_similarity': 0.8,
        'max_workers': PARALLEL_WORKERS,
        'query_batch_size': QUERY_BATCH_SIZE,
    }

    tabpfn_params = {
        'max_antecedents': 2,
        'ant_similarity': 0.5,
        'cons_similarity': 0.8,
        'max_workers': PARALLEL_WORKERS,
        'query_batch_size': QUERY_BATCH_SIZE,
    }

    tabdpt_params = {
        'max_antecedents': 2,
        'ant_similarity': 0.5,
        'cons_similarity': 0.8,
        'n_ensembles': 8,
        'max_workers': PARALLEL_WORKERS,
        'query_batch_size': QUERY_BATCH_SIZE,
    }

    fpgrowth_params_0_1 = {'max_len': 2, 'min_confidence': 0.8, 'min_support': 0.1, 'compute_stats': False}
    fpgrowth_params_0_05 = {'max_len': 2, 'min_confidence': 0.8, 'min_support': 0.05, 'compute_stats': False}
    fpgrowth_params_0_01 = {'max_len': 2, 'min_confidence': 0.8, 'min_support': 0.01, 'compute_stats': False}

    # ---- Load Dataset ----

    print("Scalability Experiment: Time & Memory vs. Number of Encoded Columns")
    print(f"Dataset: T10I4D100K | rows={N_ROWS} | up to 1998 OHE-encoded columns\n")

    print(f"Loading {T10I4D100K_PATH} ({N_ROWS} rows)...")
    dataset = load_t10i4d100k_as_table(T10I4D100K_PATH, n_rows=N_ROWS, seed=base_seed)
    dataset = filter_single_value_columns(dataset)
    encoded_data, classes_per_feature, feature_names, encoder = prepare_categorical_data(dataset)
    n_encoded_columns = encoded_data.shape[1]
    n_original_features = len(classes_per_feature)
    print(f"Items: {n_original_features}  |  OHE cols: {n_encoded_columns}  |  rows: {dataset.shape[0]}")

    # ---- Methods Registry ----
    # Sequential and parallel TFM variants are both included so the paper can
    # show the speedup from parallelisation on the same plot.

    methods = {
        'aerial': (aerial_rule_learning, aerial_params),
        'tabicl': (tabicl_rule_learning, tabicl_params),
        'tabpfn': (tabpfn_rule_learning, tabpfn_params),
        'tabdpt': (tabdpt_rule_learning, tabdpt_params),
        'fpgrowth_0.1': (fpgrowth_rule_learning, fpgrowth_params_0_1),
        'fpgrowth_0.05': (fpgrowth_rule_learning, fpgrowth_params_0_05),
        'fpgrowth_0.01': (fpgrowth_rule_learning, fpgrowth_params_0_01),
    }

    seed_sequence = generate_seed_sequence(base_seed, n_runs_per_config)
    print(f"Seeds: {seed_sequence}  |  targets: {target_column_counts}\n")

    # ---- Run Experiment ----

    all_results = []

    for target_cols in target_column_counts:
        subset_df, actual_cols, n_features = select_features_for_target_columns(
            dataset, classes_per_feature, target_cols
        )

        if actual_cols > n_encoded_columns:
            print(f"[{actual_cols} cols / {n_features} items]  SKIP (exceeds dataset size)", flush=True)
            continue

        print(f"\n[{actual_cols} cols / {n_features} items]", flush=True)

        for method_name, (method_func, method_params) in methods.items():
            run_times = []
            run_rules = []
            run_errors = []

            for run_idx in range(n_runs_per_config):
                run_seed = seed_sequence[run_idx]
                set_seed(run_seed)

                params_with_seed = method_params if 'fpgrowth' in method_name else {**method_params,
                                                                                    'random_state': run_seed}

                metrics = measure_time_and_memory(method_func, subset_df, params_with_seed, method_name)
                run_times.append(metrics['execution_time_sec'])
                run_rules.append(metrics['num_rules'])
                run_errors.append(metrics['error_msg'])

                all_results.append({
                    'method': method_name,
                    'target_n_columns': target_cols,
                    'actual_n_columns': actual_cols,
                    'n_features': n_features,
                    'run': run_idx + 1,
                    'seed': run_seed,
                    'execution_time_sec': metrics['execution_time_sec'],
                    'peak_cpu_memory_mb': metrics['peak_cpu_memory_mb'],
                    'peak_gpu_memory_mb': metrics['peak_gpu_memory_mb'],
                    'num_rules': metrics['num_rules'],
                    'success': metrics['success'],
                    'error': metrics['error_msg'] if not metrics['success'] else None,
                })

            successful_times = [t for t, e in zip(run_times, run_errors) if e is None]
            times_str = "  ".join(f"{t:.1f}s" for t in run_times)
            if successful_times:
                avg_t = np.mean(successful_times)
                std_t = np.std(successful_times)
                avg_rules = np.mean([r for r, e in zip(run_rules, run_errors) if e is None])
                summary = f"→ {avg_t:.2f}±{std_t:.2f}s  rules={avg_rules:.0f}"
            else:
                first_err = next(e for e in run_errors if e)
                summary = f"→ FAILED: {first_err[:60]}"
            print(f"  {method_name:<20} {times_str}  {summary}", flush=True)

    # ---- Summarise ----

    results_df = pd.DataFrame(all_results)

    summary_data = []
    for method in methods:
        method_results = results_df[results_df['method'] == method]
        for actual_cols in sorted(method_results['actual_n_columns'].unique()):
            subset = method_results[method_results['actual_n_columns'] == actual_cols]
            successful = subset[subset['success']]
            if len(successful) > 0:
                summary_data.append({
                    'method': method,
                    'n_columns': actual_cols,
                    'n_features': successful['n_features'].iloc[0],
                    'n_runs': len(subset),
                    'n_successful': len(successful),
                    'mean_time_sec': successful['execution_time_sec'].mean(),
                    'std_time_sec': successful['execution_time_sec'].std(),
                    'mean_cpu_mb': successful['peak_cpu_memory_mb'].mean(),
                    'std_cpu_mb': successful['peak_cpu_memory_mb'].std(),
                    'mean_gpu_mb': successful['peak_gpu_memory_mb'].mean(),
                    'std_gpu_mb': successful['peak_gpu_memory_mb'].std(),
                    'mean_num_rules': successful['num_rules'].mean(),
                    'std_num_rules': successful['num_rules'].std(),
                })

    summary_df = pd.DataFrame(summary_data)

    # ---- Save ----

    os.makedirs("out", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"out/{timestamp}_scalability.xlsx"

    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Individual Results', index=False)
        summary_df.to_excel(writer, sheet_name='Average Results', index=False)

        params_df = pd.DataFrame([{
            'dataset': 'T10I4D100K',
            'n_rows': N_ROWS,
            'n_runs_per_config': n_runs_per_config,
            'base_seed': base_seed,
            'parallel_workers': PARALLEL_WORKERS,
            'query_batch_size': QUERY_BATCH_SIZE,
            'target_column_counts': str(target_column_counts),
            'aerial_params': str(aerial_params),
            'tabicl_params': str(tabicl_params),
            'tabpfn_params': str(tabpfn_params),
            'tabdpt_params': str(tabdpt_params),
        }])
        params_df.to_excel(writer, sheet_name='Parameters', index=False)

        seeds_df = pd.DataFrame({'run': range(1, n_runs_per_config + 1), 'seed': seed_sequence})
        seeds_df.to_excel(writer, sheet_name='Seeds', index=False)

    print(f"\nResults saved to: {output_filename}")

    if len(summary_df) > 0:
        print("\nMean execution time (s) — rows=OHE cols, cols=methods")
        pivot = summary_df.pivot(index='n_columns', columns='method', values='mean_time_sec')
        print(pivot.to_string())

    print("\nDone.")
