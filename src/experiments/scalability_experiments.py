"""
Scalability Experiment: Execution Time and Peak Memory vs. Number of Columns

This experiment evaluates how Aerial, TabICL, TabPFN, and TabDPT scale with increasing
feature count (measured as one-hot encoded columns) in terms of:
- Execution time (seconds)
- Peak CPU memory consumption (MB)
- Peak GPU memory consumption (MB)

Methodology:
1. Load a dataset and one-hot encode it
2. Progressively select features to achieve target column counts (10, 15, 20, ...)
3. For each column count:
   - Run each method multiple times with different seeds
   - Measure execution time and peak memory
4. Save all results to {timestamp}_scalability.csv

For reproducibility:
- Fixed base seed (42)
- Multiple runs per configuration (n=5)
- All parameters logged
- Feature selection is deterministic (first N features)
"""
import time
import os
import gc
import psutil
import pandas as pd
from datetime import datetime
import torch
import warnings

from src.experiments.rule_mining.aerial_experiments import aerial_rule_learning
from src.experiments.rule_mining.tabicl_experiments import tabicl_rule_learning
from src.experiments.rule_mining.tabpfn_experiments import tabpfn_rule_learning
from src.experiments.rule_mining.tabdpt_experiments import tabdpt_rule_learning, filter_single_value_columns

from src.utils import get_ucimlrepo_datasets, set_seed, generate_seed_sequence
from src.utils.data_prep import prepare_categorical_data


def select_features_for_target_columns(dataset, classes_per_feature, target_n_columns):
    """
    Select the first N features such that the total one-hot encoded columns
    is closest to (but not exceeding, if possible) target_n_columns.

    Args:
        dataset: Original DataFrame
        classes_per_feature: List of number of classes per feature
        target_n_columns: Target number of columns after encoding

    Returns:
        subset_df: DataFrame with selected features
        actual_n_columns: Actual number of columns after encoding
        n_features_selected: Number of original features selected
    """
    cumulative_cols = 0
    n_features_selected = 0

    for i, n_classes in enumerate(classes_per_feature):
        if cumulative_cols + n_classes > target_n_columns and cumulative_cols > 0:
            # Adding this feature would exceed target, stop here
            break
        cumulative_cols += n_classes
        n_features_selected += 1

    # If we selected 0 features, select at least 1
    if n_features_selected == 0:
        n_features_selected = 1
        cumulative_cols = classes_per_feature[0]

    # Select the first n_features_selected columns from the original dataset
    subset_df = dataset.iloc[:, :n_features_selected]

    return subset_df, cumulative_cols, n_features_selected


def measure_time_and_memory(method_func, dataset, method_params, method_name):
    """
    Execute a method and measure execution time and peak memory consumption.

    Args:
        method_func: Function to call (e.g., aerial_rule_learning)
        dataset: DataFrame to pass to method
        method_params: Dictionary of parameters
        method_name: Name of method (for display)

    Returns:
        Dictionary with metrics: execution_time_sec, peak_cpu_memory_mb,
        peak_gpu_memory_mb, success, error_msg, num_rules
    """
    # Clear memory before measurement
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Get process for memory tracking
    process = psutil.Process()
    initial_memory_mb = process.memory_info().rss / 1024 ** 2

    # Suppress warnings during execution
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Start timing
        start_time = time.time()
        success = True
        error_msg = None
        num_rules = 0

        try:
            # Execute method
            result = method_func(dataset, **method_params)

            # Extract rules (different methods return different formats)
            if isinstance(result, tuple):
                rules = result[0]  # First element is usually the rules list
            else:
                rules = result

            num_rules = len(rules) if rules else 0

        except Exception as e:
            success = False
            error_msg = str(e)
            print(f"ERROR: {error_msg}")

        # End timing
        end_time = time.time()

    # Measure peak memory
    current_memory_mb = process.memory_info().rss / 1024 ** 2
    peak_cpu_memory_mb = current_memory_mb  # Approximate (process peak)

    peak_gpu_memory_mb = 0.0
    if torch.cuda.is_available():
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 ** 2

    execution_time_sec = end_time - start_time

    return {
        'execution_time_sec': execution_time_sec,
        'peak_cpu_memory_mb': peak_cpu_memory_mb,
        'peak_gpu_memory_mb': peak_gpu_memory_mb,
        'success': success,
        'error_msg': error_msg,
        'num_rules': num_rules
    }


# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("Scalability Experiment: Time & Memory vs. Number of Columns")
    print("=" * 80)

    # ========== Experiment Configuration ==========

    # Dataset selection
    dataset_name_to_use = 'mushroom'  # Dataset with many features

    # Column count targets (approximate, will select complete features)
    target_column_counts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

    # Number of runs per configuration for averaging
    n_runs_per_config = 10

    # Base seed for reproducibility
    base_seed = 42

    # Method parameters (aligned across all methods for fair comparison)
    # Using lightweight parameters for faster execution
    aerial_params = {
        'max_antecedents': 2,
        'ant_similarity': 0.5,
        'cons_similarity': 0.8,
        'batch_size': 64,
        'layer_dims': [4],
        'epochs': 2,
        'quality_metrics': []  # skip calculating quality metrics for fairness
    }

    tabicl_params = {
        'max_antecedents': 2,
        'ant_similarity': 0.5,
        'cons_similarity': 0.8,
    }

    tabpfn_params = {
        'max_antecedents': 2,
        'ant_similarity': 0.5,
        'cons_similarity': 0.8,
    }

    tabdpt_params = {
        'max_antecedents': 2,
        'ant_similarity': 0.5,
        'cons_similarity': 0.8,
        'n_ensembles': 8,
    }

    # ========== Load and Prepare Dataset ==========

    print(f"\nLoading dataset: {dataset_name_to_use}")
    datasets = get_ucimlrepo_datasets(size="normal", names=[dataset_name_to_use])
    dataset_info = datasets[0]
    dataset = dataset_info['data']
    dataset = filter_single_value_columns(dataset)

    print(f"Original dataset shape: {dataset.shape}")
    print(f"Original features: {dataset.shape[1]}")

    # One-hot encode to understand the column structure
    print("\nAnalyzing one-hot encoded structure...")
    encoded_data, classes_per_feature, feature_names, encoder = prepare_categorical_data(dataset)
    n_encoded_columns = encoded_data.shape[1]
    n_original_features = len(classes_per_feature)

    print(f"One-hot encoded columns: {n_encoded_columns}")
    print(f"Original features: {n_original_features}")
    print(f"Classes per feature: {classes_per_feature[:10]}..." if len(
        classes_per_feature) > 10 else f"Classes per feature: {classes_per_feature}")

    # ========== Run Scalability Experiments ==========

    # Results storage
    all_results = []

    # Generate seed sequence for reproducibility
    seed_sequence = generate_seed_sequence(base_seed, n_runs_per_config)
    print(f"\nSeed sequence for {n_runs_per_config} runs: {seed_sequence}")

    # Methods to test
    methods = {
        'aerial': (aerial_rule_learning, aerial_params),
        'tabicl': (tabicl_rule_learning, tabicl_params),
        'tabpfn': (tabpfn_rule_learning, tabpfn_params),
        'tabdpt': (tabdpt_rule_learning, tabdpt_params),
    }

    # Test each target column count
    for target_cols in target_column_counts:
        print(f"\n{'=' * 80}")
        print(f"Target Columns: {target_cols}")
        print(f"{'=' * 80}")

        # Select features to achieve target column count
        subset_df, actual_cols, n_features = select_features_for_target_columns(
            dataset, classes_per_feature, target_cols
        )

        print(f"Selected {n_features} features → {actual_cols} encoded columns")
        print(f"Subset shape: {subset_df.shape}")

        # Skip if target is too large
        if actual_cols > n_encoded_columns:
            print(f"Skipping: target exceeds dataset size")
            continue

        # Test each method
        for method_name, (method_func, method_params) in methods.items():
            print(f"\n  Method: {method_name}")

            # Run multiple times for statistical reliability
            for run_idx in range(n_runs_per_config):
                run_seed = seed_sequence[run_idx]
                print(f"    Run {run_idx + 1}/{n_runs_per_config} (seed={run_seed})...", end=" ")

                # Set seed for reproducibility
                set_seed(run_seed)

                # Add random_state to parameters
                params_with_seed = {**method_params, 'random_state': run_seed}

                # Measure execution time and memory
                metrics = measure_time_and_memory(
                    method_func,
                    subset_df,
                    params_with_seed,
                    method_name
                )

                # Display results
                if metrics['success']:
                    print(f"✓ Time: {metrics['execution_time_sec']:.2f}s, "
                          f"CPU: {metrics['peak_cpu_memory_mb']:.0f}MB, "
                          f"GPU: {metrics['peak_gpu_memory_mb']:.0f}MB, "
                          f"Rules: {metrics['num_rules']}")
                else:
                    print(f"✗ FAILED: {metrics['error_msg']}")

                # Store results
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
                    'error': metrics['error_msg'] if not metrics['success'] else None
                })

    # ========== Save Results ==========

    print(f"\n{'=' * 80}")
    print("Saving results...")

    # Create output directory
    os.makedirs("out", exist_ok=True)

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)

    # ========== Summary Statistics ==========

    print(f"\n{'=' * 80}")
    print("Calculating Summary Statistics (Mean ± Std across runs)")
    print(f"{'=' * 80}")

    # Calculate mean and std for each method and column count
    summary_data = []
    for method in methods.keys():
        method_results = results_df[results_df['method'] == method]

        for actual_cols in sorted(method_results['actual_n_columns'].unique()):
            subset = method_results[method_results['actual_n_columns'] == actual_cols]

            if len(subset) > 0:
                # Include all runs, even if some failed (mark success rate)
                successful_runs = subset[subset['success'] == True]
                n_successful = len(successful_runs)
                n_total = len(subset)

                if n_successful > 0:
                    summary_data.append({
                        'method': method,
                        'n_columns': actual_cols,
                        'n_features': successful_runs['n_features'].iloc[0],
                        'n_runs': n_total,
                        'n_successful': n_successful,
                        'mean_time_sec': successful_runs['execution_time_sec'].mean(),
                        'std_time_sec': successful_runs['execution_time_sec'].std(),
                        'mean_cpu_mb': successful_runs['peak_cpu_memory_mb'].mean(),
                        'std_cpu_mb': successful_runs['peak_cpu_memory_mb'].std(),
                        'mean_gpu_mb': successful_runs['peak_gpu_memory_mb'].mean(),
                        'std_gpu_mb': successful_runs['peak_gpu_memory_mb'].std(),
                        'mean_num_rules': successful_runs['num_rules'].mean(),
                        'std_num_rules': successful_runs['num_rules'].std(),
                    })

    summary_df = pd.DataFrame(summary_data)

    # ========== Save to Excel with Multiple Sheets ==========

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"out/{timestamp}_scalability.xlsx"

    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        # Sheet 1: All Individual Results
        results_df.to_excel(writer, sheet_name='Individual Results', index=False)

        # Sheet 2: Average Results (Mean ± Std per method per column count)
        summary_df.to_excel(writer, sheet_name='Average Results', index=False)

        # Sheet 3: Parameters (for reproducibility)
        params_df = pd.DataFrame([{
            'dataset': dataset_name_to_use,
            'n_runs_per_config': n_runs_per_config,
            'base_seed': base_seed,
            'target_column_counts': str(target_column_counts),
            'aerial_params': str(aerial_params),
            'tabicl_params': str(tabicl_params),
            'tabpfn_params': str(tabpfn_params),
            'tabdpt_params': str(tabdpt_params),
        }])
        params_df.to_excel(writer, sheet_name='Parameters', index=False)

        # Sheet 4: Seed Sequence (for reproducibility)
        seeds_df = pd.DataFrame({
            'run': list(range(1, n_runs_per_config + 1)),
            'seed': seed_sequence
        })
        seeds_df.to_excel(writer, sheet_name='Seeds', index=False)

    print(f"\nResults saved to: {output_filename}")
    print(f"  - Sheet 1: Individual Results (all runs: {len(results_df)} total)")
    print(f"  - Sheet 2: Average Results (mean ± std per method per column count)")
    print(f"  - Sheet 3: Parameters (experiment configuration)")
    print(f"  - Sheet 4: Seeds (seed sequence for reproducibility)")

    # ========== Display Summary Tables ==========

    if len(summary_df) > 0:
        print(f"\n{'=' * 80}")
        print("Summary: Execution Time (Mean ± Std seconds)")
        print("-" * 80)

        # Create a nice formatted display
        for method in methods.keys():
            method_summary = summary_df[summary_df['method'] == method]
            if len(method_summary) > 0:
                print(f"\n{method.upper()}:")
                for _, row in method_summary.iterrows():
                    print(f"  {row['n_columns']:3d} cols: {row['mean_time_sec']:7.2f} ± {row['std_time_sec']:5.2f}s  "
                          f"(CPU: {row['mean_cpu_mb']:6.0f} ± {row['std_cpu_mb']:4.0f} MB, "
                          f"GPU: {row['mean_gpu_mb']:6.0f} ± {row['std_gpu_mb']:4.0f} MB)")

        # Pivot tables for easy comparison
        print(f"\n{'=' * 80}")
        print("Comparison Table: Mean Execution Time (seconds)")
        print("-" * 80)
        pivot_time = summary_df.pivot(index='n_columns', columns='method', values='mean_time_sec')
        print(pivot_time.to_string())

        print(f"\n{'=' * 80}")
        print("Comparison Table: Mean Peak CPU Memory (MB)")
        print("-" * 80)
        pivot_memory = summary_df.pivot(index='n_columns', columns='method', values='mean_cpu_mb')
        print(pivot_memory.to_string())

    print(f"\n{'=' * 80}")
    print("Scalability experiment completed!")
    print(f"{'=' * 80}")
