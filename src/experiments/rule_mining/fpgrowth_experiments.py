"""
FP-Growth Rule Mining Experiments

This module runs association rule mining experiments using FP-Growth algorithm.

FP-Growth (Frequent Pattern Growth) is a classic frequent itemset mining algorithm
that discovers association rules from transactional data. This implementation uses
the mlxtend library (CPU-based).

IMPORTANT: FP-Growth is deterministic and does not use random seeds. It runs once
per dataset using calibrated min_support thresholds from Aerial.
"""
import time
import os
import numpy as np
import pandas as pd
from datetime import datetime
import psutil
import tracemalloc

from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

from src.utils import (
    get_ucimlrepo_datasets,
    calculate_rule_metrics,
    save_rules,
    load_fpgrowth_calibration,
)


def convert_fpgrowth_rules_to_pyaerial_format(rules_df):
    """
    Convert mlxtend association rules format to PyAerial format.

    mlxtend format:
        antecedents: frozenset({'feature=value'})
        consequents: frozenset({'feature=value'})

    PyAerial format:
        antecedents: [{"feature": "feature_name", "value": "category_value"}, ...]
        consequent: {"feature": "feature_name", "value": "category_value"}

    Args:
        rules_df: DataFrame from mlxtend association_rules

    Returns:
        List of rules in PyAerial format
    """
    pyaerial_rules = []

    for _, row in rules_df.iterrows():
        # Parse antecedents
        antecedents = []
        for item in row['antecedents']:
            feature, value = item.split('=')
            antecedents.append({
                'feature': feature,
                'value': value
            })

        # Parse consequent (mlxtend can have multiple, but we take first for consistency)
        # NOTE that a --> b & c is logically equivalent to (a --> b) & (a --> c)
        consequent_items = list(row['consequents'])
        if len(consequent_items) > 0:
            feature, value = consequent_items[0].split('=')
            consequent = {
                'feature': feature,
                'value': value
            }

            # Only add rules with single consequent (matching Aerial format)
            if len(consequent_items) == 1:
                pyaerial_rules.append({
                    'antecedents': antecedents,
                    'consequent': consequent
                })

    return pyaerial_rules


def fpgrowth_rule_learning(dataset, min_support=0.05, min_confidence=0.8, max_len=2):
    """
    Association rule mining using FP-Growth algorithm.

    FP-Growth is deterministic - no random seed needed.

    Args:
        dataset: DataFrame with categorical features
        min_support: Minimum support threshold (default: 0.05)
        min_confidence: Minimum confidence threshold (default: 0.8)
        max_len: Maximum length of itemsets (default: 2 for max_antecedents=2)

    Returns:
        rules: List of extracted association rules in PyAerial format
        stats: Statistics dictionary
    """
    feature_names = list(dataset.columns)

    print(f"Dataset shape: {dataset.shape}")
    print(f"Number of features: {len(feature_names)}")

    # Convert DataFrame to transaction format
    # Each row becomes: ['feature1=value1', 'feature2=value2', ...]
    transactions = []
    for _, row in dataset.iterrows():
        transaction = [f"{feature}={value}" for feature, value in row.items()]
        transactions.append(transaction)

    # One-hot encode transactions
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    print(f"\nRunning FP-Growth...")
    print(f"  min_support={min_support}")
    print(f"  max_len={max_len + 1}")  # +1 because max_len includes consequent

    # Run FP-Growth
    frequent_itemsets = fpgrowth(
        df_encoded,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len + 1  # +1 to account for consequent
    )

    if len(frequent_itemsets) == 0:
        print("  WARNING: No frequent itemsets found!")
        return [], {
            'rule_count': 0,
            'average_support': 0.0,
            'average_confidence': 0.0,
            'average_zhangs_metric': 0.0,
            'average_interestingness': 0.0,
            'average_coverage': 0.0,
            'data_coverage': 0.0
        }

    print(f"  Found {len(frequent_itemsets)} frequent itemsets")

    # Generate association rules
    print(f"\nGenerating association rules...")
    print(f"  min_confidence={min_confidence}")

    rules_df = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence,
        num_itemsets=len(frequent_itemsets)
    )

    if len(rules_df) == 0:
        print("  WARNING: No rules meet confidence threshold!")
        return [], {
            'rule_count': 0,
            'average_support': 0.0,
            'average_confidence': 0.0,
            'average_zhangs_metric': 0.0,
            'average_interestingness': 0.0,
            'average_coverage': 0.0,
            'data_coverage': 0.0
        }

    print(f"  Found {len(rules_df)} rules (before single-consequent filtering)")

    # Convert to PyAerial format
    pyaerial_rules = convert_fpgrowth_rules_to_pyaerial_format(rules_df)

    print(f"\n{len(pyaerial_rules)} rules in PyAerial format")

    # Calculate quality metrics using the shared utility
    if len(pyaerial_rules) > 0:
        # Convert dataset to numpy for metric calculation
        data_np = dataset.values

        # Calculate metrics
        rules_with_metrics, avg_metrics = calculate_rule_metrics(
            pyaerial_rules,
            data_np,
            feature_names=feature_names,
            metric_names=['support', 'confidence', 'rule_coverage', 'zhangs_metric', 'interestingness']
        )

        stats = {
            'rule_count': avg_metrics['num_rules'],
            'average_support': avg_metrics['support'],
            'average_confidence': avg_metrics['confidence'],
            'average_zhangs_metric': avg_metrics['zhangs_metric'],
            'average_interestingness': avg_metrics['interestingness'],
            'average_coverage': avg_metrics['rule_coverage'],
            'data_coverage': avg_metrics['data_coverage']
        }

        print(f"  Avg support={stats['average_support']:.4f}, "
              f"avg confidence={stats['average_confidence']:.4f}")

        return rules_with_metrics, stats
    else:
        stats = {
            'rule_count': 0,
            'average_support': 0.0,
            'average_confidence': 0.0,
            'average_zhangs_metric': 0.0,
            'average_interestingness': 0.0,
            'average_coverage': 0.0,
            'data_coverage': 0.0
        }

        return [], stats


# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("FP-Growth Rule Mining Experiments (Calibrated)")
    print("=" * 80)
    print("\nNOTE: FP-Growth is deterministic. It runs once per dataset using")
    print("      calibrated min_support thresholds from Aerial experiments.")
    print("=" * 80)

    # Parameters
    max_len = 2  # Max antecedents (matching other experiments)
    min_confidence = 0.8
    reference_method = "aerial"
    coverage_percentage = 0.9

    # Load datasets
    print("\nLoading datasets...")
    datasets = get_ucimlrepo_datasets(size="normal", names=["breast_cancer"])

    # Create output directory
    os.makedirs("out", exist_ok=True)

    # Results storage
    all_results = []

    # Run experiments for each dataset
    for dataset_info in datasets:
        dataset_name = dataset_info['name']
        dataset = dataset_info['data']

        print("\n" + "=" * 80)
        print(f"Dataset: {dataset_name}")
        print("=" * 80)
        print(f"Shape: {dataset.shape}")

        # Load calibrated threshold from Aerial
        print(f"\nLoading calibrated threshold from {reference_method}...")
        try:
            min_support = load_fpgrowth_calibration(
                dataset_name=dataset_name,
                reference_method=reference_method
            )
            print(f"Calibrated min_support: {min_support:.4f} ({coverage_percentage * 100:.0f}% coverage)")
        except ValueError as e:
            print(f"ERROR: {e}")
            print(f"Skipping {dataset_name}. Run {reference_method} experiments first.")
            continue

        # Track CPU memory
        process = psutil.Process()
        tracemalloc.start()
        mem_before = process.memory_info().rss / 1024 ** 2

        start_time = time.time()

        # Run FP-Growth (once, deterministically)
        extracted_rules, stats = fpgrowth_rule_learning(
            dataset=dataset,
            min_support=min_support,
            min_confidence=min_confidence,
            max_len=max_len
        )

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Get peak CPU memory usage
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_after = process.memory_info().rss / 1024 ** 2
        peak_cpu_memory_mb = max(mem_after - mem_before, peak_mem / 1024 ** 2)

        print(f"\nCompleted in {elapsed_time:.2f} seconds")
        print(f"Extracted {len(extracted_rules)} rules")
        print(f"Peak CPU Memory: {peak_cpu_memory_mb:.2f} MB")

        # Save rules (FP-Growth is deterministic, no seed needed)
        rules_file = save_rules(
            rules=extracted_rules,
            stats=stats,
            dataset_name=dataset_name,
            method_name="fpgrowth",
            seed=None  # No seed for deterministic method
        )
        print(f"Rules saved to {rules_file}")

        # Display metrics
        if len(extracted_rules) > 0:
            print(f"  Support: {stats.get('average_support', 0):.4f}")
            print(f"  Confidence: {stats.get('average_confidence', 0):.4f}")
            print(f"  Zhang's Metric: {stats.get('average_zhangs_metric', 0):.4f}")
            print(f"  Interestingness: {stats.get('average_interestingness', 0):.4f}")
            print(f"  Rule coverage: {stats.get('average_coverage', 0):.4f}")
            print(f"  Data coverage: {stats.get('data_coverage', 0):.4f}")

            result = {
                'dataset': dataset_name,
                'calibrated_min_support': min_support,
                'num_rules': stats.get('rule_count', len(extracted_rules)),
                'avg_support': stats.get('average_support', 0),
                'avg_confidence': stats.get('average_confidence', 0),
                'avg_zhangs_metric': stats.get('average_zhangs_metric', 0),
                'avg_interestingness': stats.get('average_interestingness', 0),
                'avg_rule_coverage': stats.get('average_coverage', 0),
                'data_coverage': stats.get('data_coverage', 0),
                'execution_time': elapsed_time,
                'peak_cpu_memory_mb': peak_cpu_memory_mb
            }
        else:
            print("  WARNING: No rules extracted!")
            result = {
                'dataset': dataset_name,
                'calibrated_min_support': min_support,
                'num_rules': 0,
                'avg_support': 0.0,
                'avg_confidence': 0.0,
                'avg_zhangs_metric': 0.0,
                'avg_interestingness': 0.0,
                'avg_rule_coverage': 0.0,
                'data_coverage': 0.0,
                'execution_time': elapsed_time,
                'peak_cpu_memory_mb': peak_cpu_memory_mb
            }

        all_results.append(result)

    # Save results to Excel
    if len(all_results) > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"out/fpgrowth_{timestamp}.xlsx"

        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            # Sheet 1: Results (one row per dataset)
            results_df = pd.DataFrame(all_results)
            results_df.to_excel(writer, sheet_name='Results', index=False)

            # Sheet 2: Parameters
            params_df = pd.DataFrame([{
                'max_antecedents': max_len,
                'min_confidence': min_confidence,
                'reference_method': reference_method,
                'coverage_percentage': coverage_percentage,
                'note': 'CPU-based FP-Growth using mlxtend (deterministic, calibrated thresholds)'
            }])
            params_df.to_excel(writer, sheet_name='Parameters', index=False)

        print("\n" + "=" * 80)
        print(f"Results saved to {output_filename}")
        print(f"  - Sheet 1: Results (one row per dataset)")
        print(f"  - Sheet 2: Parameters")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("No results to save. Run Aerial experiments first to generate calibration thresholds.")
        print("=" * 80)
