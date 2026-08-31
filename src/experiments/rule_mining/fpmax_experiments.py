"""
FP-Max Rule Mining Experiments (maximal-itemset baseline)
"""
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
import psutil
import tracemalloc

from mlxtend.frequent_patterns import fpgrowth, fpmax, association_rules
from mlxtend.preprocessing import TransactionEncoder

from src.utils import (
    get_ucimlrepo_datasets,
    save_rules
)
from src.experiments.rule_mining.fpgrowth_experiments import (
    convert_fpgrowth_rules_to_pyaerial_format,
    compute_rule_stats,
)

ZERO_STATS = {
    'rule_count': 0, 'average_support': 0.0, 'average_confidence': 0.0,
    'average_zhangs_metric': 0.0, 'average_interestingness': 0.0,
    'average_coverage': 0.0, 'data_coverage': 0.0,
}


def fpmax_rule_learning(dataset, min_support=0.05, min_confidence=0.8, max_len=2, compute_stats=True):
    """
    Association rule mining restricted to maximal frequent itemsets (FP-Max).

    Args:
        dataset: DataFrame with categorical features
        min_support: Minimum support threshold (default: 0.05)
        min_confidence: Minimum confidence threshold (default: 0.8)
        max_len: Maximum length of itemsets (default: 2 for max_antecedents=2)

    Returns:
        rules: List of extracted association rules in PyAerial format, restricted to
            rules whose full itemset is maximal
        stats: Statistics dictionary
    """
    feature_names = list(dataset.columns)

    print(f"Dataset shape: {dataset.shape}")
    print(f"Number of features: {len(feature_names)}")

    transactions = []
    for _, row in dataset.iterrows():
        transaction = [f"{feature}={value}" for feature, value in row.items()]
        transactions.append(transaction)

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # we need to run FP-Growth first for technical reasons
    # the association_rules() function mlxtend does not support maximal_itemsets as input,
    # and only supports the frequent itemsets from fpgrowth (and others)
    frequent_itemsets = fpgrowth(
        df_encoded,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len + 1
    )

    if len(frequent_itemsets) == 0:
        print("  WARNING: No frequent itemsets found!")
        return [], dict(ZERO_STATS)

    print(f"  Found {len(frequent_itemsets)} frequent itemsets")

    print(f"\nRunning FP-Max (maximal itemsets)...")
    maximal_itemsets = fpmax(
        df_encoded,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len + 1
    )

    if len(maximal_itemsets) == 0:
        print("  WARNING: No maximal itemsets found!")
        return [], dict(ZERO_STATS)

    maximal_sets = set(frozenset(s) for s in maximal_itemsets['itemsets'])
    print(f"  Found {len(maximal_sets)} maximal itemsets "
          f"(of {len(frequent_itemsets)} frequent itemsets)")

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
        return [], dict(ZERO_STATS)

    print(f"  Found {len(rules_df)} rules before maximal-itemset filtering")

    # Keep only rules whose full itemset (antecedent + consequent) is maximal --
    # this is what discards rules implied by a shorter, non-maximal frequent itemset.
    is_maximal = [
        frozenset(ant) | frozenset(cons) in maximal_sets
        for ant, cons in zip(rules_df['antecedents'], rules_df['consequents'])
    ]
    rules_df = rules_df[is_maximal]

    if len(rules_df) == 0:
        print("  WARNING: No rules survive maximal-itemset filtering!")
        return [], dict(ZERO_STATS)

    print(f"  {len(rules_df)} rules survive maximal-itemset filtering")

    pyaerial_rules = convert_fpgrowth_rules_to_pyaerial_format(rules_df)

    if not compute_stats:
        stats = dict(ZERO_STATS)
        stats['rule_count'] = len(pyaerial_rules)
        return pyaerial_rules, stats

    stats = compute_rule_stats(pyaerial_rules, dataset, feature_names)
    print(f"  Avg support={stats['average_support']:.4f}, "
          f"avg confidence={stats['average_confidence']:.4f}")

    return pyaerial_rules, stats


# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("FP-Max Rule Mining Experiments (maximal-itemset baseline)")
    print("=" * 80)
    print("\nNOTE: FP-Max is deterministic. It runs once per dataset using")
    print("      the same min_support thresholds as fpgrowth_experiments.py.")
    print("=" * 80)

    max_len = 2
    min_confidence = 0.8
    min_support = 0.5

    print("\nLoading datasets...")
    datasets = get_ucimlrepo_datasets(size="small") + get_ucimlrepo_datasets(size="normal")

    os.makedirs("out", exist_ok=True)
    all_results = []

    for dataset_info in datasets:
        dataset_name = dataset_info['name']
        dataset = dataset_info['data']

        print("\n" + "=" * 80)
        print(f"Dataset: {dataset_name}")
        print("=" * 80)
        print(f"Shape: {dataset.shape}")

        process = psutil.Process()
        tracemalloc.start()
        mem_before = process.memory_info().rss / 1024 ** 2

        start_time = time.time()
        extracted_rules, stats = fpmax_rule_learning(
            dataset=dataset,
            min_support=min_support,
            min_confidence=min_confidence,
            max_len=max_len
        )
        end_time = time.time()
        elapsed_time = end_time - start_time

        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_after = process.memory_info().rss / 1024 ** 2
        peak_cpu_memory_mb = max(mem_after - mem_before, peak_mem / 1024 ** 2)

        print(f"\nCompleted in {elapsed_time:.2f} seconds")
        print(f"Extracted {len(extracted_rules)} rules")
        print(f"Peak CPU Memory: {peak_cpu_memory_mb:.2f} MB")

        rules_file = save_rules(
            rules=extracted_rules,
            stats=stats,
            dataset_name=dataset_name,
            method_name="fpmax",
            seed=None
        )
        print(f"Rules saved to {rules_file}")

        result = {
            'dataset': dataset_name,
            'min_support': min_support,
            'num_rules': stats.get('rule_count', len(extracted_rules)),
            'avg_support': stats.get('average_support', 0.0),
            'avg_confidence': stats.get('average_confidence', 0.0),
            'avg_zhangs_metric': stats.get('average_zhangs_metric', 0.0),
            'avg_abs_zhangs_metric': abs(stats.get('average_zhangs_metric', 0.0)),
            'avg_interestingness': stats.get('average_interestingness', 0.0),
            'avg_rule_coverage': stats.get('average_coverage', 0.0),
            'data_coverage': stats.get('data_coverage', 0.0),
            'execution_time': elapsed_time,
            'peak_cpu_memory_mb': peak_cpu_memory_mb
        }
        if len(extracted_rules) > 0:
            print(f"  Support: {result['avg_support']:.4f}")
            print(f"  Confidence: {result['avg_confidence']:.4f}")
            print(f"  Zhang's Metric: {result['avg_zhangs_metric']:.4f}")
            print(f"  Interestingness: {result['avg_interestingness']:.4f}")
            print(f"  Rule coverage: {result['avg_rule_coverage']:.4f}")
            print(f"  Data coverage: {result['data_coverage']:.4f}")
        else:
            print("  WARNING: No rules extracted!")

        all_results.append(result)

    if len(all_results) > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"out/fpmax_{timestamp}.xlsx"

        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            results_df = pd.DataFrame(all_results)
            results_df.to_excel(writer, sheet_name='Results', index=False)

            params_df = pd.DataFrame([{
                'max_antecedents': max_len,
                'min_confidence': min_confidence,
                'min_support': min_support,
                'note': 'Maximal-itemset (FP-Max) baseline: rules restricted to those whose '
                        'full itemset is maximal, mined via mlxtend fpgrowth+fpmax'
            }])
            params_df.to_excel(writer, sheet_name='Parameters', index=False)

        print("\n" + "=" * 80)
        print(f"Results saved to {output_filename}")
        print(f"  - Sheet 1: Results (one row per dataset)")
        print(f"  - Sheet 2: Parameters")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("No results to save.")
        print("=" * 80)
