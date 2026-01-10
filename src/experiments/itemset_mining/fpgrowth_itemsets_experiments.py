"""
FP-Growth Frequent Itemset Mining Experiments

Classical frequent itemset mining with calibrated min_support from Aerial.
FP-Growth is deterministic and runs once per dataset.
"""
import time
import os
import numpy as np
import pandas as pd
from datetime import datetime

from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from src.utils import (
    get_ucimlrepo_datasets,
    save_itemsets,
    load_fpgrowth_itemset_calibration,
)


def fpgrowth_itemset_learning(dataset, min_support=0.05, max_len=3):
    """
    Extract frequent itemsets using FP-Growth.

    Args:
        dataset: DataFrame with categorical features
        min_support: Minimum support threshold (default: 0.05)
        max_len: Maximum itemset length (default: 3)

    Returns:
        itemsets: List of frequent itemsets in PyAerial format
        stats: Statistics dictionary
    """
    feature_names = list(dataset.columns)

    print(f"Dataset shape: {dataset.shape}")
    print(f"Number of features: {len(feature_names)}")

    # Convert DataFrame to transaction format
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
    print(f"  max_len={max_len}")

    # Run FP-Growth to find frequent itemsets
    frequent_itemsets_df = fpgrowth(
        df_encoded,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len
    )

    if len(frequent_itemsets_df) == 0:
        print("  WARNING: No frequent itemsets found!")
        return [], {'itemset_count': 0, 'average_support': 0.0}

    print(f"  Found {len(frequent_itemsets_df)} frequent itemsets")

    # Convert to PyAerial format
    itemsets = []
    for _, row in frequent_itemsets_df.iterrows():
        itemset = []
        for item in row['itemsets']:
            feature, value = item.split('=')
            itemset.append({'feature': feature, 'value': value})

        itemsets.append({
            'itemset': itemset,
            'support': row['support']
        })

    # Calculate statistics
    stats = {
        'itemset_count': len(itemsets),
        'average_support': float(np.mean([i['support'] for i in itemsets]))
    }

    return itemsets, stats


# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("FP-Growth Frequent Itemset Mining Experiments")
    print("=" * 80)

    # FP-Growth is deterministic - no seeds
    max_len = 3

    # Load datasets
    print("\nLoading datasets...")
    datasets = get_ucimlrepo_datasets(size="small")  # Match Aerial's dataset size

    # Create output directory
    os.makedirs("out/frequent_itemsets", exist_ok=True)

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

        # Load calibration threshold
        try:
            calibration = load_fpgrowth_itemset_calibration(
                dataset_name=dataset_name,
                reference_method="aerial"
            )
            min_support = calibration['min_support']
            print(f"Using calibrated min_support={min_support:.4f} (covers 90% of Aerial itemsets)")
        except FileNotFoundError:
            # Fallback to default if calibration not available
            min_support = 0.05
            print(f"WARNING: Calibration not found, using default min_support={min_support}")

        # Track time
        start_time = time.time()

        # Extract itemsets
        extracted_itemsets, stats = fpgrowth_itemset_learning(
            dataset=dataset,
            min_support=min_support,
            max_len=max_len
        )

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"\nCompleted in {elapsed_time:.2f} seconds")
        print(f"Extracted {len(extracted_itemsets)} frequent itemsets")

        # Save itemsets (no seed for deterministic method)
        if len(extracted_itemsets) > 0:
            itemsets_file = save_itemsets(
                itemsets=extracted_itemsets,
                stats=stats,
                dataset_name=dataset_name,
                method_name="fpgrowth",
                seed=None  # Deterministic, no seed
            )
            print(f"Itemsets saved to {itemsets_file}")

            print(f"  Itemset count: {stats['itemset_count']}")
            print(f"  Average support: {stats['average_support']:.4f}")

            result = {
                'dataset': dataset_name,
                'num_itemsets': stats['itemset_count'],
                'avg_support': stats['average_support'],
                'min_support_threshold': min_support,
                'execution_time': elapsed_time
            }
        else:
            print("  WARNING: No itemsets extracted!")
            result = {
                'dataset': dataset_name,
                'num_itemsets': 0,
                'avg_support': 0.0,
                'min_support_threshold': min_support,
                'execution_time': elapsed_time
            }

        all_results.append(result)

    # Save results to Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"out/fpgrowth_itemsets_{timestamp}.xlsx"

    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        # Sheet 1: Results (one row per dataset)
        results_df = pd.DataFrame(all_results)
        results_df.to_excel(writer, sheet_name='Results', index=False)

        # Sheet 2: Parameters
        params_df = pd.DataFrame([{
            'max_len': max_len,
            'calibration_method': 'aerial',
            'calibration_coverage': 0.9,
            'deterministic': True
        }])
        params_df.to_excel(writer, sheet_name='Parameters', index=False)

    print("\n" + "=" * 80)
    print(f"Results saved to {output_filename}")
    print("=" * 80)