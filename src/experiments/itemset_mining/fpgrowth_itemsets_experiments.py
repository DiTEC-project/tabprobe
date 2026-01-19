"""
FP-Growth Frequent Itemset Mining Experiments
"""
import time
import os
import numpy as np
import pandas as pd
from datetime import datetime

from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder


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
