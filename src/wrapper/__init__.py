"""
Rule Mining with Tabular Foundation Models

A library for extracting association rules from categorical data using
tabular foundation models (TabPFN, TabICL, TabDPT).

Example usage:
    from ucimlrepo import fetch_ucirepo
    from src.wrapper import TabProbe

    # Load data
    dataset = fetch_ucirepo(id=14)  # breast_cancer
    df = dataset.data.features

    # Mine rules with TabPFN
    miner = TabProbe(method='tabpfn')
    rules = miner.fit(df)
"""

from src.wrapper.tabprobe import TabProbe

__all__ = ['TabProbe']
__version__ = '0.1.0'
