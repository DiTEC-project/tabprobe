"""
Rule Mining with Tabular Foundation Models

A library for extracting association rules from categorical data using
tabular foundation models (TabPFN, TabICL, TabDPT).

Example usage:
    from ucimlrepo import fetch_ucirepo
    from src.rulemining import RuleMiner

    # Load data
    dataset = fetch_ucirepo(id=17)  # breast_cancer
    df = dataset.data.features

    # Mine rules with TabPFN
    miner = RuleMiner(method='tabpfn')
    rules = miner.fit(df)
"""

from .rule_miner import RuleMiner

__all__ = ['RuleMiner']
__version__ = '0.1.0'