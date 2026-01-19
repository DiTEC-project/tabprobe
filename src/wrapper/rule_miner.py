"""
RuleMiner: Unified interface for association rule mining with tabular foundation models.
"""
import gc
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.utils.data_prep import prepare_categorical_data, add_gaussian_noise
from src.utils.test_matrix import generate_test_matrix
from src.utils.rule_extraction import extract_rules_from_reconstruction
from src.utils.rule_quality import calculate_rule_metrics


def discretize_numerical(
    data: pd.DataFrame,
    n_bins: int = 5,
    categorical_threshold: int = 10
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply equal-frequency binning to numerical columns.

    Args:
        data: Input DataFrame with mixed types.
        n_bins: Number of bins for discretization.
        categorical_threshold: Columns with fewer unique values than this are treated as categorical.

    Returns:
        Tuple of (discretized DataFrame, list of discretized column names)
    """
    result = data.copy()
    discretized_cols = []

    for col in result.columns:
        if result[col].dtype in ['object', 'category', 'bool']:
            continue

        n_unique = result[col].nunique()
        if n_unique <= categorical_threshold:
            result[col] = result[col].astype(str)
            continue

        try:
            result[col] = pd.qcut(
                result[col],
                q=n_bins,
                labels=[f"Q{i+1}" for i in range(n_bins)],
                duplicates='drop'
            )
            discretized_cols.append(col)
        except ValueError:
            result[col] = result[col].astype(str)

    for col in result.columns:
        result[col] = result[col].astype(str)

    return result, discretized_cols


class RuleMiner:
    """
    Unified interface for association rule mining using tabular foundation models.

    Supported methods:
        - 'tabpfn': TabPFN
        - 'tabicl': TabICL
        - 'tabdpt': TabDPT

    Example:
        >>> from ucimlrepo import fetch_ucirepo
        >>> from src.wrapper import RuleMiner
        >>>
        >>> dataset = fetch_ucirepo(id=244)  # fertility
        >>> df = dataset.data.features
        >>>
        >>> miner = RuleMiner(method='tabicl', n_estimators=2, max_antecedents=2)
        >>> rules = miner.mine_rules(df)
        >>>
        >>> for rule in rules[:5]:
        ...     ant = ' & '.join([f"{a['feature']}={a['value']}" for a in rule['antecedents']])
        ...     cons = f"{rule['consequent']['feature']}={rule['consequent']['value']}"
        ...     print(f"{ant} -> {cons} (conf: {rule['confidence']:.3f})")
        >>>
        >>> rules_df = miner.to_dataframe()
    """

    SUPPORTED_METHODS = ['tabpfn', 'tabicl', 'tabdpt']

    def __init__(
        self,
        method: str = 'tabicl',
        max_antecedents: int = 2,
        ant_similarity: float = 0.5,
        cons_similarity: float = 0.8,
        n_estimators: int = 8,
        noise_factor: float = 0.5,
        n_bins: int = 5,
        random_state: int = 42,
    ):
        """
        Initialize the RuleMiner.

        Args:
            method: Foundation model to use for rule mining.
                    Options: 'tabpfn', 'tabicl', 'tabdpt'
            max_antecedents: Maximum number of items in the rule antecedent.
                             Higher values find more complex rules but increase computation.
            ant_similarity: Similarity threshold for antecedent validation (0.0 to 1.0).
                            Lower values extract more rules but may include weaker patterns.
            cons_similarity: Similarity threshold for consequent extraction (0.0 to 1.0).
                             Higher values extract only high-confidence consequents.
            n_estimators: Number of ensemble models for prediction averaging.
                          Higher values improve stability but increase computation.
            noise_factor: Gaussian noise factor added to context data (0.0 to 1.0).
                          Helps prevent overfitting to exact patterns.
            n_bins: Number of bins for discretizing numerical columns (equal-frequency binning).
            random_state: Random seed for reproducibility.
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unknown method: '{method}'. Supported: {self.SUPPORTED_METHODS}")

        self.method = method
        self.max_antecedents = max_antecedents
        self.ant_similarity = ant_similarity
        self.cons_similarity = cons_similarity
        self.n_estimators = n_estimators
        self.noise_factor = noise_factor
        self.n_bins = n_bins
        self.random_state = random_state

        self.rules_: Optional[List[Dict]] = None
        self.feature_names_: Optional[List[str]] = None
        self.encoder_ = None
        self._discretized_data: Optional[pd.DataFrame] = None
        self._discretized_cols: Optional[List[str]] = None

    def mine_rules(self, data: pd.DataFrame, compute_metrics: bool = True) -> List[Dict]:
        """
        Extract association rules from data.

        Numerical columns are automatically discretized using equal-frequency binning.

        Args:
            data: DataFrame with features. Numerical columns will be discretized automatically.
            compute_metrics: If True, compute quality metrics (support, confidence, etc.) for each rule.

        Returns:
            List of rules. Each rule is a dictionary with:
                - 'antecedents': List of conditions, e.g., [{'feature': 'age', 'value': 'Q3'}, ...]
                - 'consequent': Single condition, e.g., {'feature': 'class', 'value': 'positive'}
                - 'support': Rule support (if compute_metrics=True)
                - 'confidence': Rule confidence (if compute_metrics=True)
                - 'rule_coverage': Antecedent support (if compute_metrics=True)
                - 'zhangs_metric': Zhang's correlation metric (if compute_metrics=True)
                - 'interestingness': Interestingness score (if compute_metrics=True)
        """
        discretized_data, discretized_cols = discretize_numerical(data, n_bins=self.n_bins)
        self._discretized_data = discretized_data
        self._discretized_cols = discretized_cols

        encoded_data, classes_per_feature, feature_names, encoder = prepare_categorical_data(discretized_data)
        self.feature_names_ = feature_names
        self.encoder_ = encoder

        test_matrix, test_descriptions, feature_value_indices = generate_test_matrix(
            n_features=len(classes_per_feature),
            classes_per_feature=classes_per_feature,
            max_antecedents=self.max_antecedents,
            use_zeros_for_unmarked=False
        )

        noisy_context = add_gaussian_noise(encoded_data, noise_factor=self.noise_factor)

        reconstruction_probs = self._reconstruct(
            noisy_context, encoded_data, test_matrix, feature_value_indices
        )

        rules = extract_rules_from_reconstruction(
            prob_matrix=reconstruction_probs,
            test_descriptions=test_descriptions,
            feature_value_indices=feature_value_indices,
            ant_similarity=self.ant_similarity,
            cons_similarity=self.cons_similarity,
            feature_names=feature_names,
            encoder=encoder
        )

        self.rules_ = rules

        if compute_metrics and len(rules) > 0:
            self.rules_ = self._compute_metrics(rules, discretized_data)

        return self.rules_

    def _reconstruct(self, noisy_context, clean_context, query_matrix, feature_value_indices):
        """Route to the appropriate reconstruction method."""
        if self.method == 'tabpfn':
            return self._reconstruct_tabpfn(noisy_context, clean_context, query_matrix, feature_value_indices)
        elif self.method == 'tabicl':
            return self._reconstruct_tabicl(noisy_context, clean_context, query_matrix, feature_value_indices)
        else:  # tabdpt
            return self._reconstruct_tabdpt(noisy_context, clean_context, query_matrix, feature_value_indices)

    def _reconstruct_tabpfn(self, noisy_context, clean_context, query_matrix, feature_value_indices):
        from tabpfn import TabPFNClassifier

        n_queries = query_matrix.shape[0]
        n_features_total = query_matrix.shape[1]
        reconstruction_probs = np.zeros((n_queries, n_features_total))

        for feat_info in feature_value_indices:
            start_idx, end_idx = feat_info['start'], feat_info['end']
            n_classes = end_idx - start_idx

            x_context = np.delete(noisy_context, range(start_idx, end_idx), axis=1)
            y_context = np.argmax(clean_context[:, start_idx:end_idx], axis=1)

            if len(np.unique(y_context)) < 2:
                reconstruction_probs[:, start_idx:end_idx] = 1.0 / n_classes
                continue

            model = TabPFNClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
            model.fit(x_context, y_context)

            x_query = np.delete(query_matrix, range(start_idx, end_idx), axis=1)
            probs = model.predict_proba(x_query)

            reconstruction_probs[:, start_idx:end_idx] = self._align_probs(probs, n_queries, n_classes)

        return reconstruction_probs

    def _reconstruct_tabicl(self, noisy_context, clean_context, query_matrix, feature_value_indices):
        from tabicl import TabICLClassifier

        n_queries = query_matrix.shape[0]
        n_features_total = query_matrix.shape[1]
        reconstruction_probs = np.zeros((n_queries, n_features_total))

        for feat_info in feature_value_indices:
            start_idx, end_idx = feat_info['start'], feat_info['end']
            n_classes = end_idx - start_idx

            x_context = np.delete(noisy_context, range(start_idx, end_idx), axis=1)
            y_context = np.argmax(clean_context[:, start_idx:end_idx], axis=1)

            if len(np.unique(y_context)) < 2:
                reconstruction_probs[:, start_idx:end_idx] = 1.0 / n_classes
                continue

            model = TabICLClassifier(random_state=self.random_state, n_estimators=self.n_estimators)
            model.fit(x_context, y_context)

            x_query = np.delete(query_matrix, range(start_idx, end_idx), axis=1)
            probs = model.predict_proba(x_query)

            reconstruction_probs[:, start_idx:end_idx] = self._align_probs(probs, n_queries, n_classes)

        return reconstruction_probs

    def _reconstruct_tabdpt(self, noisy_context, clean_context, query_matrix, feature_value_indices):
        from tabdpt import TabDPTClassifier

        n_queries = query_matrix.shape[0]
        n_features_total = query_matrix.shape[1]
        reconstruction_probs = np.zeros((n_queries, n_features_total))
        context_size = len(clean_context)

        for feat_info in feature_value_indices:
            start_idx, end_idx = feat_info['start'], feat_info['end']
            n_classes = end_idx - start_idx

            x_context = np.delete(noisy_context, range(start_idx, end_idx), axis=1)
            y_context = np.argmax(clean_context[:, start_idx:end_idx], axis=1)

            if len(np.unique(y_context)) < 2:
                reconstruction_probs[:, start_idx:end_idx] = 1.0 / n_classes
                continue

            model = TabDPTClassifier()
            model.fit(x_context, y_context)

            x_query = np.delete(query_matrix, range(start_idx, end_idx), axis=1)
            probs = model.ensemble_predict_proba(
                x_query,
                n_ensembles=self.n_estimators,
                context_size=context_size,
                seed=self.random_state
            )

            reconstruction_probs[:, start_idx:end_idx] = self._align_probs(probs, n_queries, n_classes)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        return reconstruction_probs

    def _align_probs(self, probs: np.ndarray, n_queries: int, n_classes: int) -> np.ndarray:
        """Align probability matrix dimensions when model returns fewer classes."""
        if probs.shape[1] == n_classes:
            return probs
        aligned = np.zeros((n_queries, n_classes))
        aligned[:, :min(probs.shape[1], n_classes)] = probs[:, :min(probs.shape[1], n_classes)]
        return aligned

    def _compute_metrics(self, rules: List[Dict], data: pd.DataFrame) -> List[Dict]:
        """Compute quality metrics for extracted rules."""
        rules_with_metrics, _ = calculate_rule_metrics(
            rules=rules,
            data=data.values,
            feature_names=self.feature_names_
        )
        return rules_with_metrics

    def get_rules(self) -> List[Dict]:
        """Return the extracted rules."""
        return self.rules_ if self.rules_ is not None else []

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert extracted rules to a pandas DataFrame.

        Returns:
            DataFrame with columns: antecedents, consequent, support, confidence,
            rule_coverage, zhangs_metric, interestingness
        """
        if not self.rules_:
            return pd.DataFrame()

        rows = []
        for rule in self.rules_:
            ant_str = ' & '.join([f"{a['feature']}={a['value']}" for a in rule['antecedents']])
            cons_str = f"{rule['consequent']['feature']}={rule['consequent']['value']}"

            rows.append({
                'antecedents': ant_str,
                'consequent': cons_str,
                'support': rule.get('support'),
                'confidence': rule.get('confidence'),
                'rule_coverage': rule.get('rule_coverage'),
                'zhangs_metric': rule.get('zhangs_metric'),
                'interestingness': rule.get('interestingness'),
            })

        return pd.DataFrame(rows)