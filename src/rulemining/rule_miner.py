"""
RuleMiner: Unified interface for association rule mining with tabular foundation models.
"""
import gc
import numpy as np
import pandas as pd
import torch

from src.utils.data_prep import prepare_categorical_data, add_gaussian_noise
from src.utils.test_matrix import generate_test_matrix
from src.utils.rule_extraction import extract_rules_from_reconstruction
from src.utils.rule_quality import calculate_rule_metrics


class RuleMiner:
    """
    Unified interface for association rule mining using tabular foundation models.

    Supports:
        - 'tabpfn': TabPFN (Prior-Data Fitted Networks)
        - 'tabicl': TabICL (In-Context Learning)
        - 'tabdpt': TabDPT (Deep Tabular)

    Example:
        >>> from ucimlrepo import fetch_ucirepo
        >>> dataset = fetch_ucirepo(id=17)  # breast_cancer
        >>> df = dataset.data.features
        >>> miner = RuleMiner(method='tabpfn')
        >>> rules = miner.fit(df)
        >>> for rule in rules[:5]:
        ...     print(f"{rule['antecedents']} -> {rule['consequent']}")
    """

    SUPPORTED_METHODS = ['tabpfn', 'tabicl', 'tabdpt']

    def __init__(
        self,
        method: str = 'tabpfn',
        max_antecedents: int = 2,
        ant_similarity: float = 0.5,
        cons_similarity: float = 0.8,
        n_estimators: int = 8,
        noise_factor: float = 0.5,
        random_state: int = 42,
    ):
        """
        Initialize the RuleMiner.

        Args:
            method: Mining method ('tabpfn', 'tabicl', 'tabdpt')
            max_antecedents: Maximum number of items in antecedent (default: 2)
            ant_similarity: Threshold for antecedent validation (default: 0.5)
            cons_similarity: Threshold for consequent extraction (default: 0.8)
            n_estimators: Number of ensemble models (default: 8)
            noise_factor: Gaussian noise factor for context (default: 0.5)
            random_state: Random seed for reproducibility (default: 42)
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unknown method: {method}. Supported: {self.SUPPORTED_METHODS}")

        self.method = method
        self.max_antecedents = max_antecedents
        self.ant_similarity = ant_similarity
        self.cons_similarity = cons_similarity
        self.n_estimators = n_estimators
        self.noise_factor = noise_factor
        self.random_state = random_state

        self.rules_ = None
        self.feature_names_ = None
        self.encoder_ = None

    def fit(self, data: pd.DataFrame, calculate_metrics: bool = True) -> list:
        """
        Extract association rules from the data.

        Args:
            data: DataFrame with categorical features
            calculate_metrics: Whether to calculate quality metrics (support, confidence, etc.)

        Returns:
            List of rules, where each rule is a dict with:
                - 'antecedents': List of {'feature': str, 'value': str/int}
                - 'consequent': {'feature': str, 'value': str/int}
                - 'support': float (if calculate_metrics=True)
                - 'confidence': float (if calculate_metrics=True)
                - 'zhangs_metric': float (if calculate_metrics=True)
                - 'interestingness': float (if calculate_metrics=True)
        """
        # Prepare data
        encoded_data, classes_per_feature, feature_names, encoder = prepare_categorical_data(data)
        self.feature_names_ = feature_names
        self.encoder_ = encoder

        # Generate test matrix
        test_matrix, test_descriptions, feature_value_indices = generate_test_matrix(
            n_features=len(classes_per_feature),
            classes_per_feature=classes_per_feature,
            max_antecedents=self.max_antecedents,
            use_zeros_for_unmarked=False
        )

        # Add noise to context
        noisy_context = add_gaussian_noise(encoded_data, noise_factor=self.noise_factor)

        # Get reconstruction probabilities
        if self.method == 'tabpfn':
            reconstruction_probs = self._reconstruct_with_tabpfn(
                noisy_context, encoded_data, test_matrix, feature_value_indices
            )
        elif self.method == 'tabicl':
            reconstruction_probs = self._reconstruct_with_tabicl(
                noisy_context, encoded_data, test_matrix, feature_value_indices
            )
        elif self.method == 'tabdpt':
            reconstruction_probs = self._reconstruct_with_tabdpt(
                noisy_context, encoded_data, test_matrix, feature_value_indices
            )

        # Extract rules
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

        if calculate_metrics and len(rules) > 0:
            rules = self._calculate_metrics(rules, data)
            self.rules_ = rules

        return rules

    def _reconstruct_with_tabpfn(self, noisy_context, clean_context, query_matrix, feature_value_indices):
        """Reconstruct using TabPFN."""
        from tabpfn import TabPFNClassifier

        n_queries = query_matrix.shape[0]
        n_features_total = query_matrix.shape[1]
        reconstruction_probs = np.zeros((n_queries, n_features_total))

        for feat_idx, feat_info in enumerate(feature_value_indices):
            start_idx = feat_info['start']
            end_idx = feat_info['end']
            n_classes = end_idx - start_idx

            x_context = np.delete(noisy_context, range(start_idx, end_idx), axis=1)
            y_context_onehot = clean_context[:, start_idx:end_idx]
            y_context = np.argmax(y_context_onehot, axis=1)

            # Skip constant features
            if len(np.unique(y_context)) < 2:
                reconstruction_probs[:, start_idx:end_idx] = 1.0 / n_classes
                continue

            model = TabPFNClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state
            )
            model.fit(x_context, y_context)

            x_query = np.delete(query_matrix, range(start_idx, end_idx), axis=1)
            probs = model.predict_proba(x_query)

            if probs.shape[1] != n_classes:
                proper_probs = np.zeros((n_queries, n_classes))
                min_cols = min(probs.shape[1], n_classes)
                proper_probs[:, :min_cols] = probs[:, :min_cols]
                reconstruction_probs[:, start_idx:end_idx] = proper_probs
            else:
                reconstruction_probs[:, start_idx:end_idx] = probs

        return reconstruction_probs

    def _reconstruct_with_tabicl(self, noisy_context, clean_context, query_matrix, feature_value_indices):
        """Reconstruct using TabICL."""
        from tabicl import TabICLClassifier

        n_queries = query_matrix.shape[0]
        n_features_total = query_matrix.shape[1]
        reconstruction_probs = np.zeros((n_queries, n_features_total))

        for feat_idx, feat_info in enumerate(feature_value_indices):
            start_idx = feat_info['start']
            end_idx = feat_info['end']
            n_classes = end_idx - start_idx

            x_context = np.delete(noisy_context, range(start_idx, end_idx), axis=1)
            y_context_onehot = clean_context[:, start_idx:end_idx]
            y_context = np.argmax(y_context_onehot, axis=1)

            if len(np.unique(y_context)) < 2:
                reconstruction_probs[:, start_idx:end_idx] = 1.0 / n_classes
                continue

            model = TabICLClassifier(
                random_state=self.random_state,
                n_estimators=self.n_estimators
            )
            model.fit(x_context, y_context)

            x_query = np.delete(query_matrix, range(start_idx, end_idx), axis=1)
            probs = model.predict_proba(x_query)

            if probs.shape[1] != n_classes:
                proper_probs = np.zeros((n_queries, n_classes))
                min_cols = min(probs.shape[1], n_classes)
                proper_probs[:, :min_cols] = probs[:, :min_cols]
                reconstruction_probs[:, start_idx:end_idx] = proper_probs
            else:
                reconstruction_probs[:, start_idx:end_idx] = probs

        return reconstruction_probs

    def _reconstruct_with_tabdpt(self, noisy_context, clean_context, query_matrix, feature_value_indices):
        """Reconstruct using TabDPT."""
        from tabdpt import TabDPTClassifier

        n_queries = query_matrix.shape[0]
        n_features_total = query_matrix.shape[1]
        reconstruction_probs = np.zeros((n_queries, n_features_total))
        context_size = len(clean_context)

        for feat_idx, feat_info in enumerate(feature_value_indices):
            start_idx = feat_info['start']
            end_idx = feat_info['end']
            n_classes = end_idx - start_idx

            x_context = np.delete(noisy_context, range(start_idx, end_idx), axis=1)
            y_context_onehot = clean_context[:, start_idx:end_idx]
            y_context = np.argmax(y_context_onehot, axis=1)

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

            if probs.shape[1] != n_classes:
                proper_probs = np.zeros((n_queries, n_classes))
                min_cols = min(probs.shape[1], n_classes)
                proper_probs[:, :min_cols] = probs[:, :min_cols]
                reconstruction_probs[:, start_idx:end_idx] = proper_probs
            else:
                reconstruction_probs[:, start_idx:end_idx] = probs

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        return reconstruction_probs

    def _calculate_metrics(self, rules: list, data: pd.DataFrame) -> list:
        """Calculate quality metrics for each rule."""
        data_array = data.values
        rules_with_metrics, _ = calculate_rule_metrics(
            rules=rules,
            data=data_array,
            feature_names=self.feature_names_
        )
        return rules_with_metrics

    def get_rules(self) -> list:
        """Return the extracted rules."""
        return self.rules_ if self.rules_ is not None else []

    def rules_to_dataframe(self) -> pd.DataFrame:
        """Convert rules to a pandas DataFrame for easier analysis."""
        if not self.rules_:
            return pd.DataFrame()

        rows = []
        for rule in self.rules_:
            ant_str = ' & '.join([f"{a['feature']}={a['value']}" for a in rule['antecedents']])
            cons_str = f"{rule['consequent']['feature']}={rule['consequent']['value']}"

            row = {
                'antecedents': ant_str,
                'consequent': cons_str,
                'support': rule.get('support', None),
                'confidence': rule.get('confidence', None),
                'zhangs_metric': rule.get('zhangs_metric', None),
                'interestingness': rule.get('interestingness', None)
            }
            rows.append(row)

        return pd.DataFrame(rows)