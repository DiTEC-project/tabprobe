#!/usr/bin/env python
import time
import os
import json
import sys
import warnings
from contextlib import contextmanager

# Suppress all warnings before any imports
os.environ['TQDM_DISABLE'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


# Redirect stderr to suppress deprecation warnings from TabDPT
@contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output."""
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from tabpfn import TabPFNClassifier

# Additional warning suppression after imports
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
if hasattr(np, 'warnings'):
    np.warnings.filterwarnings('ignore')

from src.experiments.classification.corels.corels_main import (
    create_corels_freq_items_input,
    create_corels_input_files,
    run_corels,
    parse_corels_rule_lists
)

from src.utils import (
    get_ucimlrepo_datasets,
    generate_seed_sequence,
    save_reconstruction_probs,
    load_reconstruction_probs,
    reconstruction_probs_exist,
    prepare_categorical_data,
    add_gaussian_noise,
    generate_test_matrix,
    extract_frequent_itemsets_from_reconstruction
)

# Import itemset mining functions
from src.experiments.itemset_mining.aerial_itemsets_experiments import aerial_itemset_learning
from src.experiments.itemset_mining.tabpfn_itemsets_experiments import adapt_tabpfn_for_reconstruction
from src.experiments.itemset_mining.tabicl_itemsets_experiments import adapt_tabicl_for_reconstruction
from src.experiments.itemset_mining.tabdpt_itemsets_experiments import adapt_tabdpt_for_reconstruction
from src.experiments.itemset_mining.fpgrowth_itemsets_experiments import fpgrowth_itemset_learning

# Class column names for each dataset (always the last column in original CSV files)
DATASET_CLASS_COLUMNS = {
    'breast_cancer': 'Class',
    'congressional_voting': 'Class',
    'mushroom': 'poisonous',
    'chess_king_rook_vs_king_pawn': 'wtoeg',
    'spambase': 'Class',
    # 'lung_cancer': 'class',
    'hepatitis': 'Class',
    # 'breast_cancer_coimbra': 'Classification',
    'cervical_cancer_behavior_risk': 'ca_cervix',
    'autism_screening_adolescent': 'Class/ASD',
    'acute_inflammations': 'bladder-inflammation',
    'fertility': 'diagnosis',
}


def filter_non_class_itemsets(itemsets, class_column_name):
    """
    Filter itemsets to exclude those containing the class column.
    CORELS requires feature-only itemsets (no class in itemsets).
    """
    non_class_itemsets = []
    for itemset_data in itemsets:
        # Check if class column is in this itemset
        contains_class = any(item['feature'] == class_column_name for item in itemset_data['itemset'])
        if not contains_class and len(itemset_data['itemset']) > 0:
            non_class_itemsets.append(itemset_data)
    return non_class_itemsets


def prepare_corels_data_from_itemsets(itemsets, transactions, class_column_name):
    """
    Convert frequent itemsets to CORELS input format.

    CORELS expects:
    - Frequent itemsets of FEATURES ONLY (excluding class)
    - For each itemset, a binary occurrence vector showing which rows match the pattern
    """
    rules = []

    # Remove class column from transactions for matching
    X_transactions = transactions.drop(columns=[class_column_name])

    for itemset_data in itemsets:
        # Convert itemset list to dict format
        itemset_dict = {}
        for item in itemset_data['itemset']:
            itemset_dict[item['feature']] = item['value']

        # Skip empty itemsets
        if len(itemset_dict) == 0:
            continue

        # Create CORELS format row using existing function
        corels_row = create_corels_freq_items_input(itemset_dict, X_transactions)

        # corels_row format: ["{feature1:=value1,feature2:=value2}", 1, 0, 1, ...]
        occurrence_vector = np.array(corels_row[1:])

        # Skip itemsets that don't match any rows
        if np.sum(occurrence_vector) == 0:
            continue

        rules.append(corels_row)

    return rules


def convert_corels_predictions_to_sklearn_format(corels_model, X, y_encoded, class_column_name):
    """
    Convert CORELS model predictions to sklearn-compatible format.
    """
    # Reset indices
    X_reset = X.reset_index(drop=True)
    y_reset = y_encoded.reset_index(drop=True) if isinstance(y_encoded, pd.Series) else y_encoded

    # Normalize column names to match CORELS format (spaces → dashes)
    X_normalized = X_reset.rename(columns=lambda x: x.replace(' ', '-'))

    # Get true labels
    y_true = y_reset if isinstance(y_reset, np.ndarray) else y_reset.values

    # Get default rule
    default_rule = next((rule[-1][1] for rule in corels_model if rule[0][0] == 'default'), None)

    # Predict for each instance
    y_pred = []

    for i, feature_row in X_normalized.iterrows():
        predicted_label = None

        # Iterate through conditions
        for condition in corels_model:
            # Skip default rule
            if condition[0][0] == 'default':
                continue

            # Extract conditions and expected label
            conditions = condition[:-1]
            expected_label = condition[-1]

            # Check if all conditions are satisfied
            if all(feature_row[key] == value for key, value in conditions):
                predicted_label = expected_label
                break

        # Use default if no match
        if predicted_label is None and default_rule is not None:
            predicted_label = default_rule

        y_pred.append(predicted_label)

    return np.array(y_true), np.array(y_pred)


def get_aerial_dataset_parameters(dataset_name, dataset_size='small'):
    """
    Get dataset-specific Aerial training parameters.

    Args:
        dataset_name: Name of the dataset
        dataset_size: Size category ('normal' or 'small')

    Returns:
        dict: Parameters (batch_size, layer_dims, epochs)
    """
    # Dataset-specific parameter overrides
    DATASET_PARAMS = {
        'breast_cancer': {
            'batch_size': 2,
            'layer_dims': [4],
            'epochs': 5
        },
        'congressional_voting': {
            'batch_size': 4,
            'layer_dims': [2],
            'epochs': 2
        },
        'cervical_cancer_behavior_risk': {
            'batch_size': 1,
            'layer_dims': [8],
            'epochs': 20
        }
    }

    # Default parameters by dataset size
    DEFAULT_PARAMS = {
        'normal': {
            'batch_size': 64,
            'layer_dims': [4],
            'epochs': 2
        },
        'small': {
            'batch_size': 2,
            'layer_dims': [4],
            'epochs': 10
        }
    }

    # Check for dataset-specific override first
    if dataset_name in DATASET_PARAMS:
        return DATASET_PARAMS[dataset_name]

    # Otherwise use default for the dataset size
    return DEFAULT_PARAMS.get(dataset_size, DEFAULT_PARAMS['normal'])


def save_corels_model(corels_model, method, dataset_name, seed, fold_idx=None, output_dir="out/classifiers"):
    """Save CORELS model to JSON file."""
    method_dir = os.path.join(output_dir, "corels", method, dataset_name)
    os.makedirs(method_dir, exist_ok=True)

    if seed is None and fold_idx is None:
        output_file = os.path.join(method_dir, "classifier.json")
    elif fold_idx is None:
        output_file = os.path.join(method_dir, f"seed_{seed}.json")
    elif seed is None:
        output_file = os.path.join(method_dir, f"fold_{fold_idx}.json")
    else:
        output_file = os.path.join(method_dir, f"seed_{seed}_fold_{fold_idx}.json")

    # Convert CORELS model to serializable format
    model_data = {
        'dataset': dataset_name,
        'method': method,
        'seed': seed,
        'fold': fold_idx,
        'rules': []
    }

    for condition in corels_model:
        if condition[0][0] == 'default':
            model_data['default_class'] = condition[0][1]
        else:
            rule_dict = {
                'conditions': [{'feature': key, 'value': val} for key, val in condition[:-1]],
                'class': condition[-1]
            }
            model_data['rules'].append(rule_dict)

    with open(output_file, 'w') as f:
        json.dump(model_data, f, indent=2)

    return output_file


def mine_itemsets_for_method(method, train_data, dataset_name, hyperparams, seed, fold_idx):
    """
    Mine frequent itemsets from training data using specified method.
    For tabular foundation models (TabPFN, TabICL, TabDPT), uses caching to avoid
    re-running expensive model inference when experimenting with different similarity thresholds.

    Args:
        method: Method name ('aerial', 'tabpfn', 'tabicl', 'tabdpt', 'fpgrowth_X')
        train_data: Training dataframe
        dataset_name: Name of the dataset
        hyperparams: Hyperparameters for the method
        seed: Random seed for this run
        fold_idx: Fold index (1-based)

    Returns:
        List of mined frequent itemsets
    """
    # Methods that use reconstruction probability caching
    tabular_foundation_models = ['tabpfn', 'tabicl', 'tabdpt']

    if method == 'aerial':
        # Aerial doesn't use caching (runs fast enough)
        itemsets, _ = aerial_itemset_learning(
            dataset=train_data,
            max_length=hyperparams['max_length'],
            similarity=hyperparams['similarity'],
            batch_size=hyperparams['batch_size'],
            layer_dims=hyperparams['layer_dims'],
            epochs=hyperparams['epochs'],
            random_state=hyperparams.get('random_state', 42)
        )

    elif method in tabular_foundation_models:
        # Check if reconstruction_probs are cached
        if reconstruction_probs_exist(dataset_name, method, seed, fold_idx):
            print(f"[CACHE HIT] Loading cached reconstruction_probs...", end=' ')
            reconstruction_probs, metadata = load_reconstruction_probs(dataset_name, method, seed, fold_idx)

            # Prepare encoder from current training data
            # The encoder is needed to convert numeric values back to original categorical values
            _, _, _, encoder = prepare_categorical_data(train_data)

            # Extract itemsets using current similarity threshold
            result = extract_frequent_itemsets_from_reconstruction(
                prob_matrix=reconstruction_probs,
                test_descriptions=metadata['test_descriptions'],
                feature_value_indices=metadata['feature_value_indices'],
                data=train_data,
                similarity=hyperparams['similarity'],
                feature_names=metadata['feature_names'],
                encoder=encoder
            )
            itemsets = result['itemsets']
            print(f"Extracted {len(itemsets)} itemsets")

        else:
            print(f"[CACHE MISS] Computing reconstruction_probs...", end=' ')

            # Prepare data for reconstruction
            encoded_data, classes_per_feature, feature_names, encoder = prepare_categorical_data(train_data)

            # Generate test matrix
            test_matrix, test_descriptions, feature_value_indices = generate_test_matrix(
                n_features=len(classes_per_feature),
                classes_per_feature=classes_per_feature,
                max_antecedents=hyperparams['max_length'],
                use_zeros_for_unmarked=False
            )

            # Compute reconstruction probabilities based on method
            if method == 'tabpfn':
                # Initialize TabPFN model
                tabpfn_model = TabPFNClassifier(
                    n_estimators=hyperparams.get('n_estimators', 8),
                    random_state=hyperparams.get('random_state', 42),
                    average_before_softmax=True,
                    inference_precision='auto'
                )
                reconstruction_probs = adapt_tabpfn_for_reconstruction(
                    tabpfn_model=tabpfn_model,
                    context_table=encoded_data,
                    query_matrix=test_matrix,
                    feature_value_indices=feature_value_indices,
                    n_samples=hyperparams.get('context_samples', None)
                )
            elif method == 'tabicl':
                reconstruction_probs = adapt_tabicl_for_reconstruction(
                    context_table=encoded_data,
                    query_matrix=test_matrix,
                    feature_value_indices=feature_value_indices,
                    n_samples=hyperparams.get('context_samples', None),
                    n_estimators=hyperparams.get('n_estimators', 8)
                )
            elif method == 'tabdpt':
                # suppress, otherwise tabdpt prints out sooo many warnings that can't suppressed otherwise
                with suppress_stderr():
                    reconstruction_probs = adapt_tabdpt_for_reconstruction(
                        context_table=encoded_data,
                        query_matrix=test_matrix,
                        feature_value_indices=feature_value_indices,
                        n_samples=hyperparams.get('context_samples', None),
                        n_ensembles=hyperparams.get('n_ensembles', 8)
                    )

            # Save reconstruction probabilities for future use
            save_reconstruction_probs(
                reconstruction_probs=reconstruction_probs,
                test_descriptions=test_descriptions,
                feature_value_indices=feature_value_indices,
                feature_names=feature_names,
                dataset_name=dataset_name,
                method_name=method,
                seed=seed,
                fold_idx=fold_idx
            )

            # Extract itemsets using current similarity threshold
            result = extract_frequent_itemsets_from_reconstruction(
                prob_matrix=reconstruction_probs,
                test_descriptions=test_descriptions,
                feature_value_indices=feature_value_indices,
                data=train_data,
                similarity=hyperparams['similarity'],
                feature_names=feature_names,
                encoder=encoder
            )
            itemsets = result['itemsets']

    elif method.startswith('fpgrowth'):
        # FP-Growth is deterministic and fast, no caching needed
        itemsets, _ = fpgrowth_itemset_learning(
            dataset=train_data,
            max_len=hyperparams['max_length'],
            min_support=hyperparams['min_support']
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    return itemsets


def evaluate_corels_with_cv(dataset, method, dataset_name, hyperparams, seed=None, n_folds=5, random_state=42):
    """
    Evaluate CORELS classifier with proper cross-validation.
    Itemsets are mined from training folds only to avoid data leakage.
    """
    # Get class column name from mapping
    class_column_name = DATASET_CLASS_COLUMNS.get(dataset_name)
    if class_column_name is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Please add class column name to DATASET_CLASS_COLUMNS.")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_results = []

    print(f"    Starting {n_folds}-fold cross-validation...")

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(dataset, dataset[class_column_name])):
        print(f"      Fold {fold_idx + 1}/{n_folds}: Mining itemsets from training data...", end=' ')

        train_df = dataset.iloc[train_idx].reset_index(drop=True)
        test_df = dataset.iloc[test_idx].reset_index(drop=True)

        # Mine itemsets from training data only
        fold_start_time = time.time()
        try:
            mined_itemsets = mine_itemsets_for_method(
                method=method,
                train_data=train_df,
                dataset_name=dataset_name,
                hyperparams=hyperparams,
                seed=seed,
                fold_idx=fold_idx + 1
            )

        except Exception as e:
            print(f"Error mining itemsets: {e}")
            fold_results.append({
                'fold': fold_idx + 1,
                'accuracy': 0.0,
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'exec_time': 0.0,
                'coverage': 0.0,
                'total_itemsets': 0,
                'num_feature_itemsets': 0
            })
            continue

        # Save total number of itemsets before filtering
        total_itemsets = len(mined_itemsets)

        # Filter to get feature-only itemsets (exclude class column)
        feature_itemsets = filter_non_class_itemsets(mined_itemsets, class_column_name)
        num_feature_itemsets = len(feature_itemsets)

        if len(feature_itemsets) == 0:
            print("No feature itemsets")
            fold_results.append({
                'fold': fold_idx + 1,
                'accuracy': 0.0,
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'exec_time': time.time() - fold_start_time,
                'coverage': 0.0,
                'total_itemsets': total_itemsets,
                'num_feature_itemsets': num_feature_itemsets
            })
            continue

        print(f"{len(feature_itemsets)} itemsets, building CORELS classifier...", end=' ')

        # Prepare CORELS input from itemsets
        try:
            rules = prepare_corels_data_from_itemsets(
                itemsets=feature_itemsets,
                transactions=train_df,
                class_column_name=class_column_name
            )

            if len(rules) == 0:
                print("No valid CORELS rules")
                fold_results.append({
                    'fold': fold_idx + 1,
                    'accuracy': 0.0,
                    'f1_score': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'exec_time': time.time() - fold_start_time,
                    'coverage': 0.0,
                    'total_itemsets': total_itemsets,
                    'num_feature_itemsets': num_feature_itemsets
                })
                continue

            # Factorize class labels to 0/1 encoding
            y_train = train_df[class_column_name]
            y_train_encoded, categories = pd.factorize(y_train)

            # Create CORELS input files
            create_corels_input_files(rules, y_train_encoded, dataset_name)

            # Train CORELS
            corels_model, corels_exec_time = run_corels(dataset_name)

            if corels_model is None:
                print("CORELS training failed")
                fold_results.append({
                    'fold': fold_idx + 1,
                    'accuracy': 0.0,
                    'f1_score': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'exec_time': time.time() - fold_start_time,
                    'coverage': 0.0,
                    'total_itemsets': total_itemsets,
                    'num_feature_itemsets': num_feature_itemsets
                })
                continue

            # Save classifier for this fold
            save_corels_model(corels_model, method, dataset_name, seed, fold_idx=fold_idx + 1)

            # Evaluate on test data
            X_test = test_df.drop(columns=[class_column_name])
            y_test = test_df[class_column_name]

            # Encode test labels using the same categories
            y_test_encoded = pd.Series(y_test).map(
                lambda x: categories.tolist().index(x) if x in categories else -1
            ).values

            exec_time = time.time() - fold_start_time

            # Get predictions
            y_true, y_pred = convert_corels_predictions_to_sklearn_format(
                corels_model, X_test, y_test_encoded, class_column_name
            )

            # Filter out None predictions
            valid_indices = [i for i, pred in enumerate(y_pred) if pred is not None]

            if len(valid_indices) == 0:
                print(f"No predictions (exec_time={exec_time:.2f}s)")
                fold_results.append({
                    'fold': fold_idx + 1,
                    'accuracy': 0.0,
                    'f1_score': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'exec_time': exec_time,
                    'coverage': 0.0,
                    'total_itemsets': total_itemsets,
                    'num_feature_itemsets': num_feature_itemsets
                })
                continue

            y_true_valid = y_true[valid_indices]
            y_pred_valid = y_pred[valid_indices]

            accuracy = accuracy_score(y_true_valid, y_pred_valid)
            f1 = f1_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0)
            precision = precision_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0)
            recall = recall_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0)
            coverage = len(valid_indices) / len(y_true)

            print(f"Acc={accuracy:.3f}, Cov={coverage:.3f} (exec_time={exec_time:.2f}s)")

            fold_results.append({
                'fold': fold_idx + 1,
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'exec_time': exec_time,
                'coverage': coverage,
                'total_itemsets': total_itemsets,
                'num_feature_itemsets': num_feature_itemsets
            })

        except Exception as e:
            print(f"Error in CORELS: {e}")
            import traceback
            traceback.print_exc()
            fold_results.append({
                'fold': fold_idx + 1,
                'accuracy': 0.0,
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'exec_time': time.time() - fold_start_time,
                'coverage': 0.0,
                'total_itemsets': total_itemsets,
                'num_feature_itemsets': num_feature_itemsets
            })
            continue

    if len(fold_results) == 0:
        return None, None

    # Calculate average metrics
    avg_metrics = {
        'accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'f1_score': np.mean([r['f1_score'] for r in fold_results]),
        'precision': np.mean([r['precision'] for r in fold_results]),
        'recall': np.mean([r['recall'] for r in fold_results]),
        'coverage': np.mean([r['coverage'] for r in fold_results]),
        'exec_time': np.mean([r['exec_time'] for r in fold_results]),
        'total_itemsets': np.mean([r['total_itemsets'] for r in fold_results]),
        'num_feature_itemsets': np.mean([r['num_feature_itemsets'] for r in fold_results])
    }

    return avg_metrics, fold_results


if __name__ == "__main__":
    print("=" * 80)
    print("CORELS Classification Experiments with 5-Fold Cross-Validation")
    print("=" * 80)

    # Hyperparameters for each method
    hyperparams = {
        'aerial': {'max_length': 2, 'similarity': 0.3},
        'tabpfn': {'max_length': 2, 'similarity': 0.3, 'context_samples': None, 'n_estimators': 8},
        'tabicl': {'max_length': 2, 'similarity': 0.3, 'context_samples': None, 'n_estimators': 8},
        'tabdpt': {'max_length': 2, 'similarity': 0.3, 'context_samples': None, 'n_ensembles': 8},
        'fpgrowth_0.5': {'max_length': 2, 'min_support': 0.5},
        'fpgrowth_0.3': {'max_length': 2, 'min_support': 0.3},
        'fpgrowth_0.2': {'max_length': 2, 'min_support': 0.2},
        'fpgrowth_0.1': {'max_length': 2, 'min_support': 0.1},
        'fpgrowth_0.05': {'max_length': 2, 'min_support': 0.05},
        'fpgrowth_0.01': {'max_length': 2, 'min_support': 0.01},
    }

    # Methods to run (all available methods by default)
    methods = ['aerial', 'tabpfn', 'tabicl', 'tabdpt', 'fpgrowth_0.3', 'fpgrowth_0.1']
    n_runs = 10
    base_seed = 42
    n_folds = 5

    seed_sequence = generate_seed_sequence(base_seed, n_runs)
    print(f"\nSeed sequence: {seed_sequence}")
    print(f"\nHyperparameters:")
    for method in methods:
        print(f"  {method}: {hyperparams[method]}")

    # Load all datasets by default
    datasets = get_ucimlrepo_datasets(size="small")
    all_datasets = datasets

    os.makedirs("out", exist_ok=True)

    all_seed_results = []
    all_average_results = []

    for method in methods:
        print(f"\n{'=' * 80}")
        print(f"Method: {method}")
        print(f"{'=' * 80}")

        for dataset_info in all_datasets:
            dataset_name = dataset_info['name']
            dataset = dataset_info['data']

            print(f"\nDataset: {dataset_name} (shape: {dataset.shape})")

            # Get dataset-specific parameters for Aerial
            if method == 'aerial':
                aerial_params = get_aerial_dataset_parameters(dataset_name, dataset_size='small')
                current_hyperparams = hyperparams[method].copy()
                current_hyperparams.update(aerial_params)
                print(f"  Using Aerial dataset-specific params: batch_size={aerial_params['batch_size']}, "
                      f"layer_dims={aerial_params['layer_dims']}, epochs={aerial_params['epochs']}")
            else:
                current_hyperparams = hyperparams[method]

            dataset_runs = []

            for run_idx in range(n_runs):
                run_seed = seed_sequence[run_idx]
                print(f"  Run {run_idx + 1}/{n_runs} (seed={run_seed})")

                # Add random_state to hyperparams for methods that use it
                if method in ['aerial', 'tabpfn']:
                    current_hyperparams['random_state'] = run_seed

                try:
                    avg_metrics, fold_results = evaluate_corels_with_cv(
                        dataset=dataset,
                        method=method,
                        dataset_name=dataset_name,
                        hyperparams=current_hyperparams,
                        seed=run_seed,
                        n_folds=n_folds,
                        random_state=run_seed
                    )

                    if avg_metrics is None:
                        print(f"    No results - skipping")
                        continue

                    result = {
                        'method': method,
                        'dataset': dataset_name,
                        'run': run_idx + 1,
                        'seed': run_seed,
                        'total_itemsets': avg_metrics['total_itemsets'],
                        'num_feature_itemsets': avg_metrics['num_feature_itemsets'],
                        'accuracy': avg_metrics['accuracy'],
                        'f1_score': avg_metrics['f1_score'],
                        'precision': avg_metrics['precision'],
                        'recall': avg_metrics['recall'],
                        'coverage': avg_metrics['coverage'],
                        'exec_time': avg_metrics['exec_time']
                    }

                    dataset_runs.append(result)
                    all_seed_results.append(result)

                    print(f"    FINAL: Acc={avg_metrics['accuracy']:.4f}, F1={avg_metrics['f1_score']:.4f}, "
                          f"NumFeatureItemsets={avg_metrics['num_feature_itemsets']:.1f}")

                except Exception as e:
                    import traceback

                    print(f"    Error: {e}")
                    traceback.print_exc()
                    continue

            if len(dataset_runs) > 0:
                avg_result = {
                    'method': method,
                    'dataset': dataset_name,
                    'total_itemsets': np.mean([r['total_itemsets'] for r in dataset_runs]),
                    'num_feature_itemsets': np.mean([r['num_feature_itemsets'] for r in dataset_runs]),
                    'accuracy': np.mean([r['accuracy'] for r in dataset_runs]),
                    'f1_score': np.mean([r['f1_score'] for r in dataset_runs]),
                    'precision': np.mean([r['precision'] for r in dataset_runs]),
                    'recall': np.mean([r['recall'] for r in dataset_runs]),
                    'coverage': np.mean([r['coverage'] for r in dataset_runs]),
                    'exec_time': np.mean([r['exec_time'] for r in dataset_runs]),
                }
                all_average_results.append(avg_result)

                print(f"\n  Average across {len(dataset_runs)} runs: "
                      f"Acc={avg_result['accuracy']:.4f}, F1={avg_result['f1_score']:.4f}, "
                      f"NumFeatureItemsets={avg_result['num_feature_itemsets']:.1f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"out/corels_classification_FIXED_{timestamp}.xlsx"

    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        seed_results_df = pd.DataFrame(all_seed_results)
        seed_results_df.to_excel(writer, sheet_name='Results Per Seed', index=False)

        average_df = pd.DataFrame(all_average_results)
        average_df.to_excel(writer, sheet_name='Average Across Seeds', index=False)

        params_df = pd.DataFrame([hyperparams[m] | {'method': m} for m in methods])
        params_df.to_excel(writer, sheet_name='Hyperparameters', index=False)

    print("\n" + "=" * 80)
    print(f"Results saved to {output_filename}")
    print("  - Sheet 1: Results Per Seed (avg of 5-fold CV per seed)")
    print("  - Sheet 2: Average Across Seeds")
    print("  - Sheet 3: Hyperparameters")
    print("Classifiers saved to out/classifiers/corels/{method}/{dataset}/seed_{seed}_fold_{fold}.json")
    print("=" * 80)
