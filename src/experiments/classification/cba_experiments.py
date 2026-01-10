import time
import os
import json
import warnings

import aerial
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

warnings.filterwarnings('ignore', category=DeprecationWarning)

from src.experiments.classification.cba.algorithms.m2algorithm import M2Algorithm
from src.experiments.classification.cba.data_structures.transaction_db import TransactionDB
from src.experiments.classification.cba.data_structures.car import ClassAssocationRule
from src.experiments.classification.cba.data_structures.consequent import Consequent
from src.experiments.classification.cba.data_structures.antecedent import Antecedent
from src.experiments.classification.cba.data_structures.item import Item

from src.utils import get_ucimlrepo_datasets, generate_seed_sequence, calculate_rule_metrics

# Import rule mining functions
from src.experiments.rule_mining.aerial_experiments import aerial_rule_learning
from src.experiments.rule_mining.tabpfn_experiments import tabpfn_rule_learning
from src.experiments.rule_mining.tabicl_experiments import tabicl_rule_learning
from src.experiments.rule_mining.tabdpt_experiments import tabdpt_rule_learning
from src.experiments.rule_mining.fpgrowth_experiments import fpgrowth_rule_learning

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


def filter_class_association_rules(rules, class_column_name):
    class_rules = []
    for rule in rules:
        if rule['consequent']['feature'] == class_column_name:
            class_rules.append(rule)
    return class_rules


def convert_rules_to_cars(rules):
    cars = []
    for rule in rules:
        ant_items = []
        for ant in rule['antecedents']:
            item = Item(ant['feature'], ant['value'])
            ant_items.append(item)

        antecedent = Antecedent(ant_items)
        consequent = Consequent(rule['consequent']['feature'], rule['consequent']['value'])

        support = rule['support'] * 100
        confidence = rule['confidence'] * 100

        car = ClassAssocationRule(antecedent, consequent, support, confidence)
        cars.append(car)

    cars.sort(reverse=True)
    return cars


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


def save_classifier(clf, rule_miner, dataset_name, seed, fold_idx=None, output_dir="out/classifiers"):
    method_dir = os.path.join(output_dir, "cba", rule_miner, dataset_name)
    os.makedirs(method_dir, exist_ok=True)

    if seed is None and fold_idx is None:
        output_file = os.path.join(method_dir, "classifier.json")
    elif fold_idx is None:
        output_file = os.path.join(method_dir, f"seed_{seed}.json")
    elif seed is None:
        output_file = os.path.join(method_dir, f"fold_{fold_idx}.json")
    else:
        output_file = os.path.join(method_dir, f"seed_{seed}_fold_{fold_idx}.json")

    classifier_data = {
        'dataset': dataset_name,
        'rule_miner': rule_miner,
        'seed': seed,
        'num_rules': len(clf.rules),
        'default_class': clf.default_class,
        'default_class_confidence': clf.default_class_confidence,
        'rules': []
    }

    for rule in clf.rules:
        rule_dict = {
            'antecedents': [{'feature': key, 'value': val} for key, val in rule.antecedent.itemset.items()],
            'consequent': {'feature': clf.default_class_attribute, 'value': rule.consequent.value},
            'support': rule.support,
            'confidence': rule.confidence
        }
        classifier_data['rules'].append(rule_dict)

    with open(output_file, 'w') as f:
        json.dump(classifier_data, f, indent=2)

    return output_file


def mine_rules_for_method(method, train_data, hyperparams):
    """
    Mine rules from training data using specified method.

    Args:
        method: Method name ('aerial', 'tabpfn', 'tabicl', 'tabdpt', 'fpgrowth_X')
        train_data: Training dataframe
        hyperparams: Hyperparameters for the method

    Returns:
        List of mined rules
    """
    if method == 'aerial':
        rules, _ = aerial_rule_learning(
            train_data,
            max_antecedents=hyperparams['max_antecedents'],
            ant_similarity=hyperparams['ant_similarity'],
            cons_similarity=hyperparams['cons_similarity'],
            batch_size=hyperparams['batch_size'],
            layer_dims=hyperparams['layer_dims'],
            epochs=hyperparams['epochs']
        )
    elif method == 'tabpfn':
        rules, feature_names, original_data = tabpfn_rule_learning(
            train_data,
            max_antecedents=hyperparams['max_antecedents'],
            context_samples=hyperparams.get('context_samples', None),
            ant_similarity=hyperparams['ant_similarity'],
            cons_similarity=hyperparams['cons_similarity'],
            n_estimators=hyperparams['n_estimators']
        )
        rules, _ = calculate_rule_metrics(
            rules=rules,
            data=original_data,
            feature_names=feature_names
        )
    elif method == 'tabicl':
        rules, feature_names, original_data = tabicl_rule_learning(
            train_data,
            max_antecedents=hyperparams['max_antecedents'],
            context_samples=hyperparams.get('context_samples', None),
            ant_similarity=hyperparams['ant_similarity'],
            cons_similarity=hyperparams['cons_similarity'],
            n_estimators=hyperparams['n_estimators'],
        )
        rules, _ = calculate_rule_metrics(
            rules=rules,
            data=original_data,
            feature_names=feature_names
        )
    elif method == 'tabdpt':
        rules, feature_names, original_data = tabdpt_rule_learning(
            train_data,
            max_antecedents=hyperparams['max_antecedents'],
            ant_similarity=hyperparams['ant_similarity'],
            cons_similarity=hyperparams['cons_similarity'],
            n_ensembles=hyperparams['n_ensembles']
        )
        rules, _ = calculate_rule_metrics(
            rules=rules,
            data=original_data,
            feature_names=feature_names
        )
    elif method.startswith('fpgrowth'):
        rules, _ = fpgrowth_rule_learning(
            train_data,
            max_len=hyperparams['max_antecedents'],
            min_support=hyperparams['min_support'],
            min_confidence=hyperparams['min_confidence']
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return rules


def evaluate_cba_with_cv(dataset, method, dataset_name, hyperparams,
                         seed=None, n_folds=5, random_state=42):
    """
    Evaluate CBA classifier with proper cross-validation.
    Rules are mined from training folds only to avoid data leakage.
    """
    # Get class column name from mapping
    class_column_name = DATASET_CLASS_COLUMNS.get(dataset_name)
    if class_column_name is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Please add class column name to DATASET_CLASS_COLUMNS.")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_results = []

    print(f"    Starting {n_folds}-fold cross-validation...")

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(dataset, dataset[class_column_name])):
        print(f"      Fold {fold_idx + 1}/{n_folds}: Mining rules from training data...", end=' ')

        train_df = dataset.iloc[train_idx].reset_index(drop=True)
        test_df = dataset.iloc[test_idx].reset_index(drop=True)

        # Mine rules from training data only
        fold_start_time = time.time()
        try:
            mined_rules = mine_rules_for_method(method, train_df, hyperparams)

        except Exception as e:
            print(f"Error mining rules: {e}")
            fold_results.append({
                'fold': fold_idx + 1,
                'accuracy': 0.0,
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'exec_time': 0.0,
                'coverage': 0.0,
                'total_rules_before_filter': 0,
                'num_classification_rules': 0
            })
            continue

        # Save total number of rules before filtering
        total_rules_before_filter = len(mined_rules)

        # Filter class association rules
        class_rules = filter_class_association_rules(mined_rules, class_column_name)
        num_classification_rules = len(class_rules)

        if len(class_rules) == 0:
            print("No class rules")
            fold_results.append({
                'fold': fold_idx + 1,
                'accuracy': 0.0,
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'exec_time': time.time() - fold_start_time,
                'coverage': 0.0,
                'total_rules_before_filter': total_rules_before_filter,
                'num_classification_rules': num_classification_rules
            })
            continue

        cars = convert_rules_to_cars(class_rules)
        print(f"{len(cars)} rules, building classifier...", end=' ')

        # Build classifier on training data
        train_txn = TransactionDB.from_DataFrame(train_df, target=class_column_name)
        clf = M2Algorithm(cars, train_txn).build()

        # Save classifier for this fold
        save_classifier(clf, method, dataset_name, seed, fold_idx=fold_idx + 1)

        # Evaluate on test data
        test_txn = TransactionDB.from_DataFrame(test_df, target=class_column_name)
        y_pred = clf.predict_all(test_txn)
        exec_time = time.time() - fold_start_time

        y_true = test_txn.classes
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
                'total_rules_before_filter': total_rules_before_filter,
                'num_classification_rules': num_classification_rules
            })
            continue

        y_true_valid = [y_true[i] for i in valid_indices]
        y_pred_valid = [y_pred[i] for i in valid_indices]

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
            'total_rules_before_filter': total_rules_before_filter,
            'num_classification_rules': num_classification_rules
        })

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
        'total_rules_before_filter': np.mean([r['total_rules_before_filter'] for r in fold_results]),
        'num_classification_rules': np.mean([r['num_classification_rules'] for r in fold_results])
    }

    return avg_metrics, fold_results


if __name__ == "__main__":
    print("=" * 80)
    print("CBA Classification Experiments with 5-Fold Cross-Validation")
    print("=" * 80)

    # Hyperparameters for each method
    # context_samples = None means use the entire table
    hyperparams = {
        'aerial': {'max_antecedents': 2, 'ant_similarity': 0.1, 'cons_similarity': 0.8},
        'tabpfn': {'max_antecedents': 2, 'ant_similarity': 0.1, 'cons_similarity': 0.8,
                   'context_samples': None, 'n_estimators': 4},
        'tabicl': {'max_antecedents': 2, 'ant_similarity': 0.1, 'cons_similarity': 0.8,
                   'context_samples': None, 'n_estimators': 4},
        'tabdpt': {'max_antecedents': 2, 'ant_similarity': 0.1, 'cons_similarity': 0.8,
                   'n_ensembles': 8},
        'fpgrowth_0.5': {'max_antecedents': 2, 'min_support': 0.5, 'min_confidence': 0.8},
        'fpgrowth_0.3': {'max_antecedents': 2, 'min_support': 0.3, 'min_confidence': 0.8},
        'fpgrowth_0.2': {'max_antecedents': 2, 'min_support': 0.2, 'min_confidence': 0.8},
        'fpgrowth_0.1': {'max_antecedents': 2, 'min_support': 0.1, 'min_confidence': 0.8},
        'fpgrowth_0.05': {'max_antecedents': 2, 'min_support': 0.05, 'min_confidence': 0.8},
        'fpgrowth_0.01': {'max_antecedents': 2, 'min_support': 0.01, 'min_confidence': 0.8},
    }

    methods = ['tabdpt']
    n_runs = 10
    base_seed = 42
    n_folds = 5

    seed_sequence = generate_seed_sequence(base_seed, n_runs)
    print(f"\nSeed sequence: {seed_sequence}")
    print(f"\nHyperparameters:")
    for method, params in hyperparams.items():
        print(f"  {method}: {params}")

    dataset_size = "normal"
    datasets = get_ucimlrepo_datasets(size=dataset_size, names=["mushroom"])
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
                aerial_params = get_aerial_dataset_parameters(dataset_name, dataset_size=dataset_size)
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

                try:
                    avg_metrics, fold_results = evaluate_cba_with_cv(
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
                        'total_rules_before_filter': avg_metrics['total_rules_before_filter'],
                        'num_classification_rules': avg_metrics['num_classification_rules'],
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
                          f"NumClassRules={avg_metrics['num_classification_rules']:.1f}")

                except Exception as e:
                    import traceback

                    print(f"    Error: {e}")
                    traceback.print_exc()
                    continue

            if len(dataset_runs) > 0:
                avg_result = {
                    'method': method,
                    'dataset': dataset_name,
                    'total_rules_before_filter': np.mean([r['total_rules_before_filter'] for r in dataset_runs]),
                    'num_classification_rules': np.mean([r['num_classification_rules'] for r in dataset_runs]),
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
                      f"NumClassRules={avg_result['num_classification_rules']:.1f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"out/cba_classification_FIXED_{timestamp}.xlsx"

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
    print("Classifiers saved to out/classifiers/cba/{method}/{dataset}/seed_{seed}_fold_{fold}.json")
    print("=" * 80)
