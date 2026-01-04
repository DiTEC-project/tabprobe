import time
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.experiments.classification.cba.algorithms.m1algorithm import M1Algorithm
from src.experiments.classification.cba.data_structures.transaction_db import TransactionDB
from src.experiments.classification.cba.data_structures.car import ClassAssocationRule
from src.experiments.classification.cba.data_structures.consequent import Consequent
from src.experiments.classification.cba.data_structures.antecedent import Antecedent
from src.experiments.classification.cba.data_structures.item import Item

from src.utils import get_ucimlrepo_datasets, load_rules, generate_seed_sequence

# Class column names for each dataset (always the last column in original CSV files)
DATASET_CLASS_COLUMNS = {
    'breast_cancer': 'Class',
    'congressional_voting': 'Class',
    'mushroom': 'poisonous',
    'chess_king_rook_vs_king_pawn': 'wtoeg',
    'spambase': 'Class',
    'lung_cancer': 'class',
    'hepatitis': 'Class',
    'breast_cancer_coimbra': 'Classification',
    'cervical_cancer_behavior_risk': 'ca_cervix',
    'autism_screening_adolescent': 'Class/ASD',
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


def save_classifier(clf, rule_miner, dataset_name, seed, output_dir="out/classifiers"):
    method_dir = os.path.join(output_dir, "cba", rule_miner, dataset_name)
    os.makedirs(method_dir, exist_ok=True)

    if seed is None:
        output_file = os.path.join(method_dir, "classifier.json")
    else:
        output_file = os.path.join(method_dir, f"seed_{seed}.json")

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


def evaluate_cba_with_cv(dataset, rules, rule_miner, dataset_name, seed, n_folds=5, random_state=42):
    # Get class column name from mapping (class is always last column in original CSV files)
    class_column_name = DATASET_CLASS_COLUMNS.get(dataset_name)
    if class_column_name is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Please add class column name to DATASET_CLASS_COLUMNS.")

    class_rules = filter_class_association_rules(rules, class_column_name)

    if len(class_rules) == 0:
        return None, None, None

    cars = convert_rules_to_cars(class_rules)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(dataset, dataset[class_column_name])):
        train_df = dataset.iloc[train_idx].reset_index(drop=True)
        test_df = dataset.iloc[test_idx].reset_index(drop=True)

        # Use the actual class column name (last column) for the target
        train_txn = TransactionDB.from_DataFrame(train_df, target=class_column_name)
        test_txn = TransactionDB.from_DataFrame(test_df, target=class_column_name)

        start_time = time.time()
        clf = M1Algorithm(cars, train_txn).build()
        y_pred = clf.predict_all(test_txn)
        exec_time = time.time() - start_time

        y_true = test_txn.classes

        valid_indices = [i for i, pred in enumerate(y_pred) if pred is not None]
        if len(valid_indices) == 0:
            fold_results.append({
                'fold': fold_idx + 1,
                'accuracy': 0.0,
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'exec_time': exec_time,
                'coverage': 0.0
            })
            continue

        y_true_valid = [y_true[i] for i in valid_indices]
        y_pred_valid = [y_pred[i] for i in valid_indices]

        accuracy = accuracy_score(y_true_valid, y_pred_valid)
        f1 = f1_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0)
        precision = precision_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0)
        recall = recall_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0)
        coverage = len(valid_indices) / len(y_true)

        fold_results.append({
            'fold': fold_idx + 1,
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'exec_time': exec_time,
            'coverage': coverage
        })

    avg_metrics = {
        'accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'f1_score': np.mean([r['f1_score'] for r in fold_results]),
        'precision': np.mean([r['precision'] for r in fold_results]),
        'recall': np.mean([r['recall'] for r in fold_results]),
        'coverage': np.mean([r['coverage'] for r in fold_results]),
        'exec_time': np.mean([r['exec_time'] for r in fold_results]),
        'num_rules': len(cars)
    }

    # Train final classifier on full dataset using the actual class column name (last column)
    full_txn = TransactionDB.from_DataFrame(dataset, target=class_column_name)
    final_clf = M1Algorithm(cars, full_txn).build()

    classifier_file = save_classifier(final_clf, rule_miner, dataset_name, seed)

    return avg_metrics, fold_results, classifier_file


if __name__ == "__main__":
    print("=" * 80)
    print("CBA Classification Experiments with 5-Fold Cross-Validation")
    print("=" * 80)

    methods = ['aerial', 'tabpfn', 'tabicl', 'tabdpt', 'fpgrowth']
    n_runs = 10
    base_seed = 42
    n_folds = 5

    seed_sequence = generate_seed_sequence(base_seed, n_runs)
    print(f"\nSeed sequence: {seed_sequence}")

    datasets = get_ucimlrepo_datasets(size="normal")
    small_datasets = get_ucimlrepo_datasets(size="small")
    all_datasets = datasets + small_datasets

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

            dataset_runs = []

            for run_idx in range(n_runs):
                run_seed = seed_sequence[run_idx]

                try:
                    if method == 'fpgrowth':
                        rule_data = load_rules(dataset_name, method, seed=None)
                    else:
                        rule_data = load_rules(dataset_name, method, run_seed)

                    rules = rule_data['rules']

                    if len(rules) == 0:
                        print(f"  Run {run_idx + 1}/{n_runs} (seed={run_seed}): No rules, skipping")
                        continue

                    result_data = evaluate_cba_with_cv(dataset, rules, method, dataset_name,
                                                       run_seed if method != 'fpgrowth' else None, n_folds=n_folds,
                                                       random_state=run_seed)

                    if result_data[0] is None:
                        print(f"  Run {run_idx + 1}/{n_runs} (seed={run_seed}): No class association rules, skipping")
                        continue

                    avg_metrics, fold_results, classifier_file = result_data

                    result = {
                        'method': method,
                        'dataset': dataset_name,
                        'run': run_idx + 1,
                        'seed': run_seed,
                        'num_rules': avg_metrics['num_rules'],
                        'accuracy': avg_metrics['accuracy'],
                        'f1_score': avg_metrics['f1_score'],
                        'precision': avg_metrics['precision'],
                        'recall': avg_metrics['recall'],
                        'coverage': avg_metrics['coverage'],
                        'exec_time': avg_metrics['exec_time']
                    }

                    dataset_runs.append(result)
                    all_seed_results.append(result)

                    print(
                        f"  Run {run_idx + 1}/{n_runs} (seed={run_seed}): Acc={avg_metrics['accuracy']:.4f}, F1={avg_metrics['f1_score']:.4f}, Prec={avg_metrics['precision']:.4f}, Rec={avg_metrics['recall']:.4f}, Clf saved to {classifier_file}")

                    if method == 'fpgrowth':
                        for future_run_idx in range(run_idx + 1, n_runs):
                            future_seed = seed_sequence[future_run_idx]
                            result_copy = result.copy()
                            result_copy['run'] = future_run_idx + 1
                            result_copy['seed'] = future_seed
                            dataset_runs.append(result_copy)
                            all_seed_results.append(result_copy)
                        break

                except FileNotFoundError:
                    print(f"  Run {run_idx + 1}/{n_runs} (seed={run_seed}): Rules not found, skipping")
                    continue
                except Exception as e:
                    print(f"  Run {run_idx + 1}/{n_runs} (seed={run_seed}): Error - {e}")
                    continue

            if len(dataset_runs) > 0:
                avg_result = {
                    'method': method,
                    'dataset': dataset_name,
                    'num_rules': np.mean([r['num_rules'] for r in dataset_runs]),
                    'accuracy': np.mean([r['accuracy'] for r in dataset_runs]),
                    'f1_score': np.mean([r['f1_score'] for r in dataset_runs]),
                    'precision': np.mean([r['precision'] for r in dataset_runs]),
                    'recall': np.mean([r['recall'] for r in dataset_runs]),
                    'coverage': np.mean([r['coverage'] for r in dataset_runs]),
                    'exec_time': np.mean([r['exec_time'] for r in dataset_runs]),
                }
                all_average_results.append(avg_result)

                print(
                    f"  Average across seeds: Acc={avg_result['accuracy']:.4f}, F1={avg_result['f1_score']:.4f}, Prec={avg_result['precision']:.4f}, Rec={avg_result['recall']:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"out/cba_classification_{timestamp}.xlsx"

    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        seed_results_df = pd.DataFrame(all_seed_results)
        seed_results_df.to_excel(writer, sheet_name='Results Per Seed', index=False)

        average_df = pd.DataFrame(all_average_results)
        average_df.to_excel(writer, sheet_name='Average Across Seeds', index=False)

    print("\n" + "=" * 80)
    print(f"Results saved to {output_filename}")
    print("  - Sheet 1: Results Per Seed (avg of 5-fold CV per seed)")
    print("  - Sheet 2: Average Across Seeds")
    print("Classifiers saved to out/classifiers/cba/{rule_miner}/{dataset}/")
    print("=" * 80)
