"""
Optuna-based hyperparameter tuning for RandomForest rule mining.
Tunes n_estimators, max_depth, min_samples_leaf per dataset to maximize
confidence * zhang_metric. Saves results to src/experiments/random_forest_best_parameters.json.
"""

import json
import os
import warnings

import optuna
from optuna.samplers import TPESampler

from random_forest_experiments import rf_rule_learning
from src.utils import get_ucimlrepo_datasets, calculate_rule_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

N_TRIALS = 50
BASE_SEED = 42
TUNING_SAMPLES = 2000  # cap for large datasets; small datasets use all rows
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "..", "random_forest_best_parameters.json")


def make_objective(dataset, random_state):
    n_samples = min(len(dataset), TUNING_SAMPLES)

    def objective(trial):
        max_depth = trial.suggest_int("max_depth", 2, 20)
        n_estimators = trial.suggest_int("n_estimators", 20, 200)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)

        try:
            rules, feature_names, original_data = rf_rule_learning(
                dataset=dataset,
                max_antecedents=2,
                context_samples=n_samples,
                ant_similarity=0.5,
                cons_similarity=0.8,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                max_workers=1,
                query_batch_size=512,
            )
        except Exception:
            return 0.0

        if len(rules) == 0:
            return 0.0

        _, avg_metrics = calculate_rule_metrics(
            rules=rules,
            data=original_data,
            feature_names=feature_names,
        )
        return avg_metrics["confidence"] * avg_metrics["zhangs_metric"]

    return objective


if __name__ == "__main__":
    print("=" * 80)
    print("RandomForest Hyperparameter Tuning with Optuna")
    print("=" * 80)
    print(f"Trials per dataset : {N_TRIALS}")
    print(f"Max tuning samples : {TUNING_SAMPLES}")

    datasets = get_ucimlrepo_datasets(size="small") + get_ucimlrepo_datasets(size="normal")
    best_params = {}

    for dataset_info in datasets:
        dataset_name = dataset_info["name"]
        dataset = dataset_info["data"]
        n_used = min(len(dataset), TUNING_SAMPLES)

        print(f"\nTuning dataset: {dataset_name} (shape={dataset.shape}, tuning on {n_used} rows)")

        sampler = TPESampler(seed=BASE_SEED)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            make_objective(dataset, random_state=BASE_SEED),
            n_trials=N_TRIALS,
            show_progress_bar=True,
        )

        best = study.best_params
        best_value = study.best_value
        print(f"  Best params : {best}")
        print(f"  Best value  : {best_value:.4f}  (confidence * zhang_metric)")

        best_params[dataset_name] = {
            "max_depth": best["max_depth"],
            "n_estimators": best["n_estimators"],
            "min_samples_leaf": best["min_samples_leaf"],
            "best_objective": best_value,
        }

        # written after every dataset so a later failure/timeout cannot discard already
        # finished datasets (see calibration_diagnostics.py for the same lesson)
        output_path = os.path.normpath(OUTPUT_JSON)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"  Saved progress ({len(best_params)}/{len(datasets)} datasets) to {output_path}")

    print("\n" + "=" * 80)
    print(f"Best parameters saved to {output_path}")
    print("=" * 80)