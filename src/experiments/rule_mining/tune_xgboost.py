"""
Optuna-based hyperparameter tuning for XGBoost rule mining.
Tunes max_depth, learning_rate, n_estimators per dataset to maximize
confidence * zhang_metric. Saves results to src/experiments/xgboost_best_parameters.json.

Tuning uses at most TUNING_SAMPLES rows per dataset to keep trials tractable;
the actual experiments run on the full dataset with the found parameters.
"""

import json
import os
import warnings

import optuna
from optuna.samplers import TPESampler

from xgboost_experiments import xgb_rule_learning
from src.utils import get_ucimlrepo_datasets, calculate_rule_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

N_TRIALS = 50
BASE_SEED = 42
TUNING_SAMPLES = 2000  # cap for large datasets; small datasets use all rows
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "..", "xgboost_best_parameters.json")


def make_objective(dataset, random_state):
    n_samples = min(len(dataset), TUNING_SAMPLES)

    def objective(trial):
        max_depth = trial.suggest_int("max_depth", 2, 6)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        n_estimators = trial.suggest_int("n_estimators", 20, 200)

        try:
            rules, feature_names, original_data = xgb_rule_learning(
                dataset=dataset,
                max_antecedents=2,
                context_samples=n_samples,
                ant_similarity=0.5,
                cons_similarity=0.8,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
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
    print("XGBoost Hyperparameter Tuning with Optuna")
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
            "learning_rate": best["learning_rate"],
            "n_estimators": best["n_estimators"],
            "best_objective": best_value,
        }

    output_path = os.path.normpath(OUTPUT_JSON)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(best_params, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Best parameters saved to {output_path}")
    print("=" * 80)
