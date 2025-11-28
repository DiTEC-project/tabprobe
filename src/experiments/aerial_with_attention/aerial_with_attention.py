import time
from collections import defaultdict

from src.initial_experiments.aerial_with_attention.aerial import model, model_column_row_attention
from src.initial_experiments.aerial_with_attention.aerial import rule_quality, rule_extraction
from ucimlrepo import fetch_ucirepo


def get_rule_quality_stats_summary(rule_quality_stats):
    agg = defaultdict(list)
    for d in rule_quality_stats:
        for k, v in d.items():
            agg[k].append(v)
    return {k: {'avg': round(sum(v) / len(v), 3), 'min': round(min(v), 3), 'max': round(max(v), 3)} for k, v in
            agg.items()}


breast_cancer = fetch_ucirepo(id=14).data.features
mushroom = fetch_ucirepo(id=73).data.features
NUM_RUNS = 10
EPOCHS = [5, 10]
datasets = [mushroom]

original_aerial_stats = []
updated_stats = []
for dataset in datasets:
    for epoch in EPOCHS:
        print("Epochs:", epoch)
        for i in range(NUM_RUNS):
            print(f"{i + 1}/{NUM_RUNS}")
            # Aerial+
            # start = time.time()
            # trained_autoencoder = model.train(dataset, batch_size=32, epochs=epoch)
            # association_rules = rule_extraction.generate_rules(trained_autoencoder)
            # exec_time = time.time() - start
            # if len(association_rules) > 0:
            #     stats_original, association_rules = rule_quality.calculate_rule_stats(association_rules,
            #                                                                           trained_autoencoder.input_vectors,
            #                                                                           max_workers=8)
            #     stats_original["exec_time"] = exec_time
            #     original_aerial_stats.append(stats_original)
            #     # print(stats_original)

            # Aerial+ with multi-head column and row aerial_with_attention
            start = time.time()
            ae_with_column_attention = model_column_row_attention.train(dataset, batch_size=32, epochs=epoch)
            association_rules2 = rule_extraction.generate_rules(ae_with_column_attention)
            exec_time = time.time() - start
            if len(association_rules2) > 0:
                stats, association_rules2 = rule_quality.calculate_rule_stats(association_rules2,
                                                                              ae_with_column_attention.input_vectors,
                                                                              max_workers=4)
                stats["exec_time"] = exec_time
                updated_stats.append(stats)
                # print(stats)

        print("Aerial+:", get_rule_quality_stats_summary(original_aerial_stats))
        print("Aerial+ with multi-head column and row aerial_with_attention:", get_rule_quality_stats_summary(updated_stats))
