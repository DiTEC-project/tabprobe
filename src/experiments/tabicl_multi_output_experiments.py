"""
TabICL Multi-Output Baseline

Multi-output architecture modification to TabICL for association rule mining.
Predicts all features simultaneously in a single forward pass (Aerial-style).
"""
import time
import numpy as np
import pandas as pd
import torch

from tabicl_multi import TabICLClassifier
from ucimlrepo import fetch_ucirepo
from src.experiments.tabicl_multi.model.tabicl import TabICL
from src.shared.aerial import prepare_categorical_data
from src.shared.aerial.test_matrix import generate_aerial_test_matrix
from src.shared.aerial.rule_extraction import extract_rules_from_reconstruction
from src.shared.rule_quality import calculate_rule_metrics


def load_pretrained_weights_partial(model, checkpoint_path):
    """Load pretrained weights, skipping layers with shape mismatches."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    model_state = model.state_dict()

    loaded, skipped = [], []
    for name, param in state_dict.items():
        if name in model_state and model_state[name].shape == param.shape:
            model_state[name] = param
            loaded.append(name)
        else:
            skipped.append(name)

    model.load_state_dict(model_state, strict=False)
    print(f"Loaded {len(loaded)} layers, skipped {len(skipped)} layers")
    return model


def tabicl_multi_output_reconstruction(model, context_table, query_matrix,
                                       device='cpu', noise_factor=0.5):
    """Multi-output reconstruction: predict all features in single forward pass."""
    model.eval()
    model.to(device)

    # Add Gaussian noise to context table (denoising approach)
    noisy_context = (context_table + np.random.randn(*context_table.shape) * noise_factor).clip(0, 1)

    # Convert to tensors
    X_context = torch.FloatTensor(noisy_context).to(device).unsqueeze(0)  # (1, N_context, H)
    X_query = torch.FloatTensor(query_matrix).to(device).unsqueeze(0)  # (1, N_query, H)

    with torch.no_grad():
        # Multi-output prediction: X_context for conditioning, X_query to reconstruct
        output = model.predict_multi_output(
            X_context=X_context,
            X_query=X_query,
            return_logits=False
        )
        # Output shape: (1, N_query, total_output_dim)
        reconstruction_probs = output[0].cpu().numpy()

    return reconstruction_probs


def tabicl_multi_output_rule_learning(dataset, max_antecedents=2,
                                      ant_similarity=0.0, cons_similarity=0.8,
                                      checkpoint_path=None):
    """Multi-output rule learning using modified TabICL."""

    encoded_data, classes_per_feature, feature_names, encoder = prepare_categorical_data(dataset)

    print(f"Dataset shape: {encoded_data.shape}")
    print(f"Number of features: {len(classes_per_feature)}")
    print(f"Classes per feature: {classes_per_feature}")

    test_matrix, test_descriptions, feature_value_indices = generate_aerial_test_matrix(
        n_features=len(classes_per_feature),
        classes_per_feature=classes_per_feature,
        max_antecedents=max_antecedents,
        use_zeros_for_unmarked=False
    )

    print(f"\nTest matrix shape: {test_matrix.shape}")
    print(f"Number of test vectors: {len(test_descriptions)}")

    model = TabICL(classes_per_feature=classes_per_feature)

    if checkpoint_path:
        model = load_pretrained_weights_partial(model, checkpoint_path)

    print(f"\nUsing TabICL for multi-output reconstruction...")
    reconstruction_probs = tabicl_multi_output_reconstruction(
        model=model,
        context_table=encoded_data,
        query_matrix=test_matrix
    )

    print(f"Reconstruction shape: {reconstruction_probs.shape}")

    print(f"\nExtracting rules...")
    rules = extract_rules_from_reconstruction(
        prob_matrix=reconstruction_probs,
        test_descriptions=test_descriptions,
        feature_value_indices=feature_value_indices,
        ant_similarity=ant_similarity,
        cons_similarity=cons_similarity,
        feature_names=feature_names,
        encoder=encoder
    )

    print(f"{len(rules)} rules found!")

    return rules, feature_names


if __name__ == "__main__":
    print("=" * 80)
    print("TabICL Multi-Output Baseline")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("Loading breast cancer dataset...")
    breast_cancer = fetch_ucirepo(id=14)
    X_features = breast_cancer.data.features
    y_target = breast_cancer.data.targets
    full_data = pd.concat([X_features, y_target], axis=1)

    print(f"Features shape: {X_features.shape}")
    print(f"Target shape: {y_target.shape}")
    print(f"Combined data shape: {full_data.shape}")

    print("\n" + "=" * 80)
    print("Starting rule extraction...")
    start_time = time.time()

    clf = TabICLClassifier()
    X_train = X_features.iloc[:200, :]
    X_test = X_features.iloc[201:240, :]
    y_train = y_target.iloc[:200, :]
    clf.fit(X_features, y_target)  # this is cheap
    clf.predict(X_test)  # in-context learning happens here

    extracted_rules, feature_names = tabicl_multi_output_rule_learning(
        dataset=full_data,
        max_antecedents=2,
        ant_similarity=0.05,
        cons_similarity=0.8,
        checkpoint_path=None
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\n{'=' * 80}")
    print(f"RESULTS")
    print(f"{'=' * 80}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Extracted {len(extracted_rules)} rules")

    if len(extracted_rules) > 0:
        print("\n" + "=" * 80)
        print("CALCULATING RULE QUALITY METRICS...")
        print("=" * 80)

        rules_with_metrics, avg_metrics = calculate_rule_metrics(
            rules=extracted_rules,
            data=full_data.values,
            feature_names=feature_names,
            metric_names=['support', 'confidence', 'rule_coverage', 'zhangs_metric']
        )

        print("\nFirst 10 rules with metrics:")
        for idx, rule in enumerate(rules_with_metrics[:10], 1):
            antecedents_str = " AND ".join([f"{a['feature']}={a['value']}" for a in rule['antecedents']])
            consequent_str = f"{rule['consequent']['feature']}={rule['consequent']['value']}"
            print(f"{idx}. {antecedents_str} => {consequent_str}")
            print(f"    Support: {rule['support']:.4f}, Confidence: {rule['confidence']:.4f}, "
                  f"Coverage: {rule['rule_coverage']:.4f}, Zhang: {rule['zhangs_metric']:.4f}")

        print("\n" + "=" * 80)
        print("RULE QUALITY METRICS (Averages)")
        print("=" * 80)

        for metric_name, value in avg_metrics.items():
            print(f"{metric_name}: {value:.4f}")

    else:
        print("\nWARNING: No rules extracted!")
