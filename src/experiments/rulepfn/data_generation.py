"""
tabpfn_like_generator.py
Generate many synthetic tabular datasets inspired by the TabPFN prior.
"""

import json
import uuid
import random
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.utils import shuffle


# -------------------------
# Utility functions
# -------------------------
def choice_weighted(items, weights):
    return items[np.argmax(np.random.multinomial(1, np.array(weights) / sum(weights)))]


# -------------------------
# Random graph (DAG) generator
# -------------------------
def sample_dag(num_features, edge_prob=0.2, max_parents=3, acyclic=True):
    """Sample a random DAG using Erdős–Rényi style then enforce acyclicity by ordering."""
    nodes = list(range(num_features))
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    # choose an ordering
    order = nodes.copy()
    random.shuffle(order)
    # allow edges only from earlier to later in order to ensure acyclic
    for i, u in enumerate(order):
        possible_parents = order[:i]
        if not possible_parents:
            continue
        num_parents = np.random.randint(0, min(max_parents, len(possible_parents)) + 1)
        parents = np.random.choice(possible_parents, size=num_parents, replace=False)
        for p in parents:
            if random.random() < edge_prob:
                G.add_edge(p, u)
    return G


# -------------------------
# Edge functions
# -------------------------
def apply_edge_function(parent_val, func_type, coef=1.0, bias=0.0):
    if func_type == "linear":
        return coef * parent_val + bias
    if func_type == "poly2":
        return coef * parent_val + 0.5 * coef * (parent_val ** 2) + bias
    if func_type == "sigmoid":
        return coef * (1 / (1 + np.exp(-parent_val))) + bias
    if func_type == "sin":
        return coef * np.sin(parent_val) + bias
    if func_type == "relu":
        return coef * np.maximum(0, parent_val) + bias
    if func_type == "tanh":
        return coef * np.tanh(parent_val) + bias
    # fallback
    return coef * parent_val + bias


EDGE_FUNCTIONS = ["linear", "poly2", "sigmoid", "sin", "relu", "tanh"]


# -------------------------
# Feature sampler
# -------------------------
def sample_root_marginal(n, marginal_type):
    if marginal_type == "normal":
        return np.random.normal(loc=0.0, scale=1.0, size=n)
    if marginal_type == "uniform":
        return np.random.uniform(-1.0, 1.0, size=n)
    if marginal_type == "lognormal":
        return np.random.lognormal(mean=0.0, sigma=0.5, size=n)
    if marginal_type == "bernoulli":
        return np.random.binomial(1, 0.3, size=n).astype(float)
    if marginal_type == "mixture":
        # mixture of two normals
        mask = np.random.rand(n) < 0.5
        x = np.zeros(n)
        x[mask] = np.random.normal(-1, 0.5, size=mask.sum())
        x[~mask] = np.random.normal(1, 0.8, size=(~mask).sum())
        return x
    return np.random.normal(0, 1, size=n)


MARGINALS = ["normal", "uniform", "lognormal", "bernoulli", "mixture"]


# -------------------------
# Single dataset generator
# -------------------------
def generate_one_dataset(config, rng_seed=None):
    """Generates one synthetic dataset according to config and returns a dict with metadata + dataframe."""
    if rng_seed is not None:
        np.random.seed(rng_seed)
        random.seed(rng_seed)

    n = config.get("n_samples", 200)
    k = config.get("num_features", 10)
    edge_prob = config.get("edge_prob", 0.25)
    missing_rate = config.get("missing_rate", 0.0)
    cat_prob = config.get("cat_feature_prob", 0.2)
    max_cat_card = config.get("max_cat_card", 8)
    noise_scale = config.get("noise_scale", 0.1)
    allow_uninformative = config.get("uninformative_prob", 0.05)
    class_type = config.get("class_type", "binary")  # binary or multiclass
    num_classes = config.get("num_classes", 2)

    G = sample_dag(k, edge_prob=edge_prob)
    # decide which features are categorical
    is_categorical = [random.random() < cat_prob for _ in range(k)]
    cat_cardinalities = [random.randint(2, max_cat_card) if is_categorical[i] else None for i in range(k)]

    # For each node, sample function parameters for each parent.
    edge_specs = {}
    for (u, v) in G.edges():
        func = random.choice(EDGE_FUNCTIONS)
        coef = float(np.random.normal(1.0, 0.5))
        bias = float(np.random.normal(0.0, 0.1))
        edge_specs[(u, v)] = {"func": func, "coef": coef, "bias": bias}

    # Build feature matrix
    X = np.zeros((n, k), dtype=float)
    topo = list(nx.topological_sort(G))
    for node in topo:
        parents = list(G.predecessors(node))
        if len(parents) == 0:
            # root
            marg = random.choice(MARGINALS)
            vals = sample_root_marginal(n, marg)
            # scale to moderate range
            vals = (vals - vals.mean()) / (vals.std() + 1e-8)
            X[:, node] = vals
        else:
            # combine parents' effects
            agg = np.zeros(n, dtype=float)
            for p in parents:
                spec = edge_specs[(p, node)]
                parent_vals = X[:, p]
                agg += apply_edge_function(parent_vals, spec["func"], spec["coef"], spec["bias"])
            # add heteroscedastic Gaussian noise
            local_noise = np.random.normal(0, noise_scale * (1.0 + np.abs(agg)), size=n)
            X[:, node] = (agg + local_noise)
            # normalize
            X[:, node] = (X[:, node] - X[:, node].mean()) / (X[:, node].std() + 1e-8)

    # Optionally add some uninformative (pure noise) features
    for i in range(k):
        if random.random() < allow_uninformative:
            X[:, i] = np.random.normal(size=n)

    # Convert some features into categorical by binning or discrete sampling
    df = pd.DataFrame()
    for i in range(k):
        if is_categorical[i]:
            card = cat_cardinalities[i]
            # either discrete from transformed continuous or sample categorical distribution
            if random.random() < 0.5:
                # k-means-ish binning by quantiles -> categories 0..card-1
                bins = np.quantile(X[:, i], np.linspace(0, 1, card + 1))
                cats = np.digitize(X[:, i], bins[1:-1], right=True)
            else:
                # sample discrete categories correlated weakly with variable using softmax
                logits = np.vstack(
                    [X[:, i] * (0.5 + np.random.rand()) + np.random.randn(n) * 0.1 + c for c in range(card)]).T
                probs = np.exp(logits - logits.max(axis=1, keepdims=True))
                probs = probs / probs.sum(axis=1, keepdims=True)
                cats = np.array([np.random.choice(card, p=p) for p in probs])
            df[f"x{i}"] = pd.Categorical(cats)
        else:
            df[f"x{i}"] = X[:, i]

    # Generate target (classification)
    # Create a latent score using a random subset of features and random coefficients
    use_feats = random.sample(range(k), max(1, int(0.5 * k)))
    coeffs = np.random.normal(0, 1.0, size=len(use_feats))
    latent = np.zeros(n)
    for idx, c in enumerate(use_feats):
        # if categorical, map to numeric embeddings
        if is_categorical[c]:
            emb = pd.get_dummies(df[f"x{c}"]).values.dot(np.random.normal(0, 1, size=cat_cardinalities[c]))
            latent += coeffs[idx] * emb
        else:
            latent += coeffs[idx] * df[f"x{c}"].values
    # add noise
    latent += np.random.normal(0, 0.5, size=n)

    if class_type == "binary":
        # logistic threshold
        probs = 1 / (1 + np.exp(-latent))
        y = (np.random.rand(n) < probs).astype(int)
    else:
        # multiclass: softmax over some random projections
        # generate K class logits
        K = num_classes
        logits = np.vstack([latent * (0.5 + np.random.rand()) + np.random.randn(n) * 0.3 + j for j in range(K)]).T
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        y = np.array([np.random.choice(K, p=p) for p in probs])

    df["y"] = y

    # Insert missing values at random
    if missing_rate > 0:
        for col in [c for c in df.columns if c != "y"]:
            mask = np.random.rand(n) < missing_rate
            # for categorical columns, set to NaN (pandas Categorical supports np.nan)
            df.loc[mask, col] = np.nan

    # Shuffle rows
    df = shuffle(df, random_state=np.random.randint(0, 2 ** 31))

    meta = {
        "n_samples": n,
        "num_features": k,
        "edge_count": len(G.edges()),
        "is_categorical": is_categorical,
        "cat_cardinalities": cat_cardinalities,
        "seed": rng_seed
    }
    return {"data": df, "meta": meta}


# -------------------------
# Batch generator + saving
# -------------------------
def save_dataset_sharded(out_dir, ds_index, ds_obj, fmt="parquet"):
    uid = uuid.uuid4().hex[:8]
    base = f"dataset_{ds_index:08d}_{uid}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # save data and metadata
    fmt = "csv"
    data_path = out_dir / f"{base}.parquet" if fmt == "parquet" else out_dir / f"{base}.csv"
    if fmt == "parquet":
        ds_obj["data"].to_parquet(data_path, index=False)
    else:
        ds_obj["data"].to_csv(data_path, index=False)
    meta_path = out_dir / f"{base}_meta.json"
    with open(meta_path, "w") as fh:
        json.dump(ds_obj["meta"], fh)
    return str(data_path), str(meta_path)


def work_fn(args):
    i, config, out_dir, fmt, start_index = args
    seed = np.random.randint(0, 2 ** 31)
    ds = generate_one_dataset(config, rng_seed=seed)
    return save_dataset_sharded(out_dir, start_index + i, ds, fmt=fmt)


def generate_many(out_dir, num_datasets=10000, config=None, use_multiprocessing=True, fmt="parquet", start_index=0):
    config = config or {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if use_multiprocessing:
        nproc = max(1, min(cpu_count(), 16))
        with Pool(nproc) as pool:
            args = [(i, config, out_dir, fmt, start_index) for i in range(num_datasets)]
            for i, res in enumerate(pool.imap_unordered(work_fn, args), 1):
                if i % 100 == 0:
                    print(f"generated {i}/{num_datasets} datasets")
    else:
        for i in range(num_datasets):
            work_fn(i)
            if (i + 1) % 100 == 0:
                print(f"generated {i + 1}/{num_datasets} datasets")


# -------------------------
# Example main: tune params here
# -------------------------
if __name__ == "__main__":
    OUT_DIR = "synthetic_tabpfn_like"
    NUM_DATASETS = 2  # set this to e.g. 1_000_000 in a cluster/batch run (watch disk)
    CONFIG = {
        "n_samples": 200,  # small-data focus like TabPFN
        "num_features": 12,
        "edge_prob": 0.25,
        "missing_rate": 0.05,
        "cat_feature_prob": 0.3,
        "max_cat_card": 6,
        "noise_scale": 0.15,
        "uninformative_prob": 0.05,
        "class_type": "binary",
        "num_classes": 2
    }

    # fmt="parquet"
    fmt = "csv"
    print(
        "Starting generation. This will produce many files — consider using a filesystem suited for many small files or change to a single sharded parquet dataset.")
    generate_many(OUT_DIR, num_datasets=NUM_DATASETS, config=CONFIG, use_multiprocessing=True, fmt=fmt)
    print("Done.")
