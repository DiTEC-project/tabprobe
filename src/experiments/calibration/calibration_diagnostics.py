"""
Calibration diagnostics for the TabProbe framework instantiations.
"""

import gc
import os
from datetime import datetime

import numpy as np
import pandas as pd

from src.utils import (
    get_ucimlrepo_datasets,
    generate_seed_sequence,
    prepare_categorical_data,
    add_gaussian_noise,
    generate_test_matrix,
    set_seed,
)
from src.utils.data_loading import UCIMLREPO_SMALL_DATASETS, UCIMLREPO_NORMAL_DATASETS


def _provider_tabpfn(context, query, fvi, seed, n_estimators=8, batch_size=4096,
                     max_workers=1, **_):
    import torch
    from src.experiments.rule_mining.tabpfn_experiments import _process_feature_tabpfn

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    noisy = add_gaussian_noise(context, noise_factor=0.5)
    shared = (noisy, context, query, query.shape[0], n_estimators, seed, device, batch_size)
    return _run_features(_process_feature_tabpfn, shared, query.shape, fvi, max_workers)


def _provider_tabicl(context, query, fvi, seed, n_estimators=8, batch_size=4096,
                     max_workers=1, **_):
    import torch
    from src.experiments.rule_mining.tabicl_experiments import _process_feature_tabicl

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    noisy = add_gaussian_noise(context, noise_factor=0.5)
    shared = (noisy, context, query, query.shape[0], n_estimators, seed, device, batch_size)
    return _run_features(_process_feature_tabicl, shared, query.shape, fvi, max_workers)


def _provider_tabdpt(context, query, fvi, seed, n_ensembles=8, batch_size=4096,
                     max_workers=1, **_):
    from src.experiments.rule_mining.tabdpt_experiments import _process_feature_tabdpt

    if max_workers != 1:
        print(f"    ignoring max_workers={max_workers} for TabDPT and running single-threaded: "
              f"torch.compile/dynamo is not thread-safe")
    noisy = add_gaussian_noise(context, noise_factor=0.5)
    shared = (noisy, context, query, query.shape[0], n_ensembles, len(context), seed, batch_size)
    return _run_features(_process_feature_tabdpt, shared, query.shape, fvi, 1)


def _provider_xgboost(context, query, fvi, seed, dataset_name=None, batch_size=4096,
                      max_workers=1, calibrate=False, calibration_cv=5,
                      calibration_method='sigmoid', **_):
    import json
    from src.experiments.rule_mining.xgboost_experiments import _process_feature_xgb

    try:
        import torch as _t
        device = 'cuda' if _t.cuda.is_available() else 'cpu'
    except ImportError:
        device = 'cpu'

    n_estimators, max_depth, learning_rate = 100, 3, 0.3
    params_json = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "xgboost_best_parameters.json"))
    if os.path.exists(params_json) and dataset_name:
        with open(params_json) as f:
            tuned = json.load(f).get(dataset_name, {})
        n_estimators = tuned.get('n_estimators', n_estimators)
        max_depth = tuned.get('max_depth', max_depth)
        learning_rate = tuned.get('learning_rate', learning_rate)

    shared = (context, query, query.shape[0], n_estimators, max_depth, learning_rate,
              seed, batch_size, device, calibrate, calibration_cv, calibration_method)
    return _run_features(_process_feature_xgb, shared, query.shape, fvi, max_workers)


def _provider_xgboost_calibrated(context, query, fvi, seed, dataset_name=None, **kwargs):
    """XGBoost under xgboost_experiments.py's CALIBRATE_DATASETS policy."""
    from src.experiments.rule_mining.xgboost_experiments import _resolve_calibration

    calibrate, calibration_method = _resolve_calibration(dataset_name)
    return _provider_xgboost(context, query, fvi, seed, dataset_name=dataset_name,
                             calibrate=calibrate, calibration_method=calibration_method, **kwargs)


def _provider_xgboost_calibrated_temperature(context, query, fvi, seed, dataset_name=None, **kwargs):
    """Same dataset gating as _provider_xgboost_calibrated, temperature scaling instead."""
    from src.experiments.rule_mining.xgboost_experiments import CALIBRATE_DATASETS

    calibrate = dataset_name in CALIBRATE_DATASETS
    return _provider_xgboost(context, query, fvi, seed, dataset_name=dataset_name,
                             calibrate=calibrate, calibration_method='temperature', **kwargs)


def _provider_random_forest(context, query, fvi, seed, dataset_name=None, batch_size=4096,
                            max_workers=1, **_):
    from src.utils import load_tuned_params
    from src.experiments.rule_mining.random_forest_experiments import _process_feature_rf

    n_estimators, max_depth, min_samples_leaf = 100, None, 1
    params_json = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "random_forest_best_parameters.json"))
    tuned = load_tuned_params(params_json, dataset_name)
    n_estimators = tuned.get('n_estimators', n_estimators)
    max_depth = tuned.get('max_depth', max_depth)
    min_samples_leaf = tuned.get('min_samples_leaf', min_samples_leaf)

    shared = (context, query, query.shape[0], n_estimators, max_depth, min_samples_leaf,
              seed, batch_size)
    return _run_features(_process_feature_rf, shared, query.shape, fvi, max_workers)


def _provider_aerial(context, query, fvi, seed, dataset=None, dataset_size='small', **_):
    """Multi-target instantiation: one forward pass reconstructs every feature at once."""
    import torch
    from aerial import model as aerial_model
    from src.experiments.rule_mining.aerial_experiments import get_dataset_parameters

    params = get_dataset_parameters(dataset.attrs.get('name', ''), dataset_size)
    autoencoder = aerial_model.train(dataset, layer_dims=params['layer_dims'],
                                     epochs=params['epochs'], batch_size=params['batch_size'])
    if autoencoder is None:
        raise RuntimeError("aerial training failed (non-categorical columns?)")

    device = next(autoencoder.parameters()).device
    autoencoder.eval()
    with torch.no_grad():
        x = torch.from_numpy(query.astype(np.float32)).to(device)
        out = autoencoder(x, [range(f['start'], f['end']) for f in fvi]).cpu().numpy()
    del autoencoder
    gc.collect()
    return out


def _run_features(fn, shared, query_shape, fvi, max_workers):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    probs = np.zeros(query_shape)
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for fut in as_completed([ex.submit(fn, fi, *shared) for fi in fvi]):
                s, e, p = fut.result()
                probs[:, s:e] = p
    else:
        for fi in fvi:
            s, e, p = fn(fi, *shared)
            probs[:, s:e] = p
    return probs


PROVIDERS = {
    'tabpfn': _provider_tabpfn,
    'tabicl': _provider_tabicl,
    'tabdpt': _provider_tabdpt,
    'xgboost': _provider_xgboost,
    'xgboost_calibrated': _provider_xgboost_calibrated,
    'xgboost_calibrated_temperature': _provider_xgboost_calibrated_temperature,
    'random_forest': _provider_random_forest,
    'aerial': _provider_aerial,
}

MULTI_TARGET = {'aerial'}


def empirical_conditionals(encoded, test_descriptions, fvi, min_support_count):
    """For each (probe, item) pair, the empirical conditional frequency of the item."""
    n_rows = encoded.shape[0]
    observed = np.stack([encoded[:, f['start']:f['end']].sum(axis=1) > 0 for f in fvi], axis=1)

    probe_idx, item_idx, k_arr, n_arr = [], [], [], []

    for pi, desc in enumerate(test_descriptions):
        marked = [(fvi[f]['start'] + c, f) for f, c in desc]
        marked_feats = {f for _, f in marked}

        mask_full = np.ones(n_rows, dtype=bool)
        for col, _ in marked:
            mask_full &= encoded[:, col] == 1

        for g, feat in enumerate(fvi):
            if g in marked_feats:
                continue
            mask = mask_full & observed[:, g]
            n_g = int(mask.sum())
            if n_g < min_support_count:
                continue
            counts = encoded[mask, feat['start']:feat['end']].sum(axis=0)
            for j, cnt in enumerate(counts):
                probe_idx.append(pi)
                item_idx.append(feat['start'] + j)
                k_arr.append(cnt)
                n_arr.append(n_g)

        for col, feat_idx in marked:
            mask = observed[:, feat_idx].copy()
            for c2, f2 in marked:
                if f2 != feat_idx:
                    mask &= encoded[:, c2] == 1
            n_cond = int(mask.sum())
            if n_cond < min_support_count:
                continue
            probe_idx.append(pi)
            item_idx.append(col)
            k_arr.append(int(encoded[mask, col].sum()))
            n_arr.append(n_cond)

    return (np.asarray(probe_idx, dtype=np.int64), np.asarray(item_idx, dtype=np.int64),
            np.asarray(k_arr, dtype=np.float64), np.asarray(n_arr, dtype=np.float64))


def observed_vs_predicted(method, dataset, encoded, fvi, classes_per_feature, seed,
                          max_antecedents, min_support_count, provider_kwargs):
    probe_matrix, descriptions, _ = generate_test_matrix(
        n_features=len(classes_per_feature),
        classes_per_feature=classes_per_feature,
        max_antecedents=max_antecedents,
        use_zeros_for_unmarked=False,
    )
    print(f"    {probe_matrix.shape[0]} probes")

    kwargs = dict(provider_kwargs)
    if method in MULTI_TARGET:
        kwargs['dataset'] = dataset
    probs = PROVIDERS[method](encoded, probe_matrix, fvi, seed, **kwargs)

    pi, ii, k, n = empirical_conditionals(encoded, descriptions, fvi, min_support_count)
    if len(pi) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([], dtype=bool)

    is_antecedent = np.zeros(len(pi), dtype=bool)
    for j, (probe, item) in enumerate(zip(pi, ii)):
        is_antecedent[j] = item in {fvi[f]['start'] + c for f, c in descriptions[probe]}
    return probs[pi, ii], k, n, is_antecedent


def _bins(p, k, n, n_bins):
    """Per-bin trial counts, mean prediction and observed frequency."""
    edges = np.linspace(0, 1, n_bins + 1)
    binids = np.searchsorted(edges[1:-1], p)
    total = np.bincount(binids, weights=n, minlength=n_bins)
    pred = np.bincount(binids, weights=p * n, minlength=n_bins)
    obs = np.bincount(binids, weights=k, minlength=n_bins)
    nz = total > 0
    return total[nz], pred[nz] / total[nz], obs[nz] / total[nz]


def reliability_curve(p, k, n, n_bins=10):
    """Observed versus predicted per bin: the data behind the reliability diagrams."""
    total, mean_p, obs = _bins(p, k, n, n_bins)
    return pd.DataFrame({
        'mean_predicted': mean_p, 'observed_frequency': obs, 'n_trials': total,
    })


def calibration_metrics(p, k, n, n_bins=10):
    """ECE, Brier and the Murphy decomposition Brier = reliability - resolution + uncertainty."""
    if len(p) == 0 or n.sum() == 0:
        return {'n_pairs': 0}

    total_trials = n.sum()
    total, mean_p, obs = _bins(p, k, n, n_bins)
    base_rate = k.sum() / total_trials

    return {
        'ece': float(np.average(np.abs(obs - mean_p), weights=total)),
        'brier': float((k * (1 - p) ** 2 + (n - k) * p ** 2).sum() / total_trials),
        'reliability': float((total * (mean_p - obs) ** 2).sum() / total_trials),
        'resolution': float((total * (obs - base_rate) ** 2).sum() / total_trials),
        'uncertainty': float(base_rate * (1 - base_rate)),
        'mean_predicted': float(np.average(p, weights=n)),
        'base_rate': float(base_rate),
        'n_pairs': int(len(p)),
        'n_trials': float(total_trials),
    }


def _encode(dataset):
    """Encode a dataset the way the method under test does."""
    if METHOD in MULTI_TARGET:
        from aerial.data_preparation import _one_hot_encoding_with_feature_tracking

        vectors, indices = _one_hot_encoding_with_feature_tracking(dataset.copy())
        if vectors is None:
            raise RuntimeError("aerial encoding failed (non-categorical columns?)")
        fvi = [{'start': c['start'], 'end': c['end'], 'feature': i}
               for i, c in enumerate(indices)]
        return (vectors.to_numpy(dtype=float), fvi,
                [c['end'] - c['start'] for c in indices])

    encoded, classes_per_feature, _, _ = prepare_categorical_data(dataset)
    starts = np.cumsum([0] + classes_per_feature[:-1])
    fvi = [{'start': int(s), 'end': int(s) + c, 'feature': i}
           for i, (s, c) in enumerate(zip(starts, classes_per_feature))]
    return encoded, fvi, classes_per_feature


def _write_outputs(summary_rows, curve_rows, raw, timestamp):
    xlsx = os.path.join(OUT_DIR, f"calibration_{METHOD}_{timestamp}.xlsx")
    with pd.ExcelWriter(xlsx, engine='openpyxl') as w:
        summary = pd.DataFrame(summary_rows)
        summary.to_excel(w, sheet_name='Summary', index=False)
        if not summary.empty:
            summary.drop(columns=['seed']).groupby(['method', 'dataset', 'size']) \
                .mean(numeric_only=True).reset_index().to_excel(w, sheet_name='Averaged', index=False)
        if curve_rows:
            pd.concat(curve_rows).to_excel(w, sheet_name='Reliability', index=False)
        pd.DataFrame([{
            'method': METHOD, 'dataset_size': DATASET_SIZE, 'n_seeds': N_SEEDS,
            'base_seed': BASE_SEED, 'max_antecedents': MAX_ANTECEDENTS,
            'min_support_count': MIN_SUPPORT_COUNT, 'n_bins': N_BINS,
            'max_workers': MAX_WORKERS, 'query_batch_size': QUERY_BATCH_SIZE,
        }]).to_excel(w, sheet_name='Parameters', index=False)

    npz = os.path.join(OUT_DIR, f"calibration_{METHOD}_{timestamp}_raw.npz")
    np.savez_compressed(npz, **raw)
    return xlsx, npz


def run():
    seeds = generate_seed_sequence(BASE_SEED, 10)[:N_SEEDS]
    sizes = ['small', 'normal'] if DATASET_SIZE == 'both' else [DATASET_SIZE]

    dataset_names = DATASETS
    if METHOD in ('xgboost_calibrated', 'xgboost_calibrated_temperature') and dataset_names is None:
        from src.experiments.rule_mining.xgboost_experiments import CALIBRATE_DATASETS
        dataset_names = sorted(CALIBRATE_DATASETS)

    os.makedirs(OUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_rows, curve_rows, raw = [], [], {}
    xlsx = npz = None

    for size in sizes:
        known = UCIMLREPO_SMALL_DATASETS if size == 'small' else UCIMLREPO_NORMAL_DATASETS
        names_for_size = [n for n in dataset_names if n in known] if dataset_names is not None else None
        if dataset_names is not None and not names_for_size:
            continue

        for info in get_ucimlrepo_datasets(size=size, names=names_for_size):
            name, df = info['name'], info['data']
            df.attrs['name'] = name

            try:
                encoded, fvi, classes_per_feature = _encode(df)

                print(f"\n{'=' * 70}\n{METHOD} | {name} ({size}) | "
                      f"{df.shape[0]} rows, {encoded.shape[1]} items\n{'=' * 70}")

                kwargs = {'max_workers': MAX_WORKERS, 'batch_size': QUERY_BATCH_SIZE,
                          'dataset_size': size}
                if METHOD in ('xgboost', 'xgboost_calibrated', 'xgboost_calibrated_temperature',
                              'random_forest'):
                    kwargs['dataset_name'] = name

                for seed in seeds:
                    set_seed(seed)
                    print(f"  seed={seed}")

                    p, k, n, is_ant = observed_vs_predicted(
                        METHOD, df, encoded, fvi, classes_per_feature, seed,
                        MAX_ANTECEDENTS, MIN_SUPPORT_COUNT, kwargs)
                    if len(p) == 0:
                        print("    no probes above the support floor, skipping")
                        continue

                    cons = ~is_ant
                    row = {'method': METHOD, 'dataset': name, 'size': size, 'seed': seed}
                    row.update(calibration_metrics(p[cons], k[cons], n[cons], N_BINS))
                    summary_rows.append(row)

                    curve = reliability_curve(p[cons], k[cons], n[cons], N_BINS)
                    for col, val in (('seed', seed), ('dataset', name), ('method', METHOD)):
                        curve.insert(0, col, val)
                    curve_rows.append(curve)

                    raw[f'{name}|{seed}|p'] = p.astype(np.float32)
                    raw[f'{name}|{seed}|k'] = k.astype(np.float32)
                    raw[f'{name}|{seed}|n'] = n.astype(np.float32)
                    raw[f'{name}|{seed}|is_ant'] = is_ant

                    print(f"    ECE={row['ece']:.4f}  Brier={row['brier']:.4f}  "
                          f"reliability={row['reliability']:.4f}  "
                          f"resolution={row['resolution']:.4f}")
                    gc.collect()

            except Exception as exc:
                print(f"  FAILED on {name}: {type(exc).__name__}: {exc}")

            xlsx, npz = _write_outputs(summary_rows, curve_rows, raw, timestamp)

    if xlsx is None:
        print("\nNo datasets were processed.")
        return
    print(f"\nSummary written to {xlsx}\nRaw pairs written to {npz}")


if __name__ == '__main__':
    METHOD = 'xgboost_calibrated'  # tabpfn, tabicl, tabdpt, xgboost, xgboost_calibrated,
    # xgboost_calibrated_temperature, random_forest, aerial
    DATASET_SIZE = 'both'  # small, normal, both
    DATASETS = None  # None = all datasets of the given size; for xgboost_calibrated(_temperature),
    # None auto-restricts to CALIBRATE_DATASETS in xgboost_experiments.py
    N_SEEDS = 10
    BASE_SEED = 42
    MAX_ANTECEDENTS = 2
    MIN_SUPPORT_COUNT = 10
    N_BINS = 10
    MAX_WORKERS = 4
    QUERY_BATCH_SIZE = 4096
    OUT_DIR = 'out/calibration'

    run()