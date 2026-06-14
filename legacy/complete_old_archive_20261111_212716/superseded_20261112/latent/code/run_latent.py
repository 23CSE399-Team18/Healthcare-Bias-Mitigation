"""Reproducible pre -> latent -> in-processing handoff. See PLAN.md."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import html
import json
import math
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (f1_score, roc_auc_score, average_precision_score,
    silhouette_score, normalized_mutual_info_score, adjusted_rand_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import MODELS, LATENT_DIM, DeepGCCA, encode_all, fit_gcca_projection

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT/'latent'
OUT = HERE/'outputs'
DATASETS = ('heart', 'respiratory', 'diabetes', 'hypertension', 'mimic')
METHODS = ('PCA', 'MV_DAE', 'DGCCA', 'SAINT')
SPLITS = ('train', 'validation', 'test')
SEED = 20261111
CONFIG = {'latent_dimensions': LATENT_DIM, 'seed': SEED, 'max_epochs': 8,
          'epoch_row_budget': 50000, 'patience': 3, 'batch_size': 256,
          'learning_rate': .001, 'weight_decay': .0001, 'validation_loss_sample': 4096,
          'metric_sample': 2000, 'saint_training_reference_rows': 32,
          'selection': 'highest validation probe macro-F1; no test scoring',
          'input_transform': 'training mean centering; scale=max(training SD,1)',
          'neural_training_supervision': 'none; no targets, sensitive metadata or domain IDs',
          'clinical_utility_guard': 'flag validation macro-F1 drop > 0.02 versus uncompressed probe'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding='utf-8')


def log(message):
    line = f'{datetime.now(timezone.utc).isoformat()} {message}'
    print(line, flush=True)
    HERE.mkdir(exist_ok=True)
    with (OUT/'progress.log').open('a', encoding='utf-8') as stream:
        stream.write(line+'\n')


def load_data(name):
    x, y, rows, fingerprints = {}, {}, {}, {}
    if name == 'mimic':
        base = ROOT/'preprocessing/outputs/cross_dataset_baseline/mimic_to_eicu'
        spec = json.loads((base/'feature_names.json').read_text())
        features, raw_features = spec['output_features'], spec['input_features']
        validation = json.loads((base/'transformation_validation.json').read_text())
        targets = ['heart', 'respiratory', 'diabetes', 'hypertension']
        for split in SPLITS:
            paths = [base/f'X_{split}.npz', base/f'y_{split}.npz', base/f'rows_{split}.csv']
            for path in paths:
                actual = sha(path)
                assert actual == validation['partitions'][split]['sha256'][path.name], path
                fingerprints[str(path.relative_to(ROOT))] = actual
            with np.load(paths[0], allow_pickle=False) as archive:
                x[split] = archive['X'].astype(np.float32)
            with np.load(paths[1], allow_pickle=False) as archive:
                y[split] = {t: archive[t].astype(np.int8) for t in targets}
            cols = pd.read_csv(paths[2], nrows=0).columns
            rows[split] = pd.read_csv(paths[2], dtype={c: 'string' for c in cols if c.startswith('meta_')})
    else:
        base = ROOT/'preprocessing/outputs/verified_baseline'/name
        validation = json.loads((base/'validation.json').read_text())
        features = validation['output_predictors']
        raw_features = validation['numeric']+validation['categorical']
        targets = [name]
        for split in SPLITS:
            csv = base/f'{name}_{split}_preprocessed.csv'
            actual = sha(csv)
            assert actual == validation['partitions'][split]['sha256'], csv
            fingerprints[str(csv.relative_to(ROOT))] = actual
            with np.load(base/f'X_{split}.npz', allow_pickle=False) as archive:
                x[split] = archive['X'].astype(np.float32)
            with np.load(base/f'y_{split}.npz', allow_pickle=False) as archive:
                y[split] = {name: archive['y'].astype(np.int8)}
            cols = pd.read_csv(csv, nrows=0).columns
            metadata = [c for c in cols if c.startswith(('meta_', 'sens_')) or c == 'target']
            idcols = [c for c in metadata if c in ('meta_record_id','meta_patientunitstayid','meta_split_group_uniquepid')]
            rows[split] = pd.read_csv(csv, usecols=metadata, dtype={c: 'string' for c in idcols})
    for split in SPLITS:
        assert x[split].shape == (len(rows[split]), len(features)) and np.isfinite(x[split]).all()
        for target in targets:
            column = 'target_'+target if name == 'mimic' else 'target'
            np.testing.assert_array_equal(rows[split][column], y[split][target])
        rows[split]['meta_latent_row'] = np.arange(len(rows[split]))
        rows[split]['meta_dataset'] = name
        rows[split]['meta_split'] = split
    return {'name': name, 'X': x, 'y': y, 'rows': rows, 'features': features,
            'raw_features': raw_features, 'targets': targets, 'input_sha256': fingerprints}


def feature_groups(data):
    groups = {c: [] for c in data['raw_features']}
    for i, column in enumerate(data['features']):
        short = column.split('__', 1)[-1]
        short = short.removeprefix('missingindicator_')
        candidates = [c for c in groups if short == c or short.startswith(c+'_')]
        assert candidates, column
        groups[max(candidates, key=len)].append(i)
    groups = {k: v for k, v in groups.items() if v}
    assert sorted(sum(groups.values(), [])) == list(range(len(data['features'])))
    if data['name'] == 'mimic':
        definitions = {'heart_rate': [], 'respiratory_oxygenation': [], 'demographics': []}
        for name, indices in groups.items():
            group = 'heart_rate' if name.startswith('heart_rate') else ('respiratory_oxygenation' if name.startswith(('resp_rate','spo2')) else 'demographics')
            definitions[group] += indices
    else:
        demographics = ('age','male','education','income','smok','ecig','alcohol','drink','exercise',
                        'marital','us_born','worked','insurance','cost_barrier','food_insecurity','usual_place')
        definitions = {'demographic_lifestyle_access': [], 'clinical_measurements_history': []}
        for name, indices in groups.items():
            group = 'demographic_lifestyle_access' if name.startswith(demographics) else 'clinical_measurements_history'
            definitions[group] += indices
    assert min(map(len, definitions.values())) > 0
    return list(groups.values()), list(definitions.values()), {'tokens': groups, 'paired_views': definitions}


def linear_cka(a, b):
    assert len(a) == len(b)
    a, b = a.astype(float)-a.mean(0), b.astype(float)-b.mean(0)
    numerator = np.linalg.norm(a.T@b, 'fro')**2
    denominator = np.linalg.norm(a.T@a, 'fro')*np.linalg.norm(b.T@b, 'fro')
    return float(numerator/denominator) if denominator > 0 else None


def sample_indices(y, n, seed=SEED):
    if len(y) <= n:
        return np.arange(len(y))
    ids, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    return ids


def probe_metrics(data, ztrain, zval, destination):
    result = {}
    for target in data['targets']:
        ytr, yv = data['y']['train'][target], data['y']['validation'][target]
        scale = StandardScaler().fit(ztrain)
        model = LogisticRegression(C=1., class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=SEED)
        with warnings.catch_warnings(record=True) as caught:
            model.fit(scale.transform(ztrain), ytr)
        assert not any('converg' in str(w.message).lower() for w in caught), f'{target}: probe convergence'
        p = model.predict_proba(scale.transform(zval))[:, 1]
        pred = (p >= .5).astype(int)
        result[target] = {'macro_f1': f1_score(yv, pred, average='macro', zero_division=0),
                          'auroc': roc_auc_score(yv, p), 'average_precision': average_precision_score(yv, p),
                          'validation_rows': len(yv), 'validation_positive_rows': int(yv.sum()), 'threshold': .5}
        if destination:
            joblib.dump({'scaler': scale, 'classifier': model, 'target': target}, destination/f'probe_{target}.joblib')
    return result


def train_neural(data, method, inputs, groups, views, folder):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    ref_ids = rng.choice(len(inputs['train']), min(CONFIG['saint_training_reference_rows'], len(inputs['train'])), replace=False)
    model = MODELS[method](inputs['train'].shape[1], views, groups, inputs['train'][ref_ids])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    train = inputs['train']
    order = rng.permutation(len(train))
    cursor, cycle = 0, 0
    val_ids = rng.choice(len(inputs['validation']), min(CONFIG['validation_loss_sample'], len(inputs['validation'])), replace=False)
    val = inputs['validation'][val_ids]
    best, best_state, best_epoch, stale = float('inf'), None, 0, 0
    history, visited = [], np.zeros(len(train), dtype=bool)
    min_epochs = math.ceil(len(train)/CONFIG['epoch_row_budget'])
    for epoch in range(CONFIG['max_epochs']):
        started = time.time()
        take = min(CONFIG['epoch_row_budget'], len(train))
        if cursor+take <= len(order):
            indices = order[cursor:cursor+take]
            cursor += take
        else:
            tail = order[cursor:]
            order = rng.permutation(len(train)); cursor = take-len(tail); cycle += 1
            indices = np.concatenate([tail, order[:cursor]])
        visited[indices] = True
        model.train()
        losses = []
        for start in range(0, len(indices), CONFIG['batch_size']):
            ix = indices[start:start+CONFIG['batch_size']]
            if len(ix) < 16:
                ix = np.r_[ix, indices[:16-len(ix)]]
            batch = torch.from_numpy(train[ix])
            optimizer.zero_grad(set_to_none=True)
            loss = model.objective(batch, corrupt=True)
            assert torch.isfinite(loss), (method, epoch, 'nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            values = [float(model.objective(torch.from_numpy(val[s:s+CONFIG['batch_size']]), corrupt=False))
                      for s in range(0, len(val), CONFIG['batch_size']) if len(val[s:s+CONFIG['batch_size']]) >= 16]
        score = float(np.mean(values))
        if epoch+1 >= min_epochs and score < best-1e-6:
            best, best_epoch, stale = score, epoch+1, 0
            best_state = copy.deepcopy(model.state_dict())
        elif epoch+1 >= min_epochs:
            stale += 1
        row = {'epoch': epoch+1, 'training_loss': np.mean(losses), 'validation_objective': score,
               'unique_training_rows_seen': int(visited.sum()), 'seconds': time.time()-started}
        history.append(row)
        write_json(folder/'training_progress.json', {'method': method, 'history': history, 'best_epoch': best_epoch})
        log(f"{data['name']} {method}: epoch {epoch+1}/{CONFIG['max_epochs']}, validation objective {score:.5f}, {row['seconds']:.1f}s")
        if epoch+1 >= min_epochs and stale >= CONFIG['patience']:
            break
    assert visited.all(), 'training budget must visit every training record at least once'
    model.load_state_dict(best_state)
    model.eval()
    projection = fit_gcca_projection(model, train) if method == 'DGCCA' else None
    # A fixed encoder must be independent of query batch composition and order.
    whole = encode_all(model, inputs['validation'][:23], projection, batch_size=23)
    single = encode_all(model, inputs['validation'][:23], projection, batch_size=1)
    np.testing.assert_allclose(whole, single, atol=3e-5, rtol=3e-5)
    torch.save({'state_dict': model.state_dict(), 'method': method, 'dimension': train.shape[1],
                'groups': groups, 'views': views, 'reference_indices': ref_ids.tolist(), 'config': CONFIG}, folder/'encoder.pt')
    joblib.dump(projection, folder/'gcca_projection.joblib')
    write_json(folder/'training.json', {'history': history, 'best_epoch': best_epoch,
        'fit_rows': len(train), 'unique_training_rows_seen': int(visited.sum()),
        'validation_checkpoint_rows': len(val_ids), 'test_rows_used_for_fitting_or_selection': 0,
        'labels_used_for_encoder_training': False, 'batch_invariance_passed': True,
        'reference_rows_source': 'training only' if method == 'SAINT' else None})
    return model, projection


def anchor_groups(rows):
    age = rows.sens_age_group.astype('string')
    sex = rows.sens_sex.astype('string')
    return (age+'|'+sex).fillna('Unknown').to_numpy(str)


def fit_anchor_map(z, rows):
    groups = anchor_groups(rows)
    means, target, labels, counts = [], [], [], []
    for ai, age in enumerate(('18-39','40-64','65-79','80+')):
        for sex in ('Female','Male'):
            key = age+'|'+sex
            mask = groups == key
            if mask.sum() < 20:
                continue
            a, s = (ai-1.5)/1.11803398875, (-1. if sex == 'Female' else 1.)
            means.append(z[mask].mean(0)); target.append([a, s, a*s, a*a-1])
            labels.append(key); counts.append(int(mask.sum()))
    assert len(means) >= 6, 'insufficient shared demographic groups for anchor diagnostic'
    mapper = Ridge(alpha=.1).fit(np.asarray(means), np.asarray(target))
    return mapper, {'groups': labels, 'training_counts': counts, 'dimensions': 4,
        'purpose': 'demographic alignment diagnostic only; not default in-processing features',
        'fitted_on': 'training age/sex group means; no outcomes or race', 'paired_people_across_sources': False}


def evaluate_method(data, method, z, folder):
    metrics = {'method': method, 'probe': probe_metrics(data, z['train'], z['validation'], folder)}
    primary = np.mean([v['macro_f1'] for v in metrics['probe'].values()])
    metrics['mean_validation_macro_f1'] = float(primary)
    ix = np.random.default_rng(SEED).choice(len(z['validation']), min(CONFIG['metric_sample'], len(z['validation'])), replace=False)
    metrics['same_row_input_latent_cka'] = linear_cka(data['X']['validation'][ix], z['validation'][ix])
    metrics['cka_rows'] = len(ix)
    for target in data['targets']:
        yi = data['y']['validation'][target]
        indices = sample_indices(yi, CONFIG['metric_sample'])
        metrics['probe'][target]['silhouette'] = float(silhouette_score(z['validation'][indices], yi[indices]))
        metrics['probe'][target]['silhouette_rows'] = len(indices)
    domains = {'heart':'meta_domain_state','hypertension':'meta_domain_region','respiratory':'meta_domain_hospitalid','mimic':'meta_domain_hospitalid'}
    domain = domains.get(data['name'])
    if domain is None or data['rows']['train'][domain].nunique() <= 1:
        metrics['domain_cluster_nmi'] = None
        metrics['domain_cluster_ari'] = None
        metrics['domain_metric_status'] = 'not applicable: no domain variable or only one institution'
    else:
        k = min(8, int(data['rows']['train'][domain].nunique()))
        clustering = MiniBatchKMeans(n_clusters=k, random_state=SEED, n_init=5, batch_size=2048).fit(z['train'])
        assigned = clustering.predict(z['validation'])
        truth = data['rows']['validation'][domain].fillna('Unknown').astype(str).to_numpy()
        metrics['domain_cluster_nmi'] = float(normalized_mutual_info_score(truth, assigned))
        metrics['domain_cluster_ari'] = float(adjusted_rand_score(truth, assigned))
        metrics['domain_metric_status'] = f'training-fitted {k} clusters versus validation {domain}; unweighted'
        joblib.dump(clustering, folder/'domain_clusters.joblib')
    metrics['test_metrics'] = 'not computed; test retained for final in-processing evaluation'
    return metrics


def dataset_run(name):
    log(f'Loading verified preprocessed dataset: {name}')
    data = load_data(name)
    folder = OUT/name
    folder.mkdir(parents=True, exist_ok=True)
    groups, views, definition = feature_groups(data)
    write_json(folder/'feature_views.json', definition)
    mean = data['X']['train'].mean(0, dtype=np.float64)
    scale = np.maximum(data['X']['train'].std(0, dtype=np.float64), 1.)
    inputs = {s: ((x-mean)/scale).astype(np.float32) for s, x in data['X'].items()}
    joblib.dump({'mean': mean, 'scale': scale, 'fit_rows': len(inputs['train'])}, folder/'input_normalization.joblib')
    manifest = {'dataset': name, 'input_sha256': data['input_sha256'], 'targets': data['targets'],
                'config': CONFIG, 'script_sha256': sha(Path(__file__)), 'model_code_sha256': sha(HERE/'code/models.py'),
                'metadata': {}, 'methods': {}}
    for split in SPLITS:
        path = folder/f'rows_{split}.csv'
        data['rows'][split].to_csv(path, index=False)
        np.savez_compressed(folder/f'y_{split}.npz', **data['y'][split])
        manifest['metadata'][split] = {'rows': len(data['rows'][split]), 'csv_sha256': sha(path)}
    baseline = probe_metrics(data, data['X']['train'], data['X']['validation'], None)
    manifest['uncompressed_probe'] = baseline
    manifest['uncompressed_mean_macro_f1'] = float(np.mean([v['macro_f1'] for v in baseline.values()]))
    baseline_metrics = {'method':'uncompressed', 'mean_validation_macro_f1':manifest['uncompressed_mean_macro_f1'],
                        'same_row_input_latent_cka':1., 'probe':baseline}
    methods = {}
    for method in METHODS:
        dest = folder/method
        dest.mkdir(exist_ok=True)
        log(f'{name}: fitting {method} on training rows only')
        if method == 'PCA':
            model = PCA(n_components=LATENT_DIM, svd_solver='full', random_state=SEED).fit(inputs['train'])
            raw_z = {s: model.transform(x) for s, x in inputs.items()}
            joblib.dump(model, dest/'encoder.joblib')
            write_json(dest/'training.json', {'fit_rows': len(inputs['train']), 'test_rows_used_for_fitting_or_selection':0,
                'explained_variance_ratio': model.explained_variance_ratio_, 'labels_used_for_encoder_training': False})
        else:
            model, projection = train_neural(data, method, inputs, groups, views, dest)
            raw_z = {s: encode_all(model, x, projection) for s, x in inputs.items()}
        scaler = StandardScaler().fit(raw_z['train'])
        z = {s: scaler.transform(a).astype(np.float32) for s, a in raw_z.items()}
        assert all(np.isfinite(a).all() and a.shape == (len(data['X'][s]), LATENT_DIM) for s, a in z.items())
        assert np.linalg.matrix_rank(z['train'][:min(len(z['train']), 10000)], tol=1e-4) >= LATENT_DIM-1, 'collapsed representation'
        joblib.dump(scaler, dest/'latent_scaler.joblib')
        mapper, anchor_info = fit_anchor_map(z['train'], data['rows']['train'])
        joblib.dump(mapper, dest/'anchor_projection.joblib')
        write_json(dest/'anchor_info.json', anchor_info)
        artifacts = {}
        for split, values in z.items():
            path = dest/f'Z_{split}.npz'
            np.savez_compressed(path, Z=values)
            np.savez_compressed(dest/f'Z_anchor_{split}.npz', Z=mapper.predict(values).astype(np.float32))
            with np.load(path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive['Z'], values)
            artifacts[split] = {'rows': len(values), 'dimensions': LATENT_DIM, 'sha256': sha(path),
                                 'metadata_csv_sha256': manifest['metadata'][split]['csv_sha256']}
        metrics = evaluate_method(data, method, z, dest)
        metrics['validation_macro_f1_change_vs_uncompressed'] = metrics['mean_validation_macro_f1']-manifest['uncompressed_mean_macro_f1']
        metrics['clinical_utility_guard_passed'] = metrics['validation_macro_f1_change_vs_uncompressed'] >= -.02
        write_json(dest/'metrics.json', metrics)
        write_json(dest/'artifacts.json', artifacts)
        methods[method] = metrics
        manifest['methods'][method] = {'metrics': metrics, 'artifacts': artifacts}
        write_json(folder/'manifest.json', manifest)
        log(f"{name} {method}: validation macro-F1 {metrics['mean_validation_macro_f1']:.4f}, change {metrics['validation_macro_f1_change_vs_uncompressed']:+.4f}")
    selected = max(METHODS, key=lambda m: methods[m]['mean_validation_macro_f1'])
    status = 'ready' if methods[selected]['clinical_utility_guard_passed'] else 'ready_with_utility_warning'
    manifest['selection'] = {'method': selected, 'rule': CONFIG['selection'], 'status': status,
                             'clinical_utility_guard_passed': methods[selected]['clinical_utility_guard_passed'],
                             'note': 'All candidate embeddings retained. No claim of clinical safety or fairness improvement.'}
    selected_folder = folder/'selected'
    selected_folder.mkdir(exist_ok=True)
    for split in SPLITS:
        with np.load(folder/selected/f'Z_{split}.npz') as archive:
            values = archive['Z']
        np.savez_compressed(selected_folder/f'Z_{split}.npz', Z=values)
        frame = pd.concat([pd.DataFrame(values, columns=[f'z{i+1:02}' for i in range(LATENT_DIM)]), data['rows'][split]], axis=1)
        path = selected_folder/f'{name}_{split}_latent.csv'
        frame.to_csv(path, index=False)
        loaded = pd.read_csv(path, usecols=[f'z{i+1:02}' for i in range(LATENT_DIM)])
        np.testing.assert_allclose(loaded, values, atol=1e-7, rtol=1e-7)
        manifest['selection'][split] = {'csv':str(path.relative_to(ROOT)), 'sha256':sha(path), 'rows':len(frame)}
    write_json(folder/'manifest.json', manifest)
    plot_dataset(name, baseline_metrics, methods)
    return manifest


def plot_dataset(name, baseline, methods):
    method_names=list(methods)
    short_labels={m:m.replace('pct_variance','% var').replace('PCA_','PCA ') for m in method_names}
    for m in method_names:
        dimensions=json.loads((OUT/name/m/'artifacts.json').read_text())['train']['dimensions']
        short_labels[m]+=f'\n({dimensions}D)'
    labels = ['uncompressed']+[short_labels[m] for m in method_names]
    vals = [baseline['mean_validation_macro_f1']]+[methods[m]['mean_validation_macro_f1'] for m in method_names]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0,0].bar(labels, vals, color=['#687586']+['#3569a5']*len(method_names))
    axes[0,0].set(title='Information preservation: validation Macro-F1', ylim=(0,1), ylabel='Macro-F1 (equal task average for MIMIC)')
    for i,v in enumerate(vals): axes[0,0].text(i,v+.015,f'{v:.3f}',ha='center',fontsize=9)
    axes[0,1].bar([short_labels[m] for m in method_names], [methods[m]['same_row_input_latent_cka'] for m in method_names], color='#318b71')
    axes[0,1].set(title='CKA: same validation rows, input vs latent', ylim=(0,1), ylabel='Linear CKA; not cross-patient alignment')
    targets = list(methods[method_names[0]]['probe'])
    if len(targets) == 1:
        # A shared x coordinate hid methods with similar scores behind each other.
        positions = np.arange(len(method_names))
        values = [methods[m]['probe'][targets[0]]['silhouette'] for m in method_names]
        axes[1,0].scatter(positions, values, s=45, color='#3569a5', zorder=3)
        axes[1,0].set_xticks(positions, [short_labels[m] for m in method_names])
        axes[1,0].margins(x=.12, y=.18)
        for i, value in enumerate(values):
            axes[1,0].annotate(f'{value:.4f}', (i, value), xytext=(0, 8),
                               textcoords='offset points', ha='center', fontsize=8)
    else:
        # Different diagnoses are categories: offset markers to keep every method
        # visible without implying a continuous trend between different tasks.
        positions = np.arange(len(targets))
        offsets = np.linspace(-.27, .27, len(method_names))
        for offset, m in zip(offsets, method_names):
            values = [methods[m]['probe'][target]['silhouette'] for target in targets]
            axes[1,0].scatter(positions + offset, values, s=35,
                              label=short_labels[m].replace('\n', ' '))
        axes[1,0].set_xticks(positions, targets)
        axes[1,0].margins(x=.08, y=.12)
        axes[1,0].legend(fontsize=8)
    axes[1,0].axhline(0,color='gray',lw=.8)
    axes[1,0].set(title='Clinical-class silhouette (validation sample)', ylabel='Silhouette; negative scores are retained')
    valid = [m for m in method_names if methods[m]['domain_cluster_nmi'] is not None]
    if valid:
        pos=np.arange(len(valid))
        axes[1,1].bar(pos-.18,[methods[m]['domain_cluster_nmi'] for m in valid],.36,label='NMI')
        axes[1,1].bar(pos+.18,[methods[m]['domain_cluster_ari'] for m in valid],.36,label='ARI')
        axes[1,1].set_xticks(pos,[short_labels[m] for m in valid]); axes[1,1].legend()
        axes[1,1].set(title='Clusters vs observed domain identity',ylabel='Validation association; not a fairness score')
    else:
        axes[1,1].axis('off')
        axes[1,1].text(.5,.5,'Domain NMI / ARI: not applicable\nNo domain ID or only one institution',ha='center',va='center',transform=axes[1,1].transAxes)
    for ax in axes.flat: ax.tick_params(axis='x',rotation=25,labelsize=8)
    fig.suptitle(f'{name.upper() if name=="mimic" else name.title()} | latent method comparison (validation only)',fontsize=15)
    (HERE/'plots'/name).mkdir(parents=True,exist_ok=True)
    fig.savefig(HERE/'plots'/name/'comparison.png',dpi=140,bbox_inches='tight'); plt.close(fig)


def cross_dataset_diagnostics():
    report = {}
    for method in METHODS:
        train, val, labels, means, group_labels = [], [], [], {}, {}
        for i, name in enumerate(DATASETS):
            folder = OUT/name
            for split, destination in (('train',train),('validation',val)):
                with np.load(folder/method/f'Z_anchor_{split}.npz') as a:
                    z = a['Z']
                ix = np.random.default_rng(SEED).choice(len(z), min(1000,len(z)), replace=False)
                destination.append(z[ix])
                if split == 'validation': labels.extend([i]*len(ix))
            rows = pd.read_csv(folder/'rows_validation.csv',usecols=['sens_age_group','sens_sex'])
            groups = anchor_groups(rows)
            with np.load(folder/method/'Z_validation.npz') as a:
                z = a['Z']
            means[name] = {k:z[groups==k].mean(0) for k in sorted(set(groups)) if (groups==k).sum() >= 20 and 'Unknown' not in k and '<NA>' not in k}
        clustering = MiniBatchKMeans(n_clusters=5,n_init=5,batch_size=2048,random_state=SEED).fit(np.concatenate(train))
        assigned = clustering.predict(np.concatenate(val))
        nmi = normalized_mutual_info_score(labels,assigned)
        ari = adjusted_rand_score(labels,assigned)
        rng=np.random.default_rng(SEED)
        perms=[normalized_mutual_info_score(rng.permutation(labels),assigned) for _ in range(100)]
        cka = np.ones((len(DATASETS),len(DATASETS)))
        support = np.zeros_like(cka,dtype=int)
        for i,a in enumerate(DATASETS):
            for j,b in enumerate(DATASETS):
                keys=sorted(set(means[a])&set(means[b]))
                support[i,j]=len(keys)
                cka[i,j]=linear_cka(np.array([means[a][k] for k in keys]),np.array([means[b][k] for k in keys])) if len(keys)>=4 else np.nan
        report[method]={'pooled_anchor_source_nmi':nmi,'pooled_anchor_source_ari':ari,
            'nmi_permutation_mean':np.mean(perms),'nmi_permutation_95pct':np.quantile(perms,.95),
            'validation_rows':len(labels),'dataset_order':DATASETS,'group_centroid_cka':cka,'paired_group_counts':support,
            'interpretation':'Demographic group alignment only; no paired people, common disease label or clinical equivalence implied'}
        joblib.dump(clustering,OUT/f'{method}_anchor_clusters.joblib')
    write_json(OUT/'cross_dataset_diagnostics.json',report)
    fig,axes=plt.subplots(1,4,figsize=(17,4.8),constrained_layout=True)
    for ax,m in zip(axes,METHODS):
        a=np.array(report[m]['group_centroid_cka'])
        im=ax.imshow(a,vmin=0,vmax=1,cmap='Blues')
        ax.set_xticks(range(5),DATASETS,rotation=45,ha='right');ax.set_yticks(range(5),DATASETS)
        ax.set_title(m)
        for i in range(5):
            for j in range(5):ax.text(j,i,f'{a[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if a[i,j]>.6 else 'black')
    fig.suptitle('Initial 8D candidates: validation age/sex-group centroid CKA | not paired-patient alignment',fontsize=13)
    fig.colorbar(im,ax=axes,shrink=.75)
    (HERE/'plots').mkdir(parents=True,exist_ok=True)
    fig.savefig(HERE/'plots/group_alignment.png',dpi=140,bbox_inches='tight');plt.close(fig)
    return report


def final_report(manifests, diagnostics):
    lines=['# Latent stage: results and in-processing handoff','',
        'The latent stage ran on the verified baseline preprocessing. It did not use the historical eight-bias outputs. All method selection below uses validation data; no test performance was calculated.','',
        '| Dataset | Training / validation / test rows | Selected latent method | Latent width | Validation Macro-F1 | Uncompressed reference | Utility guard |',
        '|---|---|---|---:|---:|---:|---|']
    for name in DATASETS:
        m=manifests[name];s=m['selection']['method'];v=m['methods'][s]['metrics']
        counts=' / '.join(f"{m['metadata'][p]['rows']:,}" for p in SPLITS)
        width=m['methods'][s]['artifacts']['train']['dimensions']
        lines.append(f"| {name} | {counts} | {s} | {width} | {v['mean_validation_macro_f1']:.4f} | {m['uncompressed_mean_macro_f1']:.4f} | {'pass' if v['clinical_utility_guard_passed'] else 'WARNING: >0.02 drop'} |")
    lines += ['', '## Which files to use','',
        'Use the complete split CSVs in `outputs/<dataset>/selected/` for in-processing. Only columns matching `^z[0-9]+$` are model inputs; the table above gives the actual width. `target` is the outcome for each original dataset; MIMIC instead has `target_heart`, `target_respiratory`, `target_diabetes`, and `target_hypertension`. Choose a separate prediction head/task for each MIMIC label. All `meta_*` and `sens_*` columns are metadata, not inputs.', '',
        'The authoritative compact arrays are `outputs/<dataset>/selected/Z_<split>.npz` (key `Z`). They align row-for-row with `outputs/<dataset>/rows_<split>.csv` and `y_<split>.npz`. Every output manifest records source hashes and row counts. All four candidate methods remain available in their own folders.', '',
        '**One pipeline does not mean one combined patient table.** Different datasets have different labels, populations and feature meanings. Keep their splits and tasks separate. MIMIC uses one feature population with four overlapping labels, not four independent copies of patients. The latent spaces are dataset-specific; a matching number of columns does not prove semantic alignment.', '',
        '## What was implemented','',
        '- DGCCA: within-patient paired feature views, MAXVAR correlation objective with a declared reconstruction auxiliary, followed by a global training-only GCCA projection. No arbitrary cross-database patient pairing.',
        '- SAINT-style: feature-token attention and intersample attention with a fixed bank of training rows, plus denoising and contrastive training. The fixed-bank induction and feature grouping are explicit adaptations, not an exact published benchmark replication.',
        '- MV-DAE: multiple feature-view encoders, masking and view dropout, a fused bottleneck, and reconstruction of all input views.',
        '- PCA: a linear baseline eligible for selection; a modern method is not forced to win.', '',
        '## Metrics and limits','',
        '- Macro-F1 is from fixed class-balanced logistic probes on validation labels. It measures predictive information for these recorded targets, not clinical truth. MIMIC selection averages its four task scores. Survey estimates here are unweighted; metadata preserves the original weights.',
        '- Silhouette uses clinical validation labels on a fixed stratified sample of up to 2,000 rows per task. CKA compares input and latent coordinates of the same validation people on up to 2,000 rows.',
        '- Cross-source CKA compares up to eight supported age/sex group means. It is a weak aggregate demographic diagnostic, not evidence of individual-patient clinical alignment. Do not use disease labels to match different tasks.',
        '- Domain NMI/ARI compare training-fitted clusters against validation domain IDs. Diabetes has no suitable domain ID; MIMIC has one institution, so these within-dataset metrics are NA. The separate pooled diagnostic uses a four-dimensional training-fitted demographic-anchor map, equal source sample sizes, and a permutation reference. Anchor coordinates are not the default next-stage features.',
        '- A single fixed seed and bounded training budget were used. This is a reproducible experiment, not proof of best possible model performance. All training rows were visited; validation selected checkpoints. Hyperparameter optimization, multi-seed confidence intervals and final in-processing performance remain separate work.',
        '- The final utility guard checks every target separately: no validation Macro-F1 decrease exceeding 0.02 versus its uncompressed reference. The initial eight-dimensional comparison was followed by a declared validation-only rescue using PCA retaining 95%, 99%, then 99.9% of training variance when necessary. This preserves useful information at a larger width. It is not a capacity-matched victory over the eight-dimensional neural candidates, nor an independent confirmatory study. Retain a no-latent comparator in the in-processing study. No fairness or clinical-safety improvement is claimed.',
        '- Original respiratory predicts ICU-cohort mortality; MIMIC respiratory predicts recorded respiratory diagnoses. Do not interchange these labels.', '',
        '## Reproduce and validate','',
        '`python latent/code/run_latent.py` trains, applies the utility guard and exports all five packages. `python latent/code/validate_latent.py` checks the handoff without fitting new models. The notebook `latent_pipeline.ipynb` is a single entry point.', '',
        'Method definitions and primary references are in [PLAN.md](../code/PLAN.md). Exact metrics, supports, checkpoints and hashes are in the dataset manifests and method folders.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    records=[]
    for n,m in manifests.items():
        for method,r in m['methods'].items():
            metrics=r['metrics']
            for target,v in metrics['probe'].items():
                records.append({'dataset':n,'method':method,'target':target,**v,
                    'same_row_cka':metrics['same_row_input_latent_cka'],'domain_nmi':metrics['domain_cluster_nmi'],
                    'domain_ari':metrics['domain_cluster_ari'],'selected':method==m['selection']['method']})
    for row in records:
        row['latent_dimensions']=manifests[row['dataset']]['methods'][row['method']]['artifacts']['train']['dimensions']
    pd.DataFrame(records).to_csv(OUT/'method_comparison.csv',index=False)
    write_json(OUT/'manifest.json',{'completed':True,'datasets':{n:m['selection'] for n,m in manifests.items()},
        'config':CONFIG,'config_scope':'initial eight-dimensional candidate comparison',
        'final_selection_policy':{'utility_guard':'every target validation Macro-F1 drop <= 0.02 versus its uncompressed probe',
            'dimension_refinement':'validation-driven PCA with 95%, 99%, then 99.9% training variance when needed',
            'widths':'use datasets[dataset].latent_dimensions for the final handoff'},
        'test_labels_used_for_selection':False,'input_stage':'verified fixed baseline',
        'runtime':{'python':sys.version,'torch':torch.__version__,'device':'cpu'},'completed_utc':datetime.now(timezone.utc).isoformat()})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=DATASETS)
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=4):
        if args.report_only:
            manifests={n:json.loads((OUT/n/'manifest.json').read_text()) for n in DATASETS}
        else:
            manifests={}
            for n in ([args.dataset] if args.dataset else DATASETS):
                dataset_run(n)
                from refine_utility import refine
                manifests[n]=refine(n)
            if args.dataset:
                return
        for n,m in manifests.items():
            baseline={'mean_validation_macro_f1':m['uncompressed_mean_macro_f1']}
            plot_dataset(n,baseline,{k:v['metrics'] for k,v in m['methods'].items()})
        diagnostics=cross_dataset_diagnostics()
        final_report(manifests,diagnostics)
    sys.path.insert(0,str(ROOT/'dataset_audit'))
    from publish_layout import main as publish_layout
    publish_layout()
    log('Latent stage completed; run validate_latent.py before using the handoff')


if __name__ == '__main__':
    main()
