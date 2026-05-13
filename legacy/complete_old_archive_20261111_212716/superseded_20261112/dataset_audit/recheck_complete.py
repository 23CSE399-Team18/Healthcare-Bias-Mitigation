"""Supplement the saved-output validators with plot, metric and notebook checks.

Run the preprocessing and latent validators first. This script does not alter
datasets or fit latent encoders. It refits fixed validation probes solely to
verify the previously unsaved uncompressed reference scores.
"""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'latent/code'), str(ROOT / 'preprocessing/code')]
import joblib
import nbformat
import numpy as np
import pandas as pd
import sklearn
from PIL import Image
from IPython.core.inputtransformer2 import TransformerManager
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             normalized_mutual_info_score, adjusted_rand_score)
from threadpoolctl import threadpool_limits
from run_latent import DATASETS, METHODS, SEED, load_data, probe_metrics, anchor_groups

OUT = ROOT / 'dataset_audit/recheck_current'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def close(actual, expected, atol=1e-10):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=atol)


def gram_cka(a, b):
    """Independent sample-Gram formulation for matching demographic centroids."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a - a.mean(0), b - b.mean(0)
    ka, kb = a @ a.T, b @ b.T
    return np.sum(ka * kb) / np.sqrt(np.sum(ka * ka) * np.sum(kb * kb))


def check_mimic_plot():
    plot = read(ROOT / 'preprocessing/outputs/verified_baseline/mimic/plot_data.json')
    raw = pd.read_csv(ROOT / 'cross_dataset/four_conditions/mimic/heart_model_ready.csv')
    train = raw.loc[raw.meta_split.eq('train')]
    numeric = [f'{v}_{s}' for v in ('heart_rate', 'resp_rate', 'spo2') for s in ('mean', 'min', 'max')]
    base = ROOT / 'preprocessing/outputs/cross_dataset_baseline/mimic_to_eicu'
    for split in ('train', 'validation', 'test'):
        count = int(raw.meta_split.eq(split).sum())
        assert count == plot['partition_rows'][split]
        with np.load(base / f'X_{split}.npz') as archive:
            x = archive['X']
        assert x.shape == (count, plot['feature_count'])
        close(100 * (~np.isfinite(x)).mean(), plot['nonfinite_after_percent'][split])
        if split == 'train':
            for i, name in enumerate(numeric):
                close(x[:, i].std(), plot['numeric_train_std'][name])
    for name, value in plot['missing_train_percent'].items():
        close(100 * train[name].isna().mean(), value)
    name = plot['histogram_feature']
    for kind, values in [('observed', train[name].dropna()),
                         ('imputed', train[name].fillna(train[name].median()))]:
        counts = np.histogram(values, bins=plot['histogram_edges'])[0]
        np.testing.assert_array_equal(counts, plot[f'{kind}_counts'])
        assert int(counts.sum()) == len(values) == plot[f'{kind}_denominator']
    with np.load(base / 'y_train.npz') as labels:
        for task in labels.files:
            close(100 * labels[task].mean(), plot['target_prevalence_percent'][task])
    return {'passed': True, 'counts_missingness_scaling_histogram_and_four_label_rates_recomputed': True}


def check_latent_metrics(name, comparison):
    folder = ROOT / 'latent/outputs' / name
    m = read(folder / 'manifest.json')
    data = load_data(name)
    # These probes were not persisted by the training runner. Refit the same
    # fixed train-only specification, with no new selection or test scoring.
    baseline = probe_metrics(data, data['X']['train'], data['X']['validation'], None)
    for target, result in baseline.items():
        for field in ('macro_f1', 'auroc', 'average_precision', 'threshold'):
            close(result[field], m['uncompressed_probe'][target][field])
    close(np.mean([v['macro_f1'] for v in baseline.values()]), m['uncompressed_mean_macro_f1'])
    domains = {'heart': 'meta_domain_state', 'respiratory': 'meta_domain_hospitalid',
               'hypertension': 'meta_domain_region', 'mimic': 'meta_domain_hospitalid'}
    domain = domains.get(name)
    target_rows = 0
    for method, record in m['methods'].items():
        dest = folder / method
        metrics = record['metrics']
        assert read(dest / 'metrics.json') == metrics
        with np.load(dest / 'Z_validation.npz') as archive:
            z = archive['Z']
        if domain and data['rows']['train'][domain].nunique() > 1:
            fitted = joblib.load(dest / 'domain_clusters.joblib')
            assigned = fitted.predict(z)
            truth = data['rows']['validation'][domain].fillna('Unknown').astype(str)
            close(normalized_mutual_info_score(truth, assigned), metrics['domain_cluster_nmi'])
            close(adjusted_rand_score(truth, assigned), metrics['domain_cluster_ari'])
        else:
            assert metrics['domain_cluster_nmi'] is None and metrics['domain_cluster_ari'] is None
        scores = []
        for target, result in metrics['probe'].items():
            probe = joblib.load(dest / f'probe_{target}.joblib')
            p = probe['classifier'].predict_proba(probe['scaler'].transform(z))[:, 1]
            y = data['y']['validation'][target]
            close(roc_auc_score(y, p), result['auroc'])
            close(average_precision_score(y, p), result['average_precision'])
            assert len(y) == result['validation_rows']
            assert int(y.sum()) == result['validation_positive_rows']
            row = comparison.loc[(comparison.dataset == name) & (comparison.method == method)
                                 & (comparison.target == target)]
            assert len(row) == 1
            for metric in ('macro_f1', 'auroc', 'average_precision', 'silhouette'):
                close(row.iloc[0][metric], result[metric])
            assert int(row.iloc[0].latent_dimensions) == record['artifacts']['validation']['dimensions']
            assert bool(row.iloc[0].selected) == (method == m['selection']['method'])
            close(row.iloc[0].same_row_cka, metrics['same_row_input_latent_cka'])
            for column, metric in [('domain_nmi', 'domain_cluster_nmi'), ('domain_ari', 'domain_cluster_ari')]:
                assert pd.isna(row.iloc[0][column]) if metrics[metric] is None else np.isclose(row.iloc[0][column], metrics[metric])
            scores.append(result['macro_f1'])
            target_rows += 1
        close(np.mean(scores), metrics['mean_validation_macro_f1'])
        differences = [v['macro_f1'] - baseline[t]['macro_f1'] for t, v in metrics['probe'].items()]
        assert metrics['clinical_utility_guard_passed'] == (min(differences) >= -.02)
    return {'passed': True, 'candidates': len(m['methods']), 'task_metric_rows': target_rows,
            'uncompressed_reference_refit_matches': True, 'auroc_average_precision_domain_nmi_ari_match': True,
            'comparison_csv_matches': True, 'test_predictions_computed': False}


def check_cross_source():
    base = ROOT / 'latent/outputs'
    saved = read(base / 'cross_dataset_diagnostics.json')
    checked = {}
    for method in METHODS:
        vals, labels, centroids = [], [], {}
        for i, name in enumerate(DATASETS):
            folder = base / name
            with np.load(folder / method / 'Z_anchor_validation.npz') as archive:
                z = archive['Z']
            ix = np.random.default_rng(SEED).choice(len(z), min(1000, len(z)), replace=False)
            vals.append(z[ix]); labels.extend([i] * len(ix))
            rows = pd.read_csv(folder / 'rows_validation.csv', usecols=['sens_age_group', 'sens_sex'])
            groups = anchor_groups(rows)
            with np.load(folder / method / 'Z_validation.npz') as archive:
                z = archive['Z']
            centroids[name] = {key: z[groups == key].mean(0) for key in sorted(set(groups))
                               if (groups == key).sum() >= 20 and 'Unknown' not in key and '<NA>' not in key}
        record = saved[method]
        assigned = joblib.load(base / f'{method}_anchor_clusters.joblib').predict(np.concatenate(vals))
        close(normalized_mutual_info_score(labels, assigned), record['pooled_anchor_source_nmi'])
        close(adjusted_rand_score(labels, assigned), record['pooled_anchor_source_ari'])
        rng = np.random.default_rng(SEED)
        permutations = [normalized_mutual_info_score(rng.permutation(labels), assigned) for _ in range(100)]
        close(np.mean(permutations), record['nmi_permutation_mean'])
        close(np.quantile(permutations, .95), record['nmi_permutation_95pct'])
        assert record['validation_rows'] == len(labels) == 5000
        assert record['dataset_order'] == list(DATASETS)
        for i, a in enumerate(DATASETS):
            for j, b in enumerate(DATASETS):
                keys = sorted(set(centroids[a]) & set(centroids[b]))
                assert len(keys) == record['paired_group_counts'][i][j]
                if len(keys) >= 4:
                    actual = gram_cka([centroids[a][key] for key in keys], [centroids[b][key] for key in keys])
                    close(actual, record['group_centroid_cka'][i][j], atol=1e-6)
                else:
                    assert record['group_centroid_cka'][i][j] is None
        checked[method] = {'passed': True, 'group_cka_independently_recomputed': True,
                           'pooled_nmi_ari_and_permutation_reference_match': True}
    return checked


def check_notebooks_and_images():
    records = {}
    transformer = TransformerManager()
    notebooks = sorted((ROOT / 'preprocessing/code').glob('*.ipynb')) + [ROOT / 'latent/code/latent_pipeline.ipynb']
    for path in notebooks:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                ast.parse(transformer.transform_cell(cell.source), filename=f'{path.name}:cell{i}')
                assert not any(o.output_type == 'error' for o in cell.get('outputs', []))
        records[str(path.relative_to(ROOT))] = {'syntax_and_saved_errors_passed': True, 'sha256': digest(path),
            'historical_training_reexecuted': False}
    images = [ROOT / f'preprocessing/plots/baseline/{n}/preprocessing_checks.png' for n in DATASETS]
    images += [ROOT / f'latent/plots/{n}/comparison.png' for n in DATASETS]
    images.append(ROOT / 'latent/plots/group_alignment.png')
    image_records = {}
    for path in images:
        with Image.open(path) as picture:
            size = picture.size
            picture.verify()
        image_records[str(path.relative_to(ROOT))] = {'size': list(size), 'sha256': digest(path)}
    links_checked = 0
    for path in (ROOT / 'preprocessing/plots/index.html', ROOT / 'latent/index.html'):
        for link in re.findall(r'(?:src|href)="([^"]+)"', path.read_text(encoding='utf-8')):
            assert (path.parent / link).exists(), link
            links_checked += 1
    return {'notebooks': records, 'images': image_records, 'html_links_checked': links_checked}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {'passed': False, 'checked_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'supplemental review of current and historical artifacts',
              'dataset_files_modified': False, 'latent_encoders_retrained': False,
              'test_predictions_computed': False, 'sklearn_runtime': sklearn.__version__}
    try:
        for path in ('preprocessing/outputs/audit_current/confirmation.json', 'latent/outputs/validation.json',
                     'preprocessing/outputs/audit_current/mimic_audit.json'):
            assert read(ROOT / path)['passed'], path
        result['prerequisite_reports'] = {path: digest(ROOT / path) for path in (
            'preprocessing/outputs/audit_current/confirmation.json', 'latent/outputs/validation.json',
            'preprocessing/outputs/audit_current/mimic_audit.json')}
        result['historical'] = 'Moved outside the current project; see dataset_audit/archive_manifest.json'
        result['mimic_plot'] = check_mimic_plot()
        result['latent_metrics'] = {}
        comparison = pd.read_csv(ROOT / 'latent/outputs/method_comparison.csv')
        with threadpool_limits(limits=4):
            for name in DATASETS:
                print('Checking all reported validation metrics: ' + name, flush=True)
                result['latent_metrics'][name] = check_latent_metrics(name, comparison)
            assert len(comparison) == sum(v['task_metric_rows'] for v in result['latent_metrics'].values())
            result['cross_source_diagnostics'] = check_cross_source()
        result['notebooks_and_plots'] = check_notebooks_and_images()
        result['passed'] = True
    except BaseException as error:
        result['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        (OUT / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('PASS: ' + str(OUT / 'verification.json'), flush=True)


if __name__ == '__main__':
    main()
