"""Remove dependencies on archived experiments and refresh current entry points."""
from pathlib import Path
import ast
import json
import nbformat

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'preprocessing/audit_and_prepare.py'
code = path.read_text(encoding='utf-8')
start, end = code.index('def audit_original('), code.index('def feature_frame(')
code = code[:start] + code[end:]
start, end = code.index('def write_report('), code.index('def refresh_plots(')
replacement = '''def write_report(state):
    lines = ["# Verified preprocessing baseline", "",
        "Current baseline preprocessing is complete for heart, respiratory, diabetes, hypertension and MIMIC.", "",
        "The four application baselines use training-only imputation, scaling and encoding. MIMIC uses its independently checked source-training preprocessor.", "",
        "| Dataset | Train / validation / test | Features |", "|---|---|---:|"]
    for name in DATASETS:
        record = state['baseline'][name]
        counts = ' / '.join(f"{record['partitions'][s]['rows']:,}" for s in ('train','validation','test'))
        lines.append(f"| {name} | {counts} | {record['partitions']['train']['features']} |")
    lines += ["| mimic | 52,229 / 10,829 / 10,932 | 24 |", "",
        "[Open current plots](../verified_baseline/index.html) and [the file guide](../verified_baseline/README.md).", "",
        "Saved outputs are checked for source identity, label alignment, finite values and disjoint records/patients. Original survey weights remain metadata. These checks do not establish better fairness or clinical performance.", "",
        "The respiratory mortality target differs from MIMIC's recorded respiratory diagnosis. Keep the tasks separate.", "",
        "Historical experiments have been moved outside D:/complete. See [the archive manifest](../../dataset_audit/archive_manifest.json). The eight-bias mitigation experiment and final in-processing evaluation are not complete.", "",
        "Run `python preprocessing/confirm_saved_outputs.py` to verify saved files without rebuilding. Run `python preprocessing/audit_and_prepare.py` only to rebuild the fixed baselines; refresh dependent latent artifacts after any data change."]
    (AUDIT / 'REPORT.md').write_text('\\n'.join(lines)+'\\n', encoding='utf-8')


'''
code = code[:start] + replacement + code[end:]
code = code.replace('"original": {}, "dashboards": {}, "baseline": {}', '"baseline": {}')
code = code.replace('        state["original"][name] = audit_original(name, raw)\n', '')
code = code.replace('        state["dashboards"][name] = audit_legacy_dashboards(name)\n', '')
ast.parse(code)
path.write_text(code, encoding='utf-8')

(ROOT / 'preprocessing/README.md').write_text('''# Current verified preprocessing

Open [verified_preprocessing.ipynb](verified_preprocessing.ipynb) to inspect the saved baselines and plots.

- [Verified datasets and file guide](verified_baseline/README.md)
- [Current preprocessing plots](verified_baseline/index.html)
- [Validation evidence](audit_current/confirmation.json)
- [Latent notebook](../latent/latent_pipeline.ipynb)

The old disease notebooks, experimental outputs and unfinished MIMIC notebook work have been moved outside this project. Their locations and hashes are recorded in [the archive manifest](../dataset_audit/archive_manifest.json).

To verify existing files: `python preprocessing/confirm_saved_outputs.py`.
To rebuild the fixed baselines: `python preprocessing/audit_and_prepare.py`.
Rebuilding changes upstream artifacts; revalidate and rebuild dependent latent outputs when necessary.
''', encoding='utf-8')
guide = ROOT / 'preprocessing/verified_baseline/README.md'
text = guide.read_text(encoding='utf-8').replace(
    'Historical\nnotebook outputs remain under `preprocessing/outputs/`; their test scores were\nused during method selection and must be interpreted as exploratory.',
    'Historical notebooks and experimental outputs were moved outside this project.\nSee [the archive manifest](../../dataset_audit/archive_manifest.json) for their locations.\nTheir test scores were used during selection and remain exploratory.')
guide.write_text(text, encoding='utf-8')
report = ROOT / 'dataset_audit/recheck_current/REPORT.md'
text = report.read_text(encoding='utf-8').replace('../../preprocessing/audit_20261109/mimic_audit.json', '../../preprocessing/audit_current/mimic_audit.json')
text = '> Layout update: the historical notebooks, experimental outputs and superseded study files discussed below have now been archived outside D:/complete. See [archive_manifest.json](../archive_manifest.json). This report records the review performed before that move; current validators no longer require the archived files.\n\n' + text
report.write_text(text, encoding='utf-8')
for rel in ('cross_dataset/prepare_cross_dataset.py',):
    path = ROOT / rel
    text = path.read_text(encoding='utf-8').replace('Run evaluate_generalization.py on both prepared real cohorts.', 'Use the current four-condition workflow in four_conditions/README.md; the historical mortality runner is archived.')
    path.write_text(text, encoding='utf-8')

nb = nbformat.v4.new_notebook()
nb.cells = [nbformat.v4.new_markdown_cell('''# Verified preprocessing: five datasets

This notebook reads the current saved baselines and verification evidence. Run all cells to inspect them. It does not rebuild datasets or train models.

The next stage is [the latent notebook](../latent/latent_pipeline.ipynb). Historical experiments are archived outside this project; see [the archive manifest](../dataset_audit/archive_manifest.json).
'''), nbformat.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image
ROOT = next(p for p in (Path.cwd(), Path.cwd().parent) if (p / 'preprocessing/verified_baseline').exists())
confirmation = json.loads((ROOT / 'preprocessing/audit_current/confirmation.json').read_text())
assert confirmation['passed']
records = []
for name, result in confirmation['datasets'].items():
    for split, part in result['partitions'].items():
        records.append({'dataset': name, 'split': split, 'rows': part['rows'], 'features': part['features'], 'verified': result['passed']})
mimic = json.loads((ROOT / 'preprocessing/audit_current/mimic_audit.json').read_text())
assert mimic['passed']
for split in ('train', 'validation', 'test'):
    part = mimic['directions']['mimic_to_eicu']['partitions'][split]
    records.append({'dataset': 'mimic', 'split': split, 'rows': part['rows'], 'features': part['features'], 'verified': True})
display(pd.DataFrame(records))
'''), nbformat.v4.new_code_cell('''for name in ('heart', 'respiratory', 'diabetes', 'hypertension', 'mimic'):
    display(Image(filename=str(ROOT / f'preprocessing/verified_baseline/{name}/preprocessing_checks.png')))
'''), nbformat.v4.new_markdown_cell('''Use the [baseline file guide](verified_baseline/README.md) for the three split files per dataset. MIMIC retains four diagnosis labels. Preserve the split assignments and keep metadata and labels out of predictor columns.

To rerun the read-only checks, run `python preprocessing/confirm_saved_outputs.py` from the project root. The historical eight-bias experiments and final fairness evaluation are not implied by these baseline checks.
''')]
nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nbformat.validate(nb)
nbformat.write(nb, ROOT / 'preprocessing/verified_preprocessing.ipynb')
print('Updated current pipeline paths and created verified preprocessing notebook.')
