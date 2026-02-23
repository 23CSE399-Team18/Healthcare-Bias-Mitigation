"""Put verified-pipeline guidance before the historical notebook introductions."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
changes = {}
for name in ('heart', 'respiratory', 'diabetes', 'hypertension'):
    path = ROOT / f'preprocessing/{name}_preprocessing.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    code_before = json.dumps([c for c in notebook['cells'] if c['cell_type'] == 'code'], sort_keys=True)
    index = next(i for i, c in enumerate(notebook['cells'])
                 if c['cell_type'] == 'markdown' and '## Verification update' in ''.join(c['source']))
    notice = notebook['cells'].pop(index)
    notice['source'] = (
        '## Verification update — current files to use\n\n'
        '**The code and saved results below are historical experiments. They are not the '
        'preprocessing inputs used by the current latent pipeline.** Some historical methods '
        'were selected using test scores, so those scores are exploratory. The old dashboards '
        'also have limitations described in [the preprocessing audit](audit_current/REPORT.md).\n\n'
        f'For this dataset, use the train, validation and test CSVs in `verified_baseline/{name}/`. '
        'The fixed preprocessor is fitted on training rows only.\n\n'
        '- [Verified preprocessing plots](verified_baseline/index.html)\n'
        '- [Current latent notebook](../latent/latent_pipeline.ipynb)\n'
        '- [Latent results and file guide](../latent/README.md)\n'
        '- [Latest complete-folder recheck](../dataset_audit/recheck_current/REPORT.md)\n\n'
        'Baseline preprocessing and latent representations do not establish completion of '
        'the eight-bias mitigation experiment, improved fairness, or final test performance.\n'
    ).splitlines(keepends=True)
    notebook['cells'].insert(0, notice)
    for cell in notebook['cells'][1:]:
        if cell['cell_type'] != 'markdown':
            continue
        source = ''.join(cell['source'])
        source = source.replace(
            'This notebook is the complete pre-processing stage for this disease.',
            'This notebook preserves the historical preprocessing experiment for this disease.')
        source = source.replace(
            'the test split always stays a real,\nuntouched population.',
            'the test features remain a real population. Historical method selection nevertheless\n'
            'reused test labels, as explained in the verification notice above.')
        cell['source'] = source.splitlines(keepends=True)
    code_after = json.dumps([c for c in notebook['cells'] if c['cell_type'] == 'code'], sort_keys=True)
    assert code_before == code_after
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    changes[name] = {'code_and_saved_outputs_preserved': True,
                     'code_cells_sha256': hashlib.sha256(code_after.encode()).hexdigest(),
                     'verification_notice_is_first_cell': True}
out = ROOT / 'dataset_audit/recheck_current'
out.mkdir(parents=True, exist_ok=True)
(out / 'notebook_notice_changes.json').write_text(json.dumps(changes, indent=2), encoding='utf-8')
print('Moved verified-pipeline guidance to the top of all four historical notebooks; code and outputs preserved.')
