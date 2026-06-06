"""Final lightweight checks after model/data validation; no refitting."""
from pathlib import Path
import contextlib
import io
import json
import re
import nbformat
import numpy as np
from load_for_inprocessing import load_latent

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'latent'
validation=json.loads((BASE/'outputs/validation.json').read_text())
assert validation['passed']
summary={}
csv_count=0
for name in ('heart','respiratory','diabetes','hypertension','mimic'):
    m=json.loads((BASE/'outputs'/name/'manifest.json').read_text())
    for method,r in m['methods'].items():
        differences={t:p['macro_f1']-m['uncompressed_probe'][t]['macro_f1'] for t,p in r['metrics']['probe'].items()}
        assert r['metrics']['clinical_utility_guard_passed']==(min(differences.values())>=-.02)
    selected=m['selection']['method']
    assert m['methods'][selected]['metrics']['clinical_utility_guard_passed']
    for split in ('train','validation','test'):
        csv_count+=1
        for task in (m['targets'] if name=='mimic' else [name]):
            x,y,meta=load_latent(name,split,task)
            assert len(x)==m['metadata'][split]['rows']
            assert x.shape[1]==m['selection']['latent_dimensions']
    summary[name]={'method':selected,'dimensions':m['selection']['latent_dimensions'],
                   'all_target_guards_passed':True}
try:
    load_latent('mimic')
    raise AssertionError('MIMIC accepted an unspecified target')
except ValueError:
    pass
html=(BASE/'index.html').read_text(encoding='utf-8')
links=re.findall(r'(?:src|href)="([^"]+)"',html)
for link in links:assert (BASE/link).exists(),link
nb=nbformat.read(BASE/'code/latent_pipeline.ipynb',as_version=4)
nbformat.validate(nb)
scope={}
with contextlib.redirect_stdout(io.StringIO()) as captured:
    for cell in nb.cells:
        if cell.cell_type=='code':
            exec(compile(cell.source,'latent_pipeline.ipynb','exec'),scope)
result={'passed':True,'model_data_validation_passed':True,'complete_csv_count':csv_count,
        'selected':summary,'safe_loader_all_splits_and_targets_passed':True,
        'notebook_code_cells_executed_successfully_in_python':True,
        'notebook_training_flag_during_check':False,'local_report_links_checked':len(links),
        'test_labels_scored':False}
(BASE/'outputs/handoff_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
