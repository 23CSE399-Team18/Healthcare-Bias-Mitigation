"""Validate every plotted series and local link; reconcile saved bias-specific metrics."""
from pathlib import Path
from html.parser import HTMLParser
from datetime import datetime, timezone
import ast, csv, hashlib, json, re, urllib.parse
import numpy as np
import plotly.graph_objects as go
import nbformat

ROOT=Path(__file__).resolve().parents[1]
class Page(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.ids=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        self.links += [a[k] for k in ('src','href') if k in a]
        if 'id' in a:self.ids.append(a['id'])
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    previous=ROOT/'dataset_audit/active_recheck/dashboard_validation.json'
    inventory_file=ROOT/'dataset_audit/active_recheck/ALL_STAGE_FILES.csv'
    known_hashes={}
    if previous.exists() and inventory_file.exists() and json.loads(previous.read_text()).get('passed'):
        with inventory_file.open(encoding='utf-8',newline='') as f:known_hashes={r['path']:r['sha256'] for r in csv.DictReader(f)}
    reused=0
    manifest=json.loads((ROOT/'dataset_audit/bias_dashboard_manifest.json').read_text())
    assert manifest['completed'] and manifest['individual_pages']==144
    counts={s:0 for s in ('preprocessing','latent','inprocessing')};series=states=links=0
    for entry in manifest['entries']:
        p=ROOT/entry['html'];t=p.read_text(encoding='utf-8');parser=Page();parser.feed(t)
        schema_cached=known_hashes.get(entry['html'])==sha(p)
        reused+=int(schema_cached)
        payload=json.loads(re.search(r'<script id="payload" type="application/json">(.*?)</script>',t,re.S).group(1))
        saved=json.loads((ROOT/entry['data']).read_text());counts[entry['stage']]+=1
        assert payload['kind']==entry['bias'] and payload['stage']==entry['stage']
        assert payload['default_method'] in payload['states'][payload['default_split']]
        assert len(parser.ids)==len(set(parser.ids))
        for split,methods in payload['states'].items():
            for method,value in methods.items():
                states+=1;assert value['rows']>0
                for biasstate in ('before','after'):
                    s=value[biasstate]
                    if s['label'] not in ('Not applicable','Insufficient evidence'):
                        assert s['value'] is not None
                        assert s['label']==('Detected by rule' if s['value']>s['limit'] else 'Not detected by rule')
                for variable,v in value['variables'].items():
                    edges=np.array(v['edges']);assert np.all(np.diff(edges)>0)
                    for label in ('before','after'):
                        c=np.array(v[label+'_counts']);assert len(c)==len(edges)-1 and c.sum()==v[label+'_n']
                        if v[label+'_n']:
                            density=c/v[label+'_n']/np.diff(edges);np.testing.assert_allclose(np.sum(density*np.diff(edges)),1)
                            q=np.array(v[label+'_quantiles']);assert len(q)==101 and np.all(np.diff(q)>=0)
                            assert edges[0]<=q[0]<=q[-1]<=edges[-1]
                for panel in value['panels']:
                    if 'table' in panel:continue
                    layout=panel['layout'].copy()
                    for axis in ('xaxis','yaxis'):
                        if axis in layout:
                            layout[axis]=layout[axis].copy()
                            if isinstance(layout[axis].get('title'),str):layout[axis]['title']={'text':layout[axis]['title']}
                    if not schema_cached:go.Figure(data=panel['data'],layout=layout)
                    for trace in panel['data']:
                        if 'x' in trace and 'y' in trace:assert len(trace['x'])==len(trace['y']),entry
                        series+=1
                if entry['stage']=='inprocessing':
                    base=ROOT/'inprocessing/outputs'/entry['dataset']
                    decision=json.loads((base/'selection_frozen.json').read_text());assert sha(base/'selection_frozen.json')==payload['selection_sha256']
                    assert payload['default_method']==decision['selected']
                    for key,meth in [('before','ERM'),('after',method)]:
                        m=json.loads((base/meth/f'{split}_metrics.json').read_text())
                        assert value['rows']==m['rows']
                        bias=entry['bias']
                        if bias=='demographic':v=max((m['group_disparities'][k]['equalized_odds_gap'] for k in ('sens_sex','sens_age_group','sens_race_ethnicity') if m['group_disparities'][k]['equalized_odds_gap'] is not None),default=None)
                        elif bias=='intersectional':v=m['group_disparities']['intersection']['equalized_odds_gap']
                        elif bias=='miscalibration':v=m['ece_10_bins']
                        else:continue
                        assert value[key]['value']==v
        for target in parser.links:
            if re.match(r'^[a-zA-Z][\w+.-]*:',target) or target.startswith('#'):continue
            assert (p.parent/urllib.parse.unquote(target.split('#')[0])).exists(),(p,target)
            links+=1
        assert not re.search(r'GE[ -]?71|Pima|MICE reconstruction|SMOTE selected',t,re.I),p
    assert counts=={'preprocessing':64,'latent':32,'inprocessing':48}
    inventory=[];code=[];notebooks=0;jsons=0
    for stage in counts:
        for p in sorted((ROOT/stage).rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts:continue
            relative=p.relative_to(ROOT).as_posix()
            inventory.append({'path':relative,'bytes':p.stat().st_size,'sha256':sha(p),'role':'current code' if '/code/' in relative else 'current plot or dashboard asset' if '/plots/' in relative else 'current outputs and evidence' if '/outputs/' in relative else 'current navigation'})
            if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'));code.append(relative)
            elif p.suffix=='.json':json.loads(p.read_text(encoding='utf-8'));jsons+=1
            elif p.suffix=='.ipynb':
                nb=nbformat.read(p,as_version=4);nbformat.validate(nb);notebooks+=1
                for c in nb.cells:
                    if c.cell_type=='code':ast.parse(c.source);assert c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs)
    historical=[]
    for entry in json.loads((ROOT/'dataset_audit/cleanup_completed.json').read_text()):
        p=ROOT/entry['path']
        if p.exists():historical.append(str(p))
    assert not historical,historical
    old_nb=[ROOT/'preprocessing'/f'{n}_preprocessing.ipynb' for n in ('heart','respiratory','diabetes','hypertension')]
    assert not any(p.exists() for p in old_nb)
    migration=json.loads((ROOT/'dataset_audit/layout_migration.json').read_text())
    for path,digest in migration['frozen_selection_hashes'].items():assert sha(ROOT/path)==digest
    out=ROOT/'dataset_audit/active_recheck';out.mkdir(exist_ok=True)
    with (out/'ALL_STAGE_FILES.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(inventory[0]));w.writeheader();w.writerows(inventory)
    result={'passed':True,'checked_utc':datetime.now(timezone.utc).isoformat(),'pages_by_stage':counts,'individual_pages':144,'plot_states_checked':states,'chart_traces_schema_checked':series,'individual_page_links_checked':links,'all_stage_files_inventoried':len(inventory),'json_files_parsed':jsons,'python_files_parsed':code,'executed_notebooks_checked':notebooks,'known_superseded_files_found':historical,'frozen_selections_unchanged':True,'live_browser_visual_review':False,'browser_blocker':'Browser access host is unavailable; no browser visual verification claimed.'}
    result['previous_schema_checks_reused_for_hash_identical_pages']=reused
    (out/'dashboard_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='python_files_parsed'},indent=2),flush=True)
if __name__=='__main__':main()
