"""Final static integration checks for the organized saved project."""
from pathlib import Path
import ast,hashlib,json,re,urllib.parse
from html.parser import HTMLParser
import nbformat
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

class Page(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.tabs=[];self.panels=[];self.selected=0
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        for key in ('href','src'):
            if key in a:self.links.append(a[key])
        if 'data-tab' in a:self.tabs.append(a['data-tab']);self.selected+=a.get('aria-selected')=='true'
        if 'data-panel' in a:self.panels.append((a['data-panel'],'hidden' in a))

def main():
    errors=[];links=0;pages=0;figures=0;code=0;notebooks=0
    migration=json.loads((ROOT/'dataset_audit/layout_migration.json').read_text())
    for p,expected in migration['frozen_selection_hashes'].items():assert sha(ROOT/p)==expected
    preserved=0
    for item in migration['file_moves']:
        p=ROOT/item['new']
        if p.suffix in ('.csv','.npz','.joblib','.pt','.png'):
            assert sha(p)==item['sha256_before'],str(p);preserved+=1
    for stage in ('preprocessing','latent','inprocessing'):
        base=ROOT/stage
        assert all((base/k).is_dir() for k in ('code','outputs','plots'))
        assert not list(base.glob('*.py')) and not list(base.glob('*.ipynb'))
        assert not list((base/'outputs').rglob('*.png')) and not list((base/'outputs').rglob('*.html'))
        for p in (base/'code').glob('*.py'):ast.parse(p.read_text(encoding='utf-8'));code+=1
        for p in (base/'code').glob('*.ipynb'):
            nb=nbformat.read(p,as_version=4);nbformat.validate(nb)
            for cell in nb.cells:
                if cell.cell_type=='code':
                    ast.parse(cell.source);assert cell.execution_count is not None,p
                    assert all(o.output_type!='error' for o in cell.outputs)
            notebooks+=1
        for p in (base/'plots').rglob('*.png'):
            with Image.open(p) as im:im.verify()
            figures+=1
    docs=[ROOT/'START_HERE.html',ROOT/'README.md',ROOT/'DATASETS.md',ROOT/'BIAS_DASHBOARDS.md',ROOT/'dataset_audit/CODE_REVIEW.md',ROOT/'dataset_audit/FINAL_REVIEW.md']
    docs += [p for stage in ('preprocessing','latent','inprocessing') for p in (ROOT/stage).rglob('*') if p.suffix in ('.html','.md')]
    for p in docs:
        t=p.read_text(encoding='utf-8')
        if p.suffix=='.html':
            parser=Page();parser.feed(t);targets=parser.links;pages+=1
            if parser.tabs:
                assert len(set(parser.tabs))==len(parser.tabs) and set(parser.tabs)=={s for s,h in parser.panels},p
                assert parser.selected==1 and sum(not h for s,h in parser.panels)==1,p
        else:targets=re.findall(r'\]\(([^)]+)\)',t)
        for target in targets:
            if re.match(r'^[a-zA-Z][\w+.-]*:',target) or target.startswith('#'):continue
            target=urllib.parse.unquote(target.split('#')[0])
            dest=(p.parent/target).resolve()
            if not dest.exists():errors.append({'file':p.relative_to(ROOT).as_posix(),'target':target})
            links+=1
    reports=['preprocessing/outputs/audit_current/confirmation.json','preprocessing/outputs/audit_current/mimic_audit.json','latent/outputs/validation.json','latent/outputs/hospital/independent_validation.json','inprocessing/outputs/validation.json','inprocessing/outputs/export_validation.json','dataset_audit/notebook_execution_all.json']
    for report in reports:assert json.loads((ROOT/report).read_text())['passed'],report
    result={'passed':not errors,'moved_data_model_prediction_plot_files_hash_unchanged':preserved,'all_frozen_selection_records_unchanged':True,'python_files_parsed':code,'executed_notebooks':notebooks,'plot_images_valid':figures,'html_pages_checked':pages,'local_links_checked':links,'broken_links':errors,'passed_pipeline_reports':reports,'browser_visual_review':'Not completed: Computer Use stopped because the current browser URL could not be verified. Local HTML structure, tab wiring and links were checked.'}
    (ROOT/'dataset_audit/layout_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
