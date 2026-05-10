"""Publish individual dashboards, dataset inventory, and read-only notebooks.

Reads measured artifacts only. Does not train, select, or modify predictions.
"""
from pathlib import Path
import json,html,os,csv
import nbformat

ROOT=Path(__file__).resolve().parents[1]
STAGES=('preprocessing','latent','inprocessing')
NAMES=('heart','respiratory','diabetes','hypertension','mimic')
TITLES={'heart':'Heart | BRFSS 2022','diabetes':'Diabetes | NHANES 2021-2023','hypertension':'Hypertension | NHIS 2023','respiratory':'Respiratory mortality | patient-split reference','mimic':'MIMIC-IV | source cohort','respiratory_hospital':'Respiratory mortality | eICU hospitals','eicu_hospital':'Four diagnoses | eICU hospitals','mimic_transfer':'Four diagnoses | MIMIC to eICU'}
SCOPE={'heart':'Survey record holdout; not hospitals','diabetes':'Survey record holdout; no domain ID','hypertension':'Survey record holdout; regions are not hospitals','respiratory_hospital':'137 / 34 / 29 disjoint train / validation / test hospitals','eicu_hospital':'141 / 34 / 31 disjoint hospitals; external MIMIC','mimic_transfer':'One source institution; patient holdout and external eICU'}
CSS='''*{box-sizing:border-box}body{margin:0;background:#f4f7fa;color:#183347;font:16px/1.55 system-ui}main{max-width:1450px;margin:auto;padding:32px 28px 70px}nav{background:#16354a;padding:16px 28px;display:flex;gap:24px;flex-wrap:wrap}nav a{color:#e9f4fa;text-decoration:none}a{color:#116981}h1{font-size:clamp(26px,3vw,38px);line-height:1.2;margin:12px 0}h2{font-size:23px;margin:0 0 14px}.eyebrow{font-size:13px;letter-spacing:1.6px;text-transform:uppercase;color:#37728a}.lead{color:#506878;max-width:1050px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px;margin:24px 0}.card,section{background:white;border:1px solid #d8e3eb;border-radius:12px;padding:24px;margin-bottom:24px}.card{margin:0;display:block;text-decoration:none;color:inherit}.card:hover{border-color:#258c83;box-shadow:0 6px 20px #17354a12}.card p{color:#506878}.tag{display:inline-block;background:#e6f2ef;color:#17634e;padding:4px 10px;border-radius:7px;font-size:13px}.metric{font-size:26px;font-weight:650}.muted{color:#647887;font-size:14px}.figure{width:100%;height:auto;display:block}.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:12px;border-bottom:1px solid #dfe7ec;vertical-align:top}th{background:#f0f5f8}.tabs{display:flex;gap:10px;flex-wrap:wrap;margin:25px 0}.tabs button{border:1px solid #b6ccd8;background:white;border-radius:8px;padding:11px 20px;color:#183347;cursor:pointer;font:inherit}.tabs button[aria-selected=true]{background:#166d65;color:white;border-color:#166d65}[hidden]{display:none!important}.actions{display:flex;gap:18px;flex-wrap:wrap;margin:18px 0}.note{border-left:4px solid #d6a34e;background:#fff9ef;padding:16px 20px;margin:24px 0}.small{font-size:13px}code{background:#eef2f5;padding:2px 5px}.footer{font-size:13px;color:#617786;margin-top:35px}'''
JS="""document.querySelectorAll('[data-tab]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('[data-tab]').forEach(x=>x.setAttribute('aria-selected',String(x===b)));document.querySelectorAll('[data-panel]').forEach(p=>p.hidden=p.dataset.panel!==b.dataset.tab)}));"""

def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def esc(s):return html.escape(str(s))
def rel(page,target):return os.path.relpath(ROOT/target,page.parent).replace('\\','/')
def link(page,target,label):return f'<a href="{esc(rel(page,target))}">{esc(label)}</a>'
def fig(page,target,alt):return f'<img class="figure" loading="lazy" src="{esc(rel(page,target))}" alt="{esc(alt)}">'
def page(path,title,body,subtitle=''):
    path=ROOT/path;path.parent.mkdir(parents=True,exist_ok=True)
    nav=''.join(link(path,p,t) for p,t in [('START_HERE.html','Project'),('preprocessing/plots/index.html','Preprocessing'),('latent/plots/index.html','Latent'),('inprocessing/plots/index.html','In-processing'),('DATASETS.md','Dataset catalogue')])
    path.write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+esc(title)+'</title><style>'+CSS+'</style></head><body><nav>'+nav+'</nav><main><div class="eyebrow">Preprocessing / Latent / In-processing</div><h1>'+esc(title)+'</h1><p class="lead">'+esc(subtitle)+'</p>'+body+'<p class="footer">Saved research results. Selection uses validation data only. Rates are unweighted; read support counts and method adaptations with the results.</p></main><script>'+JS+'</script></body></html>',encoding='utf-8')

def cards(current,items):
    return '<div class="grid">'+''.join(f'<a class="card" href="{esc(rel(current,p))}"><h2>{esc(title)}</h2><p>{esc(desc)}</p><span class="tag">Open dashboard</span></a>' for p,title,desc in items)+'</div>'

def csv_table(current,files):
    return '<div class="tablewrap"><table><thead><tr><th>Partition</th><th>Complete file</th></tr></thead><tbody>'+''.join('<tr><td>'+esc(split)+'</td><td>'+link(current,p,Path(p).name)+'</td></tr>' for split,p in files)+'</tbody></table></div>'

def input_dashboards():
    preitems=[];latitems=[]
    for name in NAMES:
        prepage=ROOT/f'preprocessing/plots/baseline/{name}/index.html'
        files=[(s,f'preprocessing/outputs/verified_baseline/{name}/{name}_{s}_preprocessed.csv') for s in ('train','validation','test')]
        extra='Patient-split reference; use the separate hospital branch for hospital evaluation.' if name=='respiratory' else 'Training-only fitted preprocessing; saved split CSVs retain labels and metadata separately from predictors.'
        body='<section>'+fig(prepage,f'preprocessing/plots/baseline/{name}/preprocessing_checks.png','Verified preprocessing diagnostics')+'</section><section><h2>Complete preprocessing outputs</h2>'+csv_table(prepage,files)+'</section><div class="actions">'+link(prepage,f'preprocessing/code/{name}_verified_preprocessing.ipynb','Notebook with saved outputs')+link(prepage,'preprocessing/code/README.md','Preprocessing code')+'</div>'
        page(prepage,TITLES[name]+' - preprocessing',body,extra)
        preitems.append((str(prepage.relative_to(ROOT)),TITLES[name],extra))
        m=read(f'latent/outputs/{name}/manifest.json');selection=m['selection']
        lp=ROOT/f'latent/plots/{name}/index.html'
        files=[(s,str(Path(selection[s]['csv'])).replace('\\','/')) for s in ('train','validation','test')]
        body=f'<p class="tag">Selected: {esc(selection["method"])} | {selection["latent_dimensions"]} latent features</p><section>'+fig(lp,f'latent/plots/{name}/comparison.png','Latent candidate comparison')+'</section><section><h2>Complete latent outputs</h2>'+csv_table(lp,files)+'</section><div class="actions">'+link(lp,f'latent/outputs/{name}/manifest.json','Measured metrics and provenance')+link(lp,'latent/code/latent_pipeline.ipynb','Latent notebook')+'</div>'
        page(lp,TITLES[name]+' - latent',body,'Validation-only method selection. These representations remain dataset-specific; separate latent coordinates must not be pooled.')
        latitems.append((str(lp.relative_to(ROOT)),TITLES[name],f'{selection["method"]}; {selection["latent_dimensions"]} latent features'))
    exports=read('preprocessing/outputs/hospital_baseline/export_validation.json')['exports']
    for branch,folder in [('respiratory_hospital','respiratory'),('eicu_hospital','eicu'),('mimic_transfer','mimic')]:
        m=read(f'latent/outputs/hospital/{branch}/manifest.json')
        pp=ROOT/f'preprocessing/plots/hospital/{folder}/index.html'
        files=[(s,v['csv'].replace('\\','/')) for s,v in exports[branch].items()]
        body='<section>'+fig(pp,f'preprocessing/plots/hospital/{folder}/preprocessing_checks.png','Hospital preprocessing checks')+'</section><section><h2>Complete source-transformed outputs</h2>'+csv_table(pp,files)+'</section><div class="actions">'+link(pp,'preprocessing/code/hospital_preprocessing_latent.ipynb','Hospital notebook')+link(pp,'latent/outputs/hospital/independent_validation.json','Source transformation checks')+'</div>'
        page(pp,TITLES[branch]+' - preprocessing',body,SCOPE[branch]);preitems.append((str(pp.relative_to(ROOT)),TITLES[branch],SCOPE[branch]))
        lp=ROOT/f'latent/plots/hospital/{branch}/index.html'
        files=[(s,f'latent/outputs/hospital/{branch}/{branch}_{s}_latent.csv') for s in m['partitions']]
        body='<section>'+fig(lp,f'latent/plots/hospital/{branch}/latent_checks.png','Source-fitted PCA diagnostics')+'</section><section><h2>Complete hospital latent outputs</h2>'+csv_table(lp,files)+'</section><div class="actions">'+link(lp,'latent/outputs/hospital/independent_validation.json','Independent checks')+link(lp,f'inprocessing/plots/{branch}/index.html','In-processing dashboard')+'</div>'
        page(lp,TITLES[branch]+' - latent',body,SCOPE[branch]+'; source-training PCA and latent scaling are frozen.');latitems.append((str(lp.relative_to(ROOT)),TITLES[branch],f'{m["partitions"]["train"]["dimensions"]} latent features; '+SCOPE[branch]))
    for stage,items in [('preprocessing',preitems),('latent',latitems)]:
        p=ROOT/stage/'plots/index.html';body=cards(p,items)
        body+='<p class="note">The original respiratory patient-split results are a reference. Cross-hospital in-processing uses the hospital branch. MIMIC has one source institution; survey data have no hospital IDs.</p>'
        if stage=='latent':body+='<details><summary>Aggregate demographic alignment diagnostic</summary>'+fig(p,'latent/plots/group_alignment.png','Aggregate group alignment')+'</details>'
        page(p,stage.title()+' dashboards',body,'Choose an individual dataset or hospital branch.')
    hp=ROOT/'latent/plots/hospital/index.html';page(hp,'Hospital latent dashboards',cards(hp,latitems[-3:]))

def inprocessing_dashboards():
    manifest=read('inprocessing/outputs/manifest.json');hubitems=[]
    for dataset,parts in manifest['selected_exports'].items():
        base=ROOT/'inprocessing/outputs'/dataset
        tasks=sorted(p.name for p in base.iterdir() if p.is_dir() and (p/'COMPLETE.json').exists())
        branchpage=ROOT/f'inprocessing/plots/{dataset}/index.html';taskitems=[]
        for task in tasks:
            folder=f'inprocessing/outputs/{dataset}/{task}';decision=read(folder+'/selection_frozen.json');complete=read(folder+'/COMPLETE.json');selected=decision['selected']
            target=ROOT/f'inprocessing/plots/{dataset}/{task}/index.html' if len(tasks)>1 else branchpage
            split_names=[s for s in ('test','external','validation') if s in complete['splits']]
            tabs='<div class="tabs" role="tablist" aria-label="Evaluation partition">'+''.join(f'<button role="tab" data-tab="{s}" aria-selected="{str(i==0).lower()}">{s.title()}</button>' for i,s in enumerate(split_names))+'</div>'
            body=f'<p class="tag">Validation-selected: {esc(selected)}</p><p>{esc(SCOPE[dataset])}</p>'+tabs
            for i,split in enumerate(split_names):
                measured=read(folder+'/'+selected+'/'+split+'_metrics.json');erm=read(folder+'/ERM/'+split+'_metrics.json')
                if split=='validation':plot=f'inprocessing/plots/{dataset}/{task}/method_comparison.png'
                else:plot=f'inprocessing/plots/{dataset}/{task}/{split}_comparison.png'
                panel=f'<div data-panel="{split}"'+(' hidden' if i else '')+'>'
                panel+='<div class="grid">'
                for key,label in [('macro_f1','Macro-F1'),('auroc','AUROC'),('max_equalized_odds_gap','Largest supported EO gap')]:
                    a,b=erm[key],measured[key];display=lambda v:'NA' if v is None else f'{v:.4f}'
                    panel+=f'<div class="card"><span class="muted">{label}</span><div class="metric">{display(b)}</div><span class="muted">ERM: {display(a)} | selected: {display(b)}</span></div>'
                panel+='</div><section>'+fig(target,plot,f'{dataset} {task} {split} measured comparison')+'</section><div class="actions">'+link(target,folder+'/'+selected+'/'+split+'_metrics.json','Exact metrics')+link(target,folder+'/'+selected+'/'+split+'_groups.csv','Group counts and support')+link(target,parts[split]['csv'],'Complete prediction CSV')+'</div></div>'
                body+=panel
            body+='<p class="note">A smaller fairness gap can accompany lower utility. Selection was frozen before test/external scoring. Unsupported comparisons are NA. These are single-seed experiments and documented method adaptations.</p><div class="actions">'+link(target,'inprocessing/code/PLAN.md','Method definitions')+link(target,'inprocessing/code/run_inprocessing.py','Training code')+link(target,'inprocessing/outputs/validation.json','Model and metric checks')+'</div>'
            page(target,TITLES[dataset]+' - '+task.title(),body,'Separate evaluation partitions. Use the buttons to inspect test, external (when available), and validation results.')
            taskitems.append((str(target.relative_to(ROOT)),task.title(),f'Selected: {selected}'))
        if len(tasks)>1:page(branchpage,TITLES[dataset],cards(branchpage,taskitems)+'<section><h2>Complete multi-target prediction files</h2>'+csv_table(branchpage,[(s,v['csv']) for s,v in parts.items()])+'</section>',SCOPE[dataset]+'. Each disease has its own dashboard; the labels overlap.')
        hubitems.append((str(branchpage.relative_to(ROOT)),TITLES[dataset],SCOPE[dataset]))
    hp=ROOT/'inprocessing/plots/index.html'
    page(hp,'In-processing dashboards',cards(hp,hubitems)+'<p class="note">Three surveys, one respiratory mortality task, and two sets of four ICU diagnosis tasks: 12 evaluated tasks across six analysis branches.</p>','Choose one dataset. MIMIC and eICU each provide four separate disease dashboards.')

def catalogue():
    sources=[
      {'application':'Heart','source':'CDC BRFSS 2022','rows':440111,'target':'Self-reported heart attack / coronary heart disease / angina','model_ready':'model_ready/heart_model_ready.csv','raw':'Heart_correct/LLCP2022.XPT','official':'https://www.cdc.gov/brfss/annual_data/annual_2022.html'},
      {'application':'Respiratory','source':'eICU-CRD v2.0','rows':28842,'target':'Subsequent in-hospital mortality after the 24-hour landmark in the respiratory cohort','model_ready':'model_ready/respiratory_model_ready.csv','raw':'resp_physionet/resp/official_verified','official':'https://physionet.org/content/eicu-crd/2.0/'},
      {'application':'Diabetes','source':'CDC NHANES August 2021-August 2023','rows':7876,'target':'Self-reported diagnosed diabetes','model_ready':'model_ready/diabetes_model_ready.csv','raw':'diabetes_correct/raw','official':'https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Cycle=2021-2023'},
      {'application':'Hypertension','source':'CDC NHIS 2023','rows':29471,'target':'Self-reported diagnosed hypertension','model_ready':'model_ready/hypertension_model_ready.csv','raw':'hypertension_correct/nhis2023','official':'https://www.cdc.gov/nchs/nhis/documentation/2023-nhis.html'},
      {'application':'MIMIC','source':'MIMIC-IV v3.1','rows':73990,'target':'Four overlapping recorded diagnoses: ischemic heart disease, respiratory-system disease, diabetes, systemic hypertension','model_ready':'cross_dataset/four_conditions/mimic','raw':'cross_dataset/raw','official':'https://physionet.org/content/mimiciv/3.1/'}]
    for item in sources:assert (ROOT/item['model_ready']).exists() and (ROOT/item['raw']).exists()
    items=[]
    for stage,pattern in [('preprocessing','*_preprocessed.csv'),('latent','*_latent.csv'),('inprocessing','*_predictions.csv')]:
        for p in sorted((ROOT/stage/'outputs').rglob(pattern)):
            items.append({'stage':stage,'file':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size})
    with (ROOT/'DATASET_FILES.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['stage','file','bytes']);writer.writeheader();writer.writerows(items)
    branch=[]
    for name in ('heart','diabetes','hypertension'):
        p=read(f'preprocessing/outputs/verified_baseline/{name}/validation.json');m=read(f'latent/outputs/{name}/manifest.json')
        branch.append({'branch':name,'tasks':[name],'preprocessed_features':len(p['output_predictors']),'latent_features':m['selection']['latent_dimensions'],'partitions':{s:{'rows':v['rows']} for s,v in p['partitions'].items()},'evaluation':SCOPE[name]})
    for name in ('respiratory_hospital','eicu_hospital','mimic_transfer'):
        m=read(f'latent/outputs/hospital/{name}/manifest.json');f=read(m['source'].replace('\\','/')+'/feature_names.json')
        branch.append({'branch':name,'tasks':m['targets'],'preprocessed_features':len(f.get('features',f.get('output_features'))),'latent_features':m['partitions']['train']['dimensions'],'partitions':{s:{'rows':p['rows'],'hospitals':p['hospitals']} for s,p in m['partitions'].items()},'evaluation':SCOPE[name]})
    (ROOT/'DATASETS.json').write_text(json.dumps({'sources':sources,'analysis_branches':branch,'complete_csv_inventory':items},indent=2),encoding='utf-8')
    lines=['# Datasets used in this project','','There are **five source datasets** and **six active analysis branches**, producing **12 prediction tasks**. MIMIC/eICU diagnosis tables are research-derived, not four official prepackaged disease datasets. The original respiratory patient-split baseline is retained as a reference; hospital in-processing uses the rebuilt hospital split.','','## Source datasets','','| Application | Official source/version | Local model-ready rows | Outcome | Local data |','|---|---|---:|---|---|']
    for s in sources:lines.append(f'| {s["application"]} | [{s["source"]}]({s["official"]}) | {s["rows"]:,} | {s["target"]} | [Model-ready]({s["model_ready"]}) / [Raw source]({s["raw"]}) |')
    lines+=['','The additional eICU diagnosis comparison cohort contains **109,378 ICU stays, 88,955 patients and 206 hospitals**, from the same eICU source used for respiratory mortality. [Its four target tables](cross_dataset/four_conditions/eicu) and [exact phenotype definitions](cross_dataset/four_conditions/phenotype_definitions.json) are retained. MIMIC contains 73,990 selected stays from 53,931 patients. These are project cohort counts, not the full source database sizes.','','## Active preprocessing -> latent -> in-processing branches','','| Branch | Targets | Train / validation / test rows | External rows | Preprocessed / latent width |','|---|---|---|---:|---|']
    for b in branch:
        parts=b['partitions'];counts=' / '.join(f'{parts[s]["rows"]:,}' for s in ('train','validation','test'));ext=parts.get('external',{}).get('rows',0)
        lines.append(f'| {b["branch"]} | {", ".join(b["tasks"])} | {counts} | {ext:,} | {b["preprocessed_features"]} / {b["latent_features"]} |')
    lines+=['','Respiratory hospital preprocessing excludes 550 stays from 212 patients who cross partitions, leaving 28,292 stays. The three respiratory hospital partitions contain 137 / 34 / 29 hospitals. The eICU diagnosis partitions contain 141 / 34 / 31 hospitals. Patients are disjoint across each internal partition. MIMIC internal splitting is by patient within one institution.','','## Complete files','','[DATASET_FILES.csv](DATASET_FILES.csv) lists every complete preprocessing, latent and final prediction CSV. [DATASETS.json](DATASETS.json) provides the machine-readable source/branch catalogue. Individual dashboards link the exact split files. Keep train, validation, test and external files separate.','', '| Stage | Code | Outputs | Individual dashboards |','|---|---|---|---|']
    for stage in STAGES:lines.append(f'| {stage} | [{stage}/code]({stage}/code/README.md) | [{stage}/outputs]({stage}/outputs/README.md) | [{stage}/plots]({stage}/plots/index.html) |')
    lines+=['','## Meaning and limits','','- Survey heart/diabetes/hypertension labels are self-report, not hospital diagnoses. These surveys cannot establish hospital generalization. Survey weights/design variables are preserved, but the current experiments are unweighted.','- Respiratory mortality and respiratory recorded diagnosis are different targets. The ICU diagnosis labels overlap. Zero means no matching recorded diagnosis among documentation-eligible encounters, not confirmed absence of disease. MIMIC discharge ICD coding differs from eICU diagnosis/history documentation.','- Diagnosis experiments are retrospective recorded-label classification, not future disease prediction. Shared ICU baseline features are first-24-hour HR/RR/SpO2 summaries plus age group and sex, with training-fitted imputation/encoding.','- All source and external transforms are fitted only on source training rows. No target, ID, split or `sens_*` metadata is a latent predictor. Age/sex also occur as original predictor features, as documented in the protocols.','- Cross-hospital evidence comes from held-out eICU hospitals. Cross-database institutional non-overlap cannot be proven from anonymous eICU hospital IDs. Results are single-seed, bounded-budget experiments with mixed utility/fairness effects and explicit method adaptations.']
    (ROOT/'DATASETS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return items

def notebooks():
    setup="from pathlib import Path\nimport json, pandas as pd\nfrom IPython.display import display, Image\nROOT=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'DATASETS.json').exists())\n"
    def write(path,title,code):
        nb=nbformat.v4.new_notebook();nb.cells=[nbformat.v4.new_markdown_cell('# '+title+'\n\nThis viewer reads saved outputs. Training and verification code is in this code folder. [Open the project](../../START_HERE.html). It does not retrain or select models.'),nbformat.v4.new_code_cell(setup)]+[nbformat.v4.new_code_cell(c) for c in code];nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'};nbformat.validate(nb)
        target=ROOT/path
        if target.exists():
            previous=nbformat.read(target,as_version=4)
            if [(c.cell_type,c.source) for c in previous.cells]==[(c.cell_type,c.source) for c in nb.cells]:return
        nbformat.write(nb,target)
    for name in NAMES:
        write(f'preprocessing/code/{name}_verified_preprocessing.ipynb',TITLES[name]+' - verified preprocessing',[f"FOLDER=ROOT/'preprocessing/outputs/verified_baseline/{name}'\nfor split in ('train','validation','test'):\n    print(split)\n    display(pd.read_csv(FOLDER/f'{name}_{{split}}_preprocessed.csv',nrows=5))\ndisplay(Image(filename=str(ROOT/'preprocessing/plots/baseline/{name}/preprocessing_checks.png')))",f"report=FOLDER/'{'csv_validation.json' if name=='mimic' else 'validation.json'}'\ndisplay(json.loads(report.read_text()))"])
    write('preprocessing/code/verified_preprocessing.ipynb','Verified preprocessing overview',["report=json.loads((ROOT/'preprocessing/outputs/audit_current/confirmation.json').read_text())\nassert report['passed']\ndisplay(report)\nfor name in ('heart','respiratory','diabetes','hypertension','mimic'):\n    display(Image(filename=str(ROOT/'preprocessing/plots/baseline'/name/'preprocessing_checks.png')))"])
    write('preprocessing/code/hospital_preprocessing_latent.ipynb','Hospital preprocessing and latent outputs',["report=json.loads((ROOT/'latent/outputs/hospital/independent_validation.json').read_text())\nassert report['passed']\nfor name,branch in report['branches'].items():\n    print(name)\n    display(pd.DataFrame(branch['partitions']).T)\n    display(Image(filename=str(ROOT/'latent/plots/hospital'/name/'latent_checks.png')))\nfor name in ('respiratory','eicu','mimic'):\n    display(Image(filename=str(ROOT/'preprocessing/plots/hospital'/name/'preprocessing_checks.png')))"])
    write('latent/code/latent_pipeline.ipynb','Latent representations - five sources and hospital branches',["report=json.loads((ROOT/'latent/outputs/validation.json').read_text())\nassert report['passed']\ndisplay(pd.read_csv(ROOT/'latent/outputs/method_comparison.csv'))\nfor path in sorted((ROOT/'latent/plots').rglob('*.png')):\n    display(Image(filename=str(path)))"])
    write('inprocessing/code/inprocessing_pipeline.ipynb','In-processing measured results',["report=json.loads((ROOT/'inprocessing/outputs/validation.json').read_text())\nassert report['passed']\nresults=pd.read_csv(ROOT/'inprocessing/outputs/method_comparison.csv')\ndisplay(results[results.selected & results.split.ne('validation')])\nfor path in sorted((ROOT/'inprocessing/plots').rglob('*comparison.png')):\n    if path.name!='method_comparison.png':display(Image(filename=str(path)))"])

def guides(items):
    tasks={'preprocessing':['confirm_saved_outputs.py','audit_saved_preprocessing.py','audit_and_prepare.py','prepare_four_condition_baseline.py','publish_hospital_files.py'], 'latent':['validate_latent.py','validate_hospital_latent.py','check_handoff.py','run_latent.py','build_hospital_latent.py'], 'inprocessing':['validate.py','check_exports.py','test_core.py','run_inprocessing.py','make_report.py']}
    for stage in STAGES:
        tasks[stage].append('build_bias_dashboards.py')
        code=ROOT/stage/'code';lines=['# '+stage.title()+' code','','Python scripts and notebooks live here. Fitted artifacts/complete CSVs are in `../outputs`; dashboard pages and figures are in `../plots`.','', '## Main scripts','']+[f'- [{name}]({name})' for name in tasks[stage]]
        lines+=['','Run verification from `D:/complete` using the full path, for example:', '',f'`python {stage}/code/{tasks[stage][0]}`','', 'Training/rebuild scripts overwrite their own generated artifacts when explicitly run. A changed preprocessing or latent dataset requires rebuilding and revalidating its downstream models. Existing in-processing tasks are preserved; the validator detects changed input hashes.','', 'Refresh navigation/dashboards with `python dataset_audit/publish_layout.py`, then save notebook outputs with `python dataset_audit/execute_current_notebooks.py --stage all`. This publisher reads existing results and does not retrain models.']
        if stage=='preprocessing':lines+=['','Raw clinical extraction remains in [cross_dataset](../../cross_dataset/four_conditions/README.md), and original source-to-cohort scripts remain beside their source data. The active numerical preprocessing for both ICU directions is `prepare_four_condition_baseline.py` here.']
        lines+=['',f'Generate one file per bias/metric with `python {stage}/code/build_bias_dashboards.py`. Shared rendering code and assets live in `dataset_audit` at the project root; plot data are saved under this stage\'s `outputs/bias_dashboards`. Install the dependencies listed in [requirements.txt](requirements.txt).']
        (code/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        output=['# '+stage.title()+' outputs','','These are numeric outputs, fitted models, metadata and measured reports. Plots are in [../plots](../plots/index.html).','', '| Complete CSV |','|---|']
        for item in items:
            if item['stage']==stage:output.append(f'| [{Path(item["file"]).name}]({os.path.relpath(ROOT/item["file"],ROOT/stage/"outputs").replace(chr(92),"/")}) |')
        (ROOT/stage/'outputs/README.md').write_text('\n'.join(output)+'\n',encoding='utf-8')
        (ROOT/stage/'README.md').write_text(f'# {stage.title()}\n\n- [Code and notebooks](code/README.md)\n- [Complete outputs](outputs/README.md)\n- [Individual dashboards](plots/index.html)\n- [Complete dataset catalogue](../DATASETS.md)\n\n'+('Measured results: [REPORT.md](outputs/REPORT.md). Method adaptations: [PLAN.md](code/PLAN.md).\n' if stage!='preprocessing' else 'The hospital branch is used for cross-hospital in-processing. Original respiratory patient-split outputs remain a reference.\n'),encoding='utf-8')
        p=ROOT/stage/'index.html'
        if stage=='inprocessing':
            branches=read('inprocessing/outputs/manifest.json')['selected_exports']
            body=cards(p,[(f'inprocessing/plots/{b}/index.html',TITLES[b],SCOPE[b]) for b in branches])
            body+='<div class="actions">'+link(p,'inprocessing/code/README.md','Code and notebooks')+link(p,'inprocessing/outputs/README.md','All complete prediction outputs')+'</div>'
            page(p,'Individual in-processing dashboards',body,'Choose a dataset. MIMIC and eICU each contain separate disease dashboards.')
        else:page(p,stage.title(),cards(p,[(f'{stage}/plots/index.html','Individual dashboards','Choose a dataset or hospital branch.'),(f'{stage}/code/README.md','Code and notebooks','Active scripts and saved notebook outputs.'),(f'{stage}/outputs/README.md','Outputs','Complete CSVs, fitted models and verification evidence.')]))
    (ROOT/'preprocessing/FILES.md').write_text('# Preprocessing files\n\n[Individual dashboards](plots/index.html) | [All complete CSVs](outputs/README.md) | [Code and notebooks](code/README.md) | [Dataset catalogue](../DATASETS.md)\n',encoding='utf-8')
    root=['# Complete project: current organized files','','**[Open START_HERE.html](START_HERE.html)** or [the complete dataset catalogue](DATASETS.md).','', '```text','preprocessing/   code/   outputs/   plots/','latent/          code/   outputs/   plots/','inprocessing/    code/   outputs/   plots/','```','', 'Each stage has separate code, outputs and plots. In-processing has six individual branch dashboards and separate disease pages for the MIMIC/eICU diagnosis tasks.','', '| Stage | Code | Outputs | Dashboards |','|---|---|---|---|']
    for s in STAGES:root.append(f'| {s} | [Code]({s}/code/README.md) | [Outputs]({s}/outputs/README.md) | [Dashboards]({s}/plots/index.html) |')
    root+=['','Five source datasets produce 12 prediction tasks. The respiratory hospital experiment uses 29 test hospitals; eICU diagnosis experiments use 31 test hospitals. MIMIC/eICU external transfer is reported separately. Survey results are not hospital evidence. Methods are explicit adaptations; performance/fairness effects are mixed.','', '[Measured in-processing results](inprocessing/outputs/REPORT.md) | [All complete CSVs](DATASET_FILES.csv) | [Current organization and validation checks](dataset_audit/layout_validation.json)','', 'Old experiments are outside this folder in `D:/complete_old_archive_20261111_212716`, including the five historical audit/code copies found during this cleanup. [Original archive](dataset_audit/archive_manifest.json) / [Additional archive record](dataset_audit/remaining_old_audit_archive.json). Frozen selections and model/prediction files are preserved; [layout move hashes](dataset_audit/layout_migration.json) record the relocation.']
    (ROOT/'README.md').write_text('\n'.join(root)+'\n',encoding='utf-8')
    p=ROOT/'START_HERE.html';page(p,'Your complete project',cards(p,[(f'{s}/plots/index.html',s.title()+' dashboards','Individual dataset dashboards, with direct links to code and outputs.') for s in STAGES])+cards(p,[('DATASETS.md','Complete dataset catalogue','Official versions, actual targets, cohort counts, splits and all complete files.'),('DATASET_FILES.csv','Complete CSV inventory',f'{len(items)} complete preprocessing, latent and prediction files.'),('inprocessing/outputs/REPORT.md','Measured results','All 12 tasks, internal tests and external database evaluations.')])+'<section><h2>Folder structure</h2><p>Each stage contains <code>code/</code>, <code>outputs/</code> and <code>plots/</code>. Open a dataset dashboard to reach its exact files.</p></section>','Preprocessing -> latent -> in-processing. Current data, code and plots in D:/complete.')

def main():
    items=catalogue();input_dashboards();inprocessing_dashboards();notebooks();guides(items)
    if (ROOT/'dataset_audit/bias_dashboard_manifest.json').exists():
        from build_bias_dashboards import publish_saved
        publish_saved()
        path=ROOT/'README.md'
        path.write_text(path.read_text(encoding='utf-8')+'\n[One file per bias: open the dashboard catalogue](BIAS_DASHBOARDS.md). Each stage plot index now lists the individual bias/metric HTML files. Summary PNGs are current supporting diagnostics.\n\n[Final verification report](dataset_audit/FINAL_REVIEW.md) | [Code and algorithm review](dataset_audit/CODE_REVIEW.md) | [Every current stage file](dataset_audit/active_recheck/ALL_STAGE_FILES.csv)\n',encoding='utf-8')
        path=ROOT/'START_HERE.html';text=path.read_text(encoding='utf-8')
        text=text.replace('Individual dataset dashboards, with direct links to code and outputs.','One separate HTML file per bias or latent metric.')
        extra='<section><h2>Individual bias files and verification</h2><p><a href="BIAS_DASHBOARDS.md">All 144 individual dashboards</a> | <a href="dataset_audit/FINAL_REVIEW.md">Final review</a> | <a href="dataset_audit/CODE_REVIEW.md">Code and algorithm review</a></p></section>'
        path.write_text(text.replace('</main>',extra+'</main>'),encoding='utf-8')
    print(f'Published individual dashboards, nine notebooks, complete dataset catalogue and {len(items)} CSV links.',flush=True)

if __name__=='__main__':main()
