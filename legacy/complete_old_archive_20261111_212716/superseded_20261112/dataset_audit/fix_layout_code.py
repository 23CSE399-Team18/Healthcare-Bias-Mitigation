"""Update split path construction after the recorded layout migration."""
from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[1]

def edit(path,replacements):
    p=ROOT/path;t=p.read_text(encoding='utf-8')
    for old,new in replacements:
        if old not in t:
            assert new in t,(path,old)
            continue
        t=t.replace(old,new)
    ast.parse(t);p.write_text(t,encoding='utf-8')

edit('latent/code/load_for_inprocessing.py',[("Path(__file__).resolve().parent/'outputs'","Path(__file__).resolve().parents[1]/'outputs'")])
edit('inprocessing/code/run_inprocessing.py',[("str(ROOT/'latent')","str(ROOT/'latent/code')"),("'hospital_outputs' if hospital","'outputs/hospital' if hospital"),("HERE/'core.py'","HERE/'code/core.py'"),("HERE/'progress.log'","OUT/'progress.log'")])
edit('inprocessing/code/validate.py',[("assert decision['input_hashes']==hashes and not decision['config']['test_used_for_selection']","# Frozen records keep original paths; compare through the hash-verified relocation map.\n    migration=json.loads((ROOT/'dataset_audit/layout_migration.json').read_text())\n    mapping={str(ROOT/r['old']):str(ROOT/r['new']) for r in migration['file_moves']}\n    expected_hashes={mapping.get(p,p):v for p,v in decision['input_hashes'].items()}\n    assert expected_hashes==hashes and not decision['config']['test_used_for_selection']")])
edit('preprocessing/code/prepare_four_condition_baseline.py',[("default=ROOT / \"four_conditions/baseline_preprocessed\"","default=ROOT.parent / \"preprocessing/outputs/cross_dataset_baseline\"")])
edit('preprocessing/code/audit_saved_preprocessing.py',[("ROOT / f\"four_conditions/baseline_preprocessed/{source_name}_to_{external_name}\"","PROJECT / f\"preprocessing/outputs/cross_dataset_baseline/{source_name}_to_{external_name}\"")])
for name in ('confirm_saved_outputs.py','audit_and_prepare.py'):
    edit('preprocessing/code/'+name,[("root / \"baseline_preprocessed\"","ROOT / \"preprocessing/outputs/cross_dataset_baseline\"")])
edit('latent/code/run_latent.py',[("HERE/'models.py'","HERE/'code/models.py'"),("HERE/'progress.log'","OUT/'progress.log'"),("fig.savefig(OUT/name/'comparison.png'","(HERE/'plots'/name).mkdir(parents=True,exist_ok=True)\n    fig.savefig(HERE/'plots'/name/'comparison.png'"),("fig.savefig(OUT/'group_alignment.png'","(HERE/'plots').mkdir(parents=True,exist_ok=True)\n    fig.savefig(HERE/'plots/group_alignment.png'"),("(HERE/'REPORT.md')","(OUT/'REPORT.md')")])
edit('latent/code/check_handoff.py',[("BASE/'latent_pipeline.ipynb'","BASE/'code/latent_pipeline.ipynb'")])
edit('dataset_audit/recheck_complete.py',[("str(ROOT / 'latent'), str(ROOT / 'preprocessing')","str(ROOT / 'latent/code'), str(ROOT / 'preprocessing/code')"),("(ROOT / 'preprocessing').glob('*.ipynb')","(ROOT / 'preprocessing/code').glob('*.ipynb')"),("preprocessing/outputs/verified_baseline/{n}/preprocessing_checks.png","preprocessing/plots/baseline/{n}/preprocessing_checks.png")])
edit('cross_dataset/tests/test_four_condition_baseline.py',[("sys.path.insert(0, str(Path(__file__).resolve().parents[1]))","sys.path.insert(0, str(Path(__file__).resolve().parents[1]))\nsys.path.insert(0, str(Path(__file__).resolve().parents[2]/'preprocessing/code'))")])
edit('inprocessing/code/check_exports.py',[("for page in (HERE/'index.html',ROOT/'latent/outputs/hospital/index.html'):","for page in list((HERE/'plots').rglob('*.html'))+list((ROOT/'latent/plots').rglob('*.html')):")])
edit('dataset_audit/execute_current_notebooks.py',[("choices=('preprocessing','inprocessing','all')","choices=('preprocessing','latent','inprocessing','all')"),("ROOT/'preprocessing'/f'{name}_verified_preprocessing.ipynb'","ROOT/'preprocessing/code'/f'{name}_verified_preprocessing.ipynb'"),("if args.stage in ('all','inprocessing'):","if args.stage in ('all','latent'):paths.append(ROOT/'latent/code/latent_pipeline.ipynb')\nif args.stage in ('all','inprocessing'):")])
# Every plot generator writes only to the corresponding plots folder.
for indent in ('    ','        '):
    edit('preprocessing/code/audit_and_prepare.py',[("\n"+indent+'fig.savefig(base / "preprocessing_checks.png"',"\n"+indent+"(ROOT/'preprocessing/plots/baseline'/base.name).mkdir(parents=True,exist_ok=True)\n"+indent+"fig.savefig(ROOT/'preprocessing/plots/baseline'/base.name/'preprocessing_checks.png'")])

publish="\nimport sys\nsys.path.insert(0,str(ROOT/'dataset_audit'))\nfrom publish_layout import main as publish_layout\npublish_layout()\n"
p=ROOT/'preprocessing/code/publish_verified_files.py';t=p.read_text(encoding='utf-8');t=t[:t.index('\nlines = [')]+publish;ast.parse(t);p.write_text(t,encoding='utf-8')
p=ROOT/'preprocessing/code/publish_hospital_files.py';t=p.read_text(encoding='utf-8')
t=t.replace("plot=pre/'preprocessing_checks.png'","plot=ROOT/'preprocessing/plots/hospital'/pre.name/'preprocessing_checks.png';plot.parent.mkdir(parents=True,exist_ok=True)")
t=t.replace("fig.savefig(folder/'latent_checks.png'","(ROOT/'latent/plots/hospital'/name).mkdir(parents=True,exist_ok=True);fig.savefig(ROOT/'latent/plots/hospital'/name/'latent_checks.png'")
t=t[:t.index("    guide+=[''")]+"    (ROOT/'preprocessing/outputs/hospital_baseline/export_validation.json').write_text(json.dumps({'passed':True,'exports':exports},indent=2),encoding='utf-8')\n"+'\n'.join('    '+line for line in publish.strip().splitlines())+"\n\nif __name__=='__main__':main()\n";ast.parse(t);p.write_text(t,encoding='utf-8')
p=ROOT/'inprocessing/code/make_report.py';t=p.read_text(encoding='utf-8')
t=t.replace("folder=OUT/dataset/task; selected=decision['selected']","folder=OUT/dataset/task; selected=decision['selected']\n    plotfolder=HERE/'plots'/dataset/task;plotfolder.mkdir(parents=True,exist_ok=True)")
t=t.replace("fig.savefig(folder/", "fig.savefig(plotfolder/")
t='\n'.join(line for line in t.splitlines() if not line.strip().startswith("(HERE/'index.html').write_text"))+'\n'
t=t.replace("(HERE/'REPORT.md')","(OUT/'REPORT.md')")
start=t.index("    guide=['# In-processing:");end=t.index("\nif __name__=='__main__':")
t=t[:start]+'\n'.join('    '+line for line in publish.strip().splitlines())+'\n'+t[end:];ast.parse(t);p.write_text(t,encoding='utf-8')
# A single publisher owns notebooks and individual dashboards, avoiding duplicate layouts.
p=ROOT/'latent/code/build_notebook.py';p.write_text("from pathlib import Path\nROOT=Path(__file__).resolve().parents[2]\n"+publish,encoding='utf-8')
for stage in ('preprocessing','latent','inprocessing'):
    for p in (ROOT/stage/'code').glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
print('Code path updates completed; all stage Python files parse.',flush=True)
