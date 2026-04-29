"""Execute read-only result notebooks and persist their visible outputs."""
from pathlib import Path
import argparse,json
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('preprocessing','latent','inprocessing','all'),default='all');args=parser.parse_args()
paths=[]
if args.stage in ('all','preprocessing'):
    paths.extend(ROOT/'preprocessing/code'/f'{name}_verified_preprocessing.ipynb' for name in ('heart','respiratory','diabetes','hypertension','mimic'))
    paths.append(ROOT/'preprocessing/code/hospital_preprocessing_latent.ipynb')
    paths.append(ROOT/'preprocessing/code/verified_preprocessing.ipynb')
if args.stage in ('all','latent'):paths.append(ROOT/'latent/code/latent_pipeline.ipynb')
if args.stage in ('all','inprocessing'):paths.append(ROOT/'inprocessing/code/inprocessing_pipeline.ipynb')
report={'passed':False,'notebooks':{}}
try:
    for path in paths:
        print('Execute '+path.name,flush=True)
        nb=nbformat.read(path,as_version=4)
        client=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
        client.execute()
        for cell in nb.cells:
            if cell.cell_type=='code':
                assert cell.execution_count is not None
                assert all(out.output_type!='error' for out in cell.outputs)
        nbformat.validate(nb);nbformat.write(nb,path)
        report['notebooks'][path.relative_to(ROOT).as_posix()]={'passed':True,'executed_cells':sum(c.cell_type=='code' for c in nb.cells),'visible_outputs_saved':True}
    report['passed']=True
finally:(ROOT/'dataset_audit'/f'notebook_execution_{args.stage}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('Notebook execution passed',flush=True)
