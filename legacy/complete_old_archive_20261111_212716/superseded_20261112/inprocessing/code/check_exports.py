"""Check complete prediction CSVs and links against frozen model artifacts."""
import json,re
import numpy as np
import pandas as pd
from PIL import Image
from run_inprocessing import ROOT,HERE,OUT,JOBS,load,sha,write

def main():
    manifest=json.loads((OUT/'manifest.json').read_text());assert manifest['completed']
    report={'passed':False,'exports':{}}
    count=0
    for dataset,parts in manifest['selected_exports'].items():
        print('Check CSVs '+dataset,flush=True)
        tasks=[t for d,t in JOBS if d==dataset];x,y,rows,_=load(dataset,tasks[0])
        report['exports'][dataset]={}
        for split,item in parts.items():
            path=ROOT/item['csv'];assert sha(path)==item['sha256']
            frame=pd.read_csv(path,dtype={c:'string' for c in rows[split] if c.startswith('meta_') and not pd.api.types.is_numeric_dtype(rows[split][c])})
            assert len(frame)==len(x[split])==item['rows']
            zcols=[f'z{i+1:02}' for i in range(x[split].shape[1])]
            np.testing.assert_allclose(frame[zcols],x[split],atol=1e-6,rtol=1e-6)
            pd.testing.assert_frame_equal(frame[list(rows[split])],rows[split],check_dtype=False)
            assert frame.meta_evaluation_split.eq(split).all()
            for task in tasks:
                folder=OUT/dataset/task;decision=json.loads((folder/'selection_frozen.json').read_text())
                method=decision['selected'];threshold=decision['validation'][method]['threshold']
                with np.load(folder/method/f'{split}_predictions.npz') as a:p=a['probability']
                np.testing.assert_allclose(frame['probability_'+task],p,atol=1e-7,rtol=1e-7)
                np.testing.assert_array_equal(frame['prediction_'+task],(p>=threshold).astype(int))
                target='target_'+task if dataset in ('eicu_hospital','mimic_transfer') else 'target'
                if target=='target':expected=y[split]
                else:
                    with np.load(ROOT/'latent/outputs/hospital'/dataset/f'y_{split}.npz') as a:expected=a[task]
                np.testing.assert_array_equal(frame[target],expected)
            report['exports'][dataset][split]={'passed':True,'rows':len(frame),'tasks':tasks};count+=1
    summary=pd.read_csv(OUT/'method_comparison.csv');seen=0
    for dataset,task in JOBS:
        folder=OUT/dataset/task;complete=json.loads((folder/'COMPLETE.json').read_text())
        for method in complete['methods']:
            for split in complete['splits']:
                if split=='train':continue
                row=summary[(summary.dataset==dataset)&(summary.task==task)&(summary.method==method)&(summary.split==split)]
                assert len(row)==1
                m=json.loads((folder/method/f'{split}_metrics.json').read_text())
                for key,value in m.items():
                    if key=='group_disparities':continue
                    if value is None:assert pd.isna(row.iloc[0][key])
                    else:np.testing.assert_allclose(row.iloc[0][key],value,atol=1e-8,rtol=1e-8)
                assert bool(row.iloc[0]['selected'])==(method==complete['selected']);seen+=1
    assert len(summary)==seen
    images=0;links=0
    for page in list((HERE/'plots').rglob('*.html'))+list((ROOT/'latent/plots').rglob('*.html')):
        for link in re.findall(r'(?:src|href)="([^"]+)"',page.read_text(encoding='utf-8')):
            path=page.parent/link;assert path.exists(),str(path);links+=1
            if path.suffix=='.png':
                with Image.open(path) as im:assert im.width>600 and im.height>300;im.verify()
                images+=1
    report.update({'passed':True,'complete_prediction_csvs':count,'summary_metric_rows_verified':seen,'linked_plots_checked':images,'local_links_checked':links})
    write(OUT/'export_validation.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='exports'}),flush=True)

if __name__=='__main__':main()
