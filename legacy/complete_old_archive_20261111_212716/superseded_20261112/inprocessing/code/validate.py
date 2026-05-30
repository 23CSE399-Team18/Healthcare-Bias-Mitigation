"""Read-back checks of saved models, frozen decisions and measured holdouts."""
import argparse,json
import numpy as np
import pandas as pd
import torch,joblib
from threadpoolctl import threadpool_limits
from run_inprocessing import ROOT,HERE,OUT,JOBS,load,sha,write,CONFIG
from core import Network,probabilities,grouping,evaluate,threshold_for

def same(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for key in a:same(a[key],b[key])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    elif a is None:assert b is None
    elif isinstance(a,(int,float,np.generic)):np.testing.assert_allclose(a,b,atol=1e-8,rtol=1e-8)
    else:assert a==b

def check_task(dataset,task):
    raw,y,rows,hashes=load(dataset,task);folder=OUT/dataset/task
    complete=json.loads((folder/'COMPLETE.json').read_text());decision=json.loads((folder/'selection_frozen.json').read_text())
    assert complete['passed'] and sha(folder/'selection_frozen.json')==complete['selection_sha256']
    # Frozen records keep original paths; compare through the hash-verified relocation map.
    migration=json.loads((ROOT/'dataset_audit/layout_migration.json').read_text())
    mapping={str(ROOT/r['old']):str(ROOT/r['new']) for r in migration['file_moves']}
    expected_hashes={mapping.get(p,p):v for p,v in decision['input_hashes'].items()}
    assert expected_hashes==hashes and not decision['config']['test_used_for_selection']
    scale=joblib.load(folder/'input_scaler.joblib')
    np.testing.assert_allclose(scale.mean_,raw['train'].mean(0,dtype=float),atol=1e-8)
    assert scale.n_samples_seen_==len(y['train'])
    x={s:scale.transform(a).astype(np.float32) for s,a in raw.items()}
    eligible=[];base=decision['validation']['ERM']
    for method,m in decision['validation'].items():
        if method!='Uncompressed_logistic_reference' and m['macro_f1']>=base['macro_f1']-.02 and m['auroc']>=base['auroc']-.02 and m['positive_recall']>=base['positive_recall']-.05:eligible.append(method)
    assert sorted(eligible)==sorted(decision['eligible'])
    def rank(m):
        v=decision['validation'][m];gap=v['max_equalized_odds_gap'];return (gap if gap is not None else float('inf'),-v['macro_f1'],v['brier'],m!='ERM')
    assert decision['selected']==min(eligible,key=rank)
    methods={}
    for method in complete['methods']:
        dest=folder/method;training=json.loads((dest/'training.json').read_text())
        assert training['test_rows_used']==training['external_rows_used']==0
        if (dest/'model.pt').exists():
            saved=torch.load(dest/'model.pt',map_location='cpu',weights_only=True)
            model=Network(saved['dimension'],saved['heads']);model.load_state_dict(saved['state_dict']);model.eval()
            assert all(h['training_rows']==len(y['train']) for h in training['history'])
        else:model=joblib.load(dest/'model.joblib')
        for split in x:
            with np.load(dest/f'{split}_predictions.npz') as a:p=a['probability']
            assert p.shape==(len(y[split]),) and np.isfinite(p).all() and np.all((p>=0)&(p<=1))
            ix=np.random.default_rng(37).choice(len(p),min(131,len(p)),replace=False)
            if method=='Uncompressed_logistic_reference':
                if dataset in ('respiratory_hospital','eicu_hospital','mimic_transfer'):
                    source=json.loads((ROOT/'latent/outputs/hospital'/dataset/'manifest.json').read_text())['source'];path=ROOT/source
                else:path=ROOT/'preprocessing/outputs/verified_baseline'/dataset
                with np.load(path/f'X_{split}.npz') as a:query=a['X'][ix]
                expected=model.predict_proba(query)[:,1]
            elif method=='FOMO_weight_search':expected=model.predict_proba(x[split][ix])[:,1]
            else:expected=probabilities(model,x[split][ix])
            np.testing.assert_allclose(p[ix],expected,atol=1e-6,rtol=1e-6)
            if split=='validation':same(threshold_for(y[split],p),decision['validation'][method]['threshold'])
            if split!='train':
                metrics,groups,bins=evaluate(y[split],p,grouping(rows[split]),decision['validation'][method]['threshold'])
                same(metrics,json.loads((dest/f'{split}_metrics.json').read_text()))
                same(bins,json.loads((dest/f'{split}_reliability.json').read_text()))
                csv=pd.read_csv(dest/f'{split}_groups.csv')
                pd.testing.assert_frame_equal(csv,pd.DataFrame(groups),check_dtype=False,atol=1e-8,rtol=1e-8)
        methods[method]={'passed':True,'saved_model_reproduces_predictions':True,'all_metrics_and_group_tables_recomputed':True}
    if dataset in ('respiratory_hospital','eicu_hospital'):
        patient='meta_split_group_uniquepid' if dataset=='respiratory_hospital' else 'meta_patient_id'
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            assert not set(rows[a].meta_domain_hospitalid)&set(rows[b].meta_domain_hospitalid)
            assert not set(rows[a][patient])&set(rows[b][patient])
    return {'passed':True,'selected':decision['selected'],'methods':methods,'frozen_selection_verified':True}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset');args=parser.parse_args();torch.set_num_threads(2)
    report={'passed':False,'tasks':{}}
    try:
        with threadpool_limits(limits=2):
            for dataset,task in JOBS:
                if args.dataset and dataset!=args.dataset:continue
                print('Verify '+dataset+'/'+task,flush=True);report['tasks'][dataset+'/'+task]=check_task(dataset,task)
        report['passed']=True
    finally:write(OUT/(f'validation_{args.dataset}.json' if args.dataset else 'validation.json'),report)
    print('In-processing verification passed',flush=True)

if __name__=='__main__':main()
