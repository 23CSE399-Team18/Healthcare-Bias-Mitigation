"""Train source-only classifiers, freeze selection, then score locked holdouts."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, copy, hashlib, json, sys, warnings
import numpy as np
import pandas as pd
import joblib
import torch
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, f1_score
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
from core import *

ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'inprocessing';OUT=HERE/'outputs'
sys.path.insert(0,str(ROOT/'latent/code'))
from load_for_inprocessing import load_latent
SEED=20261111
CONFIG={'epochs':6,'batch_size':1024,'learning_rate':.001,'weight_decay':.0001,'seed':SEED,'threshold_grid':[.05,.95,.025],
        'selection':'minimum max protected-group EO gap subject to F1/AUROC/recall guards relative to latent ERM','test_used_for_selection':False}
JOBS=[('heart','heart'),('diabetes','diabetes'),('hypertension','hypertension'),('respiratory_hospital','respiratory')]+[(n,t) for n in ('eicu_hospital','mimic_transfer') for t in ('heart','respiratory','diabetes','hypertension')]

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def log(message):
    line=datetime.now(timezone.utc).isoformat()+' '+message;print(line,flush=True)
    with (OUT/'progress.log').open('a',encoding='utf-8') as f:f.write(line+'\n')

def load(dataset,task):
    x,y,rows,hashes={},{},{},{}
    hospital=dataset in ('respiratory_hospital','eicu_hospital','mimic_transfer')
    base=ROOT/'latent'/('outputs/hospital' if hospital else 'outputs')/dataset
    manifest=json.loads((base/'manifest.json').read_text())
    splits=list(manifest['partitions']) if hospital else ['train','validation','test']
    for split in splits:
        if hospital:
            for filename,expected in manifest['partitions'][split]['hashes'].items():assert sha(base/filename)==expected
            with np.load(base/f'Z_{split}.npz') as a:x[split]=a['Z']
            with np.load(base/f'y_{split}.npz') as a:y[split]=a[task]
            cols=pd.read_csv(base/f'rows_{split}.csv',nrows=0).columns
            rows[split]=pd.read_csv(base/f'rows_{split}.csv',dtype={c:'string' for c in cols if c.startswith('meta_')})
            hashes[str(base/f'Z_{split}.npz')]=sha(base/f'Z_{split}.npz')
        else:
            x[split],y[split],rows[split]=load_latent(dataset,split)
            hashes[str(base/'selected'/f'Z_{split}.npz')]=sha(base/'selected'/f'Z_{split}.npz')
        hashes[str(base/f'rows_{split}.csv')]=sha(base/f'rows_{split}.csv')
        hashes[str(base/f'y_{split}.npz')]=sha(base/f'y_{split}.npz')
        assert len(x[split])==len(y[split])==len(rows[split]) and np.isfinite(x[split]).all()
    return x,y,rows,hashes

def train_neural(method,x,y,codes,levels,out):
    torch.manual_seed(SEED);rng=np.random.default_rng(SEED)
    keys=ATTRS if method=='AdversarialFair_tabular' else (('intersection',) if method=='Intersectional_adversary' else (('domain',) if method=='DANN_source_domains' else ()))
    heads=[max(1,len(levels[k])) for k in keys]
    model=Network(x['train'].shape[1],heads)
    optimizer=torch.optim.Adam(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    state={'dual':0.,'correct':torch.zeros(len(y['train'])),'ent_sum':0.,'ent_n':0}
    if method.startswith('GroupDRO'):
        key='sex_label' if method=='GroupDRO_sex_label' else 'intersection_label';state['q']=torch.ones(len(levels[key]))/len(levels[key])
    tensors={k:torch.from_numpy(v) for k,v in codes.items()}
    best=float('inf');checkpoint=None;history=[]
    for epoch in range(CONFIG['epochs']):
        model.train();order=rng.permutation(len(y['train']));total=0.;seen=0
        for start in range(0,len(order),CONFIG['batch_size']):
            ix=order[start:start+CONFIG['batch_size']];a=torch.from_numpy(x['train'][ix]);b=torch.from_numpy(y['train'][ix].astype(np.float32))
            g={k:v[ix] for k,v in tensors.items()}
            loss=loss_for(method,model,a,b,g,state,ix,epoch)
            assert torch.isfinite(loss)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            if method=='Fish_first_order':
                # Explicit ERM + first-order Fish hybrid: every row contributes
                # to BCE; up to four source domains supply the meta-update.
                labels=[int(v) for v in torch.unique(g['domain']) if v>=0 and (g['domain']==v).sum()>=4]
                if len(labels)>=2:
                    before=copy.deepcopy(model.state_dict());inner=torch.optim.SGD(model.parameters(),lr=.005)
                    for label in rng.choice(labels,min(4,len(labels)),replace=False):
                        mask=g['domain']==label;inner.zero_grad();F.binary_cross_entropy_with_logits(model(a[mask])[0],b[mask]).backward();inner.step()
                    after=model.state_dict();model.load_state_dict({k:before[k]+.1*(after[k]-before[k]) for k in before})
            total+=loss.detach().item()*len(ix);seen+=len(ix)
        valp=probabilities(model,x['validation']);score=float(log_loss(y['validation'],np.clip(valp,1e-7,1-1e-7),labels=[0,1]))
        history.append({'epoch':epoch+1,'training_rows':seen,'training_objective':total/seen,'validation_log_loss':score})
        if score<best:best=score;checkpoint=copy.deepcopy(model.state_dict());best_epoch=epoch+1
    model.load_state_dict(checkpoint)
    torch.save({'state_dict':checkpoint,'dimension':x['train'].shape[1],'heads':heads,'method':method},out/'model.pt')
    write(out/'training.json',{'method':method,'history':history,'chosen_epoch':best_epoch,'training_rows':len(y['train']),'test_rows_used':0,'external_rows_used':0,'all_training_rows_visited_each_epoch':True})
    return model

def train_fomo(x,y,groups,out):
    rng=np.random.default_rng(SEED);labels=sorted(set(groups['train']['intersection']))
    code={v:i for i,v in enumerate(labels)};ix=rng.choice(len(y['train']),min(20000,len(y['train'])),replace=False)
    groupids=np.array([code[v] for v in groups['train']['intersection']]);population=[np.zeros(len(labels))]+[rng.normal(0,.5,len(labels)) for _ in range(7)]
    history=[];pool=[]
    for generation in range(3):
        evaluated=[]
        for weights in population:
            model=LogisticRegression(C=.1,max_iter=400,tol=1e-4).fit(x['train'][ix],y['train'][ix],sample_weight=np.exp(weights[groupids[ix]]))
            p=model.predict_proba(x['validation'])[:,1];threshold=threshold_for(y['validation'],p)
            metrics,_,_=evaluate(y['validation'],p,groups['validation'],threshold)
            error=1-metrics['macro_f1'];gap=metrics['max_equalized_odds_gap']
            evaluated.append((error,gap if gap is not None else 1.,weights.copy()))
        pool+=evaluated
        frontier=[a for a in pool if not any(b[0]<=a[0] and b[1]<=a[1] and (b[0]<a[0] or b[1]<a[1]) for b in pool)]
        history.append({'generation':generation+1,'validation_objectives':[[a[0],a[1]] for a in evaluated],'pareto_count':len(frontier)})
        population=[frontier[i%len(frontier)][2].copy() for i in range(2)]+[np.clip(frontier[rng.integers(len(frontier))][2]+rng.normal(0,.2,len(labels)),-2,2) for _ in range(6)]
    # Within 0.02 F1 of the best searched weighting, minimize the fairness gap.
    best_error=min(a[0] for a in pool);winner=min((a for a in pool if a[0]<=best_error+.02),key=lambda a:(a[1],a[0]))
    model=LogisticRegression(C=.1,max_iter=1000).fit(x['train'],y['train'],sample_weight=np.exp(winner[2][groupids]))
    joblib.dump(model,out/'model.joblib')
    write(out/'training.json',{'method':'FOMO_weight_search','search_training_rows':len(ix),'refit_training_rows':len(y['train']),'test_rows_used':0,'external_rows_used':0,'history':history,'group_weight_labels':labels,'group_log_weights':winner[2].tolist()})
    return model

def task_run(dataset,task):
    folder=OUT/dataset/task;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'COMPLETE.json').exists():
        saved=json.loads((folder/'COMPLETE.json').read_text());assert saved['passed'];log('Already complete '+dataset+'/'+task);return
    raw,y,rows,hashes=load(dataset,task);groups={s:grouping(r) for s,r in rows.items()}
    scaler=StandardScaler().fit(raw['train']);x={s:scaler.transform(v).astype(np.float32) for s,v in raw.items()};joblib.dump(scaler,folder/'input_scaler.joblib')
    codes,levels=code_groups(groups['train'],y['train']);write(folder/'groups.json',levels)
    methods=METHODS+['FOMO_weight_search'];results={};models={};skipped={}
    for method in methods:
        if method in DOMAIN_METHODS and len(levels.get('domain',[]))<2:
            skipped[method]='Fewer than two observed source-training domains with 50 rows; not applicable.';continue
        dest=folder/method;dest.mkdir(exist_ok=True)
        log(f'Train {dataset}/{task}/{method}')
        model=train_fomo(x,y,groups,dest) if method=='FOMO_weight_search' else train_neural(method,x,y,codes,levels,dest)
        p=model.predict_proba(x['validation'])[:,1] if method=='FOMO_weight_search' else probabilities(model,x['validation'])
        threshold=threshold_for(y['validation'],p);metrics,table,bins=evaluate(y['validation'],p,groups['validation'],threshold)
        np.savez_compressed(dest/'validation_predictions.npz',probability=p)
        write(dest/'validation_metrics.json',metrics);pd.DataFrame(table).to_csv(dest/'validation_groups.csv',index=False);write(dest/'validation_reliability.json',bins)
        results[method]=metrics;models[method]=model
    # Separate no-latent reference, fitted with the same source training rows.
    if dataset in ('respiratory_hospital','eicu_hospital','mimic_transfer'):
        source_record=json.loads((ROOT/'latent/outputs/hospital'/dataset/'manifest.json').read_text())
        input_folder=ROOT/source_record['source']
    else:input_folder=ROOT/'preprocessing/outputs/verified_baseline'/dataset
    uncompressed={}
    for split in x:
        with np.load(input_folder/f'X_{split}.npz') as archive:uncompressed[split]=archive['X']
    reference=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000))
    reference.fit(uncompressed['train'],y['train'])
    dest=folder/'Uncompressed_logistic_reference';dest.mkdir(exist_ok=True)
    joblib.dump(reference,dest/'model.joblib')
    p=reference.predict_proba(uncompressed['validation'])[:,1]
    threshold=threshold_for(y['validation'],p);metrics,table,bins=evaluate(y['validation'],p,groups['validation'],threshold)
    np.savez_compressed(dest/'validation_predictions.npz',probability=p)
    write(dest/'validation_metrics.json',metrics);pd.DataFrame(table).to_csv(dest/'validation_groups.csv',index=False);write(dest/'validation_reliability.json',bins)
    results['Uncompressed_logistic_reference']=metrics;models['Uncompressed_logistic_reference']=reference
    write(dest/'training.json',{'training_rows':len(y['train']),'test_rows_used':0,'external_rows_used':0,'reference_only':True})
    base=results['ERM'];eligible=[]
    for method,m in results.items():
        if method=='Uncompressed_logistic_reference':continue
        if m['macro_f1']>=base['macro_f1']-.02 and m['auroc']>=base['auroc']-.02 and m['positive_recall']>=base['positive_recall']-.05:eligible.append(method)
    def selection_key(method):
        m=results[method];gap=m['max_equalized_odds_gap'];return (gap if gap is not None else float('inf'),-m['macro_f1'],m['brier'],method!='ERM')
    selected=min(eligible,key=selection_key)
    decision={'dataset':dataset,'task':task,'selected':selected,'eligible':eligible,'validation':results,'skipped':skipped,'input_hashes':hashes,'config':CONFIG,'script_hash':sha(__file__),'core_hash':sha(HERE/'code/core.py'),'frozen_utc':datetime.now(timezone.utc).isoformat(),'holdout_predictions_scored':False}
    write(folder/'selection_frozen.json',decision);frozen_hash=sha(folder/'selection_frozen.json')
    # No fitting or selection occurs below this line.
    for method,model in models.items():
        dest=folder/method
        for split in x:
            if split=='validation':continue
            if method=='Uncompressed_logistic_reference':p=model.predict_proba(uncompressed[split])[:,1]
            elif method=='FOMO_weight_search':p=model.predict_proba(x[split])[:,1]
            else:p=probabilities(model,x[split])
            np.savez_compressed(dest/f'{split}_predictions.npz',probability=p)
            if split!='train':
                metrics,table,bins=evaluate(y[split],p,groups[split],results[method]['threshold'])
                write(dest/f'{split}_metrics.json',metrics);pd.DataFrame(table).to_csv(dest/f'{split}_groups.csv',index=False);write(dest/f'{split}_reliability.json',bins)
    assert sha(folder/'selection_frozen.json')==frozen_hash
    write(folder/'COMPLETE.json',{'passed':True,'selected':selected,'selection_sha256':frozen_hash,'methods':list(models),'splits':list(x),'completed_utc':datetime.now(timezone.utc).isoformat()})
    log(f'Complete {dataset}/{task}: {selected}')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset');args=parser.parse_args()
    HERE.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):
        for dataset,task in JOBS:
            if args.dataset and dataset!=args.dataset:continue
            task_run(dataset,task)
