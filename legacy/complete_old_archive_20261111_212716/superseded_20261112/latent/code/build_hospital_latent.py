"""Prepare genuinely hospital-disjoint ICU branches and frozen source encodings."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'latent/outputs/hospital'
SPLITS = ('train','validation','test')
TASKS = ('heart','respiratory','diabetes','hypertension')

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def write(path, obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2),encoding='utf-8')

def hospital_split(value):
    fraction=int.from_bytes(hashlib.sha256(f'four-condition-v1-hospital:{value}'.encode()).digest()[:8],'big')/2**64
    return 'train' if fraction<.70 else ('validation' if fraction<.85 else 'test')

def respiratory():
    source=ROOT/'model_ready/respiratory_model_ready.csv'
    raw=pd.read_csv(source,dtype={'meta_patientunitstayid':'string','meta_split_group_uniquepid':'string','meta_domain_hospitalid':'string'})
    assert raw.meta_domain_hospitalid.notna().all()
    raw['meta_split']=raw.meta_domain_hospitalid.map(hospital_split)
    crossing=raw.groupby('meta_split_group_uniquepid').meta_split.nunique()
    remove=set(crossing[crossing>1].index)
    removed=raw[raw.meta_split_group_uniquepid.isin(remove)]
    raw=raw[~raw.meta_split_group_uniquepid.isin(remove)].copy()
    features=[c for c in raw if c!='target' and not c.startswith(('meta_','sens_'))]
    nums=[c for c in features if pd.api.types.is_numeric_dtype(raw[c])]
    cats=[c for c in features if c not in nums]
    parts={s:raw[raw.meta_split.eq(s)].reset_index(drop=True) for s in SPLITS}
    def inputs(d):
        d=d[features].copy()
        for c in cats:d[c]=d[c].astype(object).where(d[c].notna(),np.nan)
        return d
    prep=ColumnTransformer([
        ('numeric',Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler())]),nums),
        ('categorical',Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value='__MISSING__',keep_empty_features=True)),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cats)])
    prep.fit(inputs(parts['train']))
    base=ROOT/'preprocessing/outputs/hospital_baseline/respiratory'
    base.mkdir(parents=True,exist_ok=True)
    joblib.dump(prep,base/'preprocessor.joblib')
    names=prep.get_feature_names_out().tolist()
    np.testing.assert_allclose(prep.named_transformers_['numeric']['imputer'].statistics_,parts['train'][nums].median().fillna(0))
    xs,ys,rows={},{},{}
    for s,d in parts.items():
        xs[s]=prep.transform(inputs(d)).astype(np.float32)
        ys[s]={'respiratory':d.target.to_numpy(np.int8)}
        rows[s]=d[[c for c in d if c=='target' or c.startswith(('meta_','sens_'))]].copy()
        np.savez_compressed(base/f'X_{s}.npz',X=xs[s]);np.savez_compressed(base/f'y_{s}.npz',**ys[s])
        rows[s].to_csv(base/f'rows_{s}.csv',index=False)
        exported=pd.concat([pd.DataFrame(xs[s],columns=names),rows[s]],axis=1)
        exported.to_csv(base/f'respiratory_{s}_preprocessed.csv',index=False)
        np.testing.assert_allclose(pd.read_csv(base/f'respiratory_{s}_preprocessed.csv',usecols=names),xs[s],atol=1e-6,rtol=1e-6)
        assert np.isfinite(xs[s]).all()
    removed[['meta_patientunitstayid','meta_split_group_uniquepid','meta_domain_hospitalid','meta_split']].to_csv(base/'cross_partition_patients_excluded.csv',index=False)
    write(base/'feature_names.json',{'features':names})
    write(base/'validation.json',{'passed':True,'source_sha256':sha(source),'excluded_cross_partition_patients':len(remove),'excluded_stays':len(removed),'fit_partition':'train','features':len(names),'partitions':{s:{'rows':len(d),'hospitals':int(d.meta_domain_hospitalid.nunique())} for s,d in rows.items()}})
    return xs,ys,rows,base

def source_direction(direction):
    base=ROOT/'preprocessing/outputs/cross_dataset_baseline'/direction
    record=json.loads((base/'transformation_validation.json').read_text())
    xs,ys,rows={},{},{}
    for s in (*SPLITS,'external'):
        for name,expected in record['partitions'][s]['sha256'].items():assert sha(base/name)==expected
        with np.load(base/f'X_{s}.npz') as a:xs[s]=a['X'].astype(np.float32)
        with np.load(base/f'y_{s}.npz') as a:ys[s]={t:a[t] for t in a.files}
        cols=pd.read_csv(base/f'rows_{s}.csv',nrows=0).columns
        rows[s]=pd.read_csv(base/f'rows_{s}.csv',dtype={c:'string' for c in cols if c.startswith('meta_')})
    return xs,ys,rows,base

def probes(xs,ys):
    scaler=StandardScaler().fit(xs['train'])
    scores={}
    for task in ys['train']:
        model=LogisticRegression(C=1,class_weight='balanced',max_iter=1000).fit(scaler.transform(xs['train']),ys['train'][task])
        scores[task]=float(f1_score(ys['validation'][task],model.predict(scaler.transform(xs['validation'])),average='macro'))
    return scores

def build(name,data,reuse=False):
    xs,ys,rows,source=data
    out=OUT/name;out.mkdir(parents=True,exist_ok=True)
    patient='meta_split_group_uniquepid' if name=='respiratory_hospital' else 'meta_patient_id'
    hospitals={s:set(rows[s].meta_domain_hospitalid.astype(str)) for s in SPLITS}
    patients={s:set(rows[s][patient].astype(str)) for s in SPLITS}
    for i,a in enumerate(SPLITS):
        for b in SPLITS[i+1:]:
            assert not patients[a]&patients[b]
            if name!='mimic_transfer':assert not hospitals[a]&hospitals[b]
    if reuse:
        previous=ROOT/'latent/outputs/mimic'
        manifest=json.loads((previous/'manifest.json').read_text()); method=manifest['selection']['method']
        normal=joblib.load(previous/'input_normalization.joblib')
        encoder=joblib.load(previous/method/'encoder.joblib'); scaler=joblib.load(previous/method/'latent_scaler.joblib')
        norm={s:((x-normal['mean'])/normal['scale']).astype(np.float32) for s,x in xs.items()}
        zs={s:scaler.transform(encoder.transform(x)).astype(np.float32) for s,x in norm.items()}
        for s in SPLITS:
            with np.load(previous/'selected'/f'Z_{s}.npz') as a:np.testing.assert_allclose(zs[s],a['Z'],atol=1e-5,rtol=1e-5)
        reference=manifest['uncompressed_probe']; attempts=[]
    else:
        normal={'mean':xs['train'].mean(0,dtype=float),'scale':np.maximum(xs['train'].std(0,dtype=float),1.)}
        norm={s:((x-normal['mean'])/normal['scale']).astype(np.float32) for s,x in xs.items()}
        reference=probes(xs,ys);attempts=[]
        for variance in (.999,.9999,None):
            encoder=PCA(n_components=variance,svd_solver='full').fit(norm['train'])
            raw={s:encoder.transform(x) for s,x in norm.items()}
            scaler=StandardScaler().fit(raw['train'])
            zs={s:scaler.transform(x).astype(np.float32) for s,x in raw.items()}
            scores=probes(zs,ys);passed=all(scores[t]>=reference[t]-.02 for t in scores)
            attempts.append({'variance':variance,'dimensions':zs['train'].shape[1],'validation_macro_f1':scores,'guard_passed':passed})
            if passed:break
        method='PCA_source_training';assert passed
    joblib.dump(normal,out/'input_normalization.joblib');joblib.dump(encoder,out/'encoder.joblib');joblib.dump(scaler,out/'latent_scaler.joblib')
    parts={}
    for s,z in zs.items():
        np.savez_compressed(out/f'Z_{s}.npz',Z=z);np.savez_compressed(out/f'y_{s}.npz',**ys[s])
        rows[s].to_csv(out/f'rows_{s}.csv',index=False)
        joined=pd.concat([pd.DataFrame(z,columns=[f'z{i+1:02}' for i in range(z.shape[1])]),rows[s]],axis=1)
        joined.to_csv(out/f'{name}_{s}_latent.csv',index=False)
        parts[s]={'rows':len(z),'dimensions':z.shape[1],'hospitals':int(rows[s].meta_domain_hospitalid.nunique()),'hashes':{f:sha(out/f) for f in (f'Z_{s}.npz',f'y_{s}.npz',f'rows_{s}.csv',f'{name}_{s}_latent.csv')}}
    record={'passed':True,'name':name,'source':str(source.relative_to(ROOT)),'method':method,'targets':list(ys['train']),'partitions':parts,'hospital_disjoint_internal':name!='mimic_transfer','patient_disjoint_internal':True,'validation_refinement':attempts,'test_outcomes_used_for_selection':False,'external_outcomes_used_for_selection':False,'hospital_ids':{s:sorted(v) for s,v in hospitals.items()},'test_predictions_scored':False}
    write(out/'manifest.json',record)
    print(name,json.dumps({s:(p['rows'],p['hospitals'],p['dimensions']) for s,p in parts.items()}),flush=True)
    return record

if __name__=='__main__':
    with threadpool_limits(limits=4):
        results={}
        results['respiratory_hospital']=build('respiratory_hospital',respiratory())
        results['eicu_hospital']=build('eicu_hospital',source_direction('eicu_to_mimic'))
        results['mimic_transfer']=build('mimic_transfer',source_direction('mimic_to_eicu'),reuse=True)
        write(OUT/'manifest.json',{'passed':True,'branches':results})
