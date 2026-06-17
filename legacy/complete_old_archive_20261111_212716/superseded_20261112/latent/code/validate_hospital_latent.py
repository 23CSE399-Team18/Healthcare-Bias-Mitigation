"""Independently reproduce hospital preprocessing, encodings and CSV exports."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'latent/outputs/hospital'
SPLITS=('train','validation','test')

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read_rows(path):
    cols=pd.read_csv(path,nrows=0).columns
    return pd.read_csv(path,dtype={c:'string' for c in cols if c.startswith('meta_')})

def respiratory_source():
    base=ROOT/'preprocessing/outputs/hospital_baseline/respiratory'
    record=json.loads((base/'validation.json').read_text())
    path=ROOT/'model_ready/respiratory_model_ready.csv'
    assert sha(path)==record['source_sha256']
    raw=read_rows(path).copy()
    def assign(hospital):
        f=int.from_bytes(hashlib.sha256(('four-condition-v1-hospital:'+hospital).encode()).digest()[:8],'big')/2**64
        return 'train' if f<.7 else 'validation' if f<.85 else 'test'
    raw['meta_split']=raw.meta_domain_hospitalid.map(assign)
    crossing=raw.groupby('meta_split_group_uniquepid').meta_split.nunique()
    excluded=set(crossing[crossing>1].index)
    removed=raw[raw.meta_split_group_uniquepid.isin(excluded)]
    assert len(excluded)==record['excluded_cross_partition_patients'] and len(removed)==record['excluded_stays']
    saved_removed=read_rows(base/'cross_partition_patients_excluded.csv')
    pd.testing.assert_frame_equal(saved_removed,removed[list(saved_removed)].reset_index(drop=True),check_dtype=False)
    raw=raw[~raw.meta_split_group_uniquepid.isin(excluded)]
    prep=joblib.load(base/'preprocessor.joblib')
    features=list(prep.feature_names_in_)
    assert all(c!='target' and not c.startswith(('meta_','sens_')) for c in features)
    nums=list(prep.transformers_[0][2]);cats=list(prep.transformers_[1][2])
    train=raw[raw.meta_split.eq('train')]
    imputer=prep.named_transformers_['numeric']['imputer']
    np.testing.assert_allclose(imputer.statistics_,train[nums].median().fillna(0))
    ref=StandardScaler().fit(imputer.transform(train[nums]))
    scale=prep.named_transformers_['numeric']['scale']
    np.testing.assert_allclose(scale.mean_,ref.mean_)
    np.testing.assert_allclose(scale.scale_,ref.scale_)
    assert scale.n_samples_seen_==len(train)
    cpipe=prep.named_transformers_['categorical']
    expected_cats=cpipe['imputer'].transform(train[cats].astype(object).where(train[cats].notna(),np.nan))
    for i,levels in enumerate(cpipe['encode'].categories_):np.testing.assert_array_equal(levels,np.unique(expected_cats[:,i]))
    names=prep.get_feature_names_out().tolist()
    assert names==json.loads((base/'feature_names.json').read_text())['features']
    for split in SPLITS:
        d=raw[raw.meta_split.eq(split)].reset_index(drop=True)
        data=d[features].copy()
        for c in cats:data[c]=data[c].astype(object).where(data[c].notna(),np.nan)
        expected=prep.transform(data).astype(np.float32)
        with np.load(base/f'X_{split}.npz') as a:np.testing.assert_allclose(a['X'],expected,atol=1e-6,rtol=1e-6)
        rows=read_rows(base/f'rows_{split}.csv')
        pd.testing.assert_frame_equal(rows,d[list(rows)],check_dtype=False)
        csv=read_rows(base/f'respiratory_{split}_preprocessed.csv')
        np.testing.assert_allclose(csv[names],expected,atol=1e-6,rtol=1e-6)
        pd.testing.assert_frame_equal(csv[list(rows)],rows,check_dtype=False)
    return {'passed':True,'original_rows_accounted_for':len(raw)+len(removed),'excluded_stays':len(removed),'train_only_statistics_verified':True,'all_original_rows_and_transforms_reproduced':True}

def check(name):
    folder=OUT/name;manifest=json.loads((folder/'manifest.json').read_text());source=ROOT/manifest['source']
    normal=joblib.load(folder/'input_normalization.joblib');encoder=joblib.load(folder/'encoder.joblib');scale=joblib.load(folder/'latent_scaler.joblib')
    with np.load(source/'X_train.npz') as a:train=a['X'].astype(np.float32)
    np.testing.assert_allclose(normal['mean'],train.mean(0,dtype=float),atol=1e-6,rtol=1e-6)
    np.testing.assert_allclose(normal['scale'],np.maximum(train.std(0,dtype=float),1.),atol=1e-6,rtol=1e-6)
    normalized=((train-normal['mean'])/normal['scale']).astype(np.float32)
    assert encoder.n_samples_==scale.n_samples_seen_==len(train)
    np.testing.assert_allclose(encoder.mean_,normalized.mean(0),atol=1e-6,rtol=1e-6)
    independent=StandardScaler().fit(encoder.transform(normalized))
    np.testing.assert_allclose(scale.mean_,independent.mean_,atol=1e-6,rtol=1e-6)
    np.testing.assert_allclose(scale.scale_,independent.scale_,atol=1e-6,rtol=1e-6)
    rows={};result={}
    for split,part in manifest['partitions'].items():
        for f,expected in part['hashes'].items():assert sha(folder/f)==expected
        with np.load(source/f'X_{split}.npz') as a:x=a['X'].astype(np.float32)
        z=scale.transform(encoder.transform(((x-normal['mean'])/normal['scale']).astype(np.float32))).astype(np.float32)
        assert np.isfinite(z).all()
        with np.load(folder/f'Z_{split}.npz') as a:saved_z=a['Z']
        # BLAS thread scheduling changes float32 PCA accumulation slightly.
        np.testing.assert_allclose(saved_z,z,atol=5e-5,rtol=1e-5)
        max_error=float(np.max(np.abs(saved_z-z)))
        rows[split]=read_rows(folder/f'rows_{split}.csv')
        pd.testing.assert_frame_equal(rows[split],read_rows(source/f'rows_{split}.csv'),check_dtype=False)
        csv=read_rows(folder/f'{name}_{split}_latent.csv')
        np.testing.assert_allclose(csv[[f'z{i+1:02}' for i in range(z.shape[1])]],saved_z,atol=1e-6,rtol=1e-6)
        pd.testing.assert_frame_equal(csv[list(rows[split])],rows[split],check_dtype=False)
        with np.load(folder/f'y_{split}.npz') as labels,np.load(source/f'y_{split}.npz') as expected:
            assert set(labels.files)==set(expected.files)==set(manifest['targets'])
            for task in labels.files:
                np.testing.assert_array_equal(labels[task],expected[task])
                np.testing.assert_array_equal(labels[task],csv['target' if name=='respiratory_hospital' else 'target_'+task])
        result[split]={'rows':len(z),'latent_dimensions':z.shape[1],'hospitals':int(rows[split].meta_domain_hospitalid.nunique()),'csv_and_source_transform_verified':True,'max_float32_reproduction_error':max_error}
    patient='meta_split_group_uniquepid' if name=='respiratory_hospital' else 'meta_patient_id'
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not set(rows[a][patient])&set(rows[b][patient])
        if name!='mimic_transfer':assert not set(rows[a].meta_domain_hospitalid)&set(rows[b].meta_domain_hospitalid)
    return {'passed':True,'train_only_statistics_verified':True,'patient_disjoint':True,'hospital_disjoint':name!='mimic_transfer','partitions':result}

if __name__=='__main__':
    report={'passed':False,'branches':{}}
    try:
        with threadpool_limits(limits=2):
            report['respiratory_preprocessing']=respiratory_source()
            for name in ('respiratory_hospital','eicu_hospital','mimic_transfer'):
                print('Reproduce '+name,flush=True);report['branches'][name]=check(name)
        report['passed']=True
    finally:(OUT/'independent_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Hospital preprocessing and latent verification passed',flush=True)
