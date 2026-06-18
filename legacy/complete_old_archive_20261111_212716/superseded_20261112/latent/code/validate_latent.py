"""Verify source lineage, split integrity, saved encoders and complete CSV handoff."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import numpy as np
import pandas as pd
import joblib
import torch
from threadpoolctl import threadpool_limits
from models import MODELS, encode_all
from run_latent import (ROOT, HERE, OUT, DATASETS, METHODS, SPLITS, LATENT_DIM, SEED,
    load_data, sha, write_json, linear_cka, sample_indices, anchor_groups)
from sklearn.metrics import f1_score, silhouette_score


def validate_dataset(name):
    data=load_data(name)
    folder=OUT/name
    manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['input_sha256']==data['input_sha256']
    normalization=joblib.load(folder/'input_normalization.joblib')
    np.testing.assert_allclose(normalization['mean'],data['X']['train'].mean(0,dtype=float))
    np.testing.assert_allclose(normalization['scale'],np.maximum(data['X']['train'].std(0,dtype=float),1))
    input_frames={s:((x-normalization['mean'])/normalization['scale']).astype(np.float32) for s,x in data['X'].items()}
    keys,patient_sets,metadata={},{},{}
    for split in SPLITS:
        path=folder/f'rows_{split}.csv'
        assert sha(path)==manifest['metadata'][split]['csv_sha256']
        expected=data['rows'][split]
        # All fields, including original weights/design and four MIMIC targets.
        df=pd.read_csv(path,dtype={c:'string' for c in expected if isinstance(expected[c].dtype,pd.StringDtype)})
        pd.testing.assert_frame_equal(df,expected,check_dtype=False,rtol=1e-12,atol=1e-12)
        metadata[split]=df
        idcols={'heart':['meta_domain_state','meta_record_id'],'respiratory':['meta_patientunitstayid'],
                'diabetes':['meta_record_id'],'hypertension':['meta_record_id'],'mimic':['meta_stay_id']}[name]
        key=pd.MultiIndex.from_frame(df[idcols])
        assert key.is_unique
        keys[split]=set(key)
        group={'respiratory':'meta_split_group_uniquepid','mimic':'meta_patient_id'}.get(name)
        if group:
            patient_sets[split]=set(df[group])
        with np.load(folder/f'y_{split}.npz',allow_pickle=False) as archive:
            assert set(archive.files)==set(data['targets'])
            for target in data['targets']:np.testing.assert_array_equal(archive[target],data['y'][split][target])
    for a,b in [('train','test'),('train','validation'),('validation','test')]:
        assert not keys[a]&keys[b]
        if patient_sets:assert not patient_sets[a]&patient_sets[b]
    results={}
    for method in manifest['methods']:
        dest=folder/method
        training=json.loads((dest/'training.json').read_text())
        assert training['test_rows_used_for_fitting_or_selection']==0
        assert not training['labels_used_for_encoder_training']
        if method.startswith('PCA'):
            model=joblib.load(dest/'encoder.joblib')
            projection=None
        else:
            checkpoint=torch.load(dest/'encoder.pt',map_location='cpu',weights_only=True)
            references=checkpoint['state_dict'].get('reference',torch.zeros(1,checkpoint['dimension'])).numpy()
            model=MODELS[method](checkpoint['dimension'],checkpoint['views'],checkpoint['groups'],references)
            model.load_state_dict(checkpoint['state_dict']);model.eval()
            projection=joblib.load(dest/'gcca_projection.joblib')
            assert training['unique_training_rows_seen']==len(data['X']['train'])
            if method=='SAINT':
                np.testing.assert_array_equal(references,input_frames['train'][checkpoint['reference_indices']])
            if method=='DGCCA':assert projection['fit_rows']==len(input_frames['train'])
        scaler=joblib.load(dest/'latent_scaler.joblib')
        mapper=joblib.load(dest/'anchor_projection.joblib')
        assert scaler.n_samples_seen_==len(input_frames['train'])
        arrays={}
        for split in SPLITS:
            path=dest/f'Z_{split}.npz'
            assert sha(path)==manifest['methods'][method]['artifacts'][split]['sha256']
            with np.load(path,allow_pickle=False) as archive:z=archive['Z']
            arrays[split]=z
            dimension=manifest['methods'][method]['artifacts'][split]['dimensions']
            assert z.shape==(len(data['X'][split]),dimension) and np.isfinite(z).all()
            ix=np.random.default_rng(SEED).choice(len(z),min(len(z),257),replace=False)
            query=input_frames[split][ix]
            raw=model.transform(query) if method.startswith('PCA') else encode_all(model,query,projection,batch_size=128)
            np.testing.assert_allclose(scaler.transform(raw),z[ix],atol=2e-4,rtol=2e-4)
            with np.load(dest/f'Z_anchor_{split}.npz',allow_pickle=False) as archive:
                np.testing.assert_allclose(archive['Z'],mapper.predict(z),atol=1e-5,rtol=1e-5)
        metrics=manifest['methods'][method]['metrics']
        for target in data['targets']:
            probe=joblib.load(dest/f'probe_{target}.joblib')
            pred=probe['classifier'].predict_proba(probe['scaler'].transform(arrays['validation']))[:,1]>=.5
            actual=f1_score(data['y']['validation'][target],pred,average='macro',zero_division=0)
            np.testing.assert_allclose(actual,metrics['probe'][target]['macro_f1'],atol=1e-12)
            ix=sample_indices(data['y']['validation'][target],manifest['config']['metric_sample'])
            actual_sil=silhouette_score(arrays['validation'][ix],data['y']['validation'][target][ix])
            np.testing.assert_allclose(actual_sil,metrics['probe'][target]['silhouette'],atol=1e-7)
        ix=np.random.default_rng(SEED).choice(len(arrays['validation']),min(manifest['config']['metric_sample'],len(arrays['validation'])),replace=False)
        np.testing.assert_allclose(linear_cka(data['X']['validation'][ix],arrays['validation'][ix]),metrics['same_row_input_latent_cka'],atol=1e-12)
        results[method]={'passed':True,'finite_aligned_arrays_all_splits':True,'saved_encoder_reproduces_sampled_rows':True,
                         'metric_recomputation_passed':True,'test_scoring_performed':False}
    selected=manifest['selection']['method']
    eligible=[m for m,r in manifest['methods'].items() if r['metrics']['clinical_utility_guard_passed']]
    expected=max(eligible or list(manifest['methods']),key=lambda m:manifest['methods'][m]['metrics']['mean_validation_macro_f1'])
    assert selected==expected
    for split in SPLITS:
        path=ROOT/manifest['selection'][split]['csv']
        assert sha(path)==manifest['selection'][split]['sha256']
        # Float32 predictors are rounded only by normal CSV serialization.
        expected=metadata[split]
        csv=pd.read_csv(path,dtype={c:'string' for c in expected if isinstance(expected[c].dtype,pd.StringDtype)})
        pd.testing.assert_frame_equal(csv[list(expected)],expected,check_dtype=False,rtol=1e-12,atol=1e-12)
        dimension=manifest['methods'][selected]['artifacts'][split]['dimensions']
        with np.load(folder/selected/f'Z_{split}.npz') as archive:
            np.testing.assert_allclose(csv[[f'z{i+1:02}' for i in range(dimension)]],archive['Z'],rtol=1e-7,atol=1e-7)
        with np.load(folder/'selected'/f'Z_{split}.npz') as archive:
            np.testing.assert_allclose(csv[[f'z{i+1:02}' for i in range(dimension)]],archive['Z'],rtol=1e-7,atol=1e-7)
    return {'passed':True,'methods':results,'selected':selected,'all_metadata_and_labels_preserved':True,
            'disjoint_records':True,'disjoint_patients':True if patient_sets else None,'complete_csv_handoff_passed':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=DATASETS);args=parser.parse_args()
    torch.set_num_threads(4)
    result={'passed':False,'checked_utc':datetime.now(timezone.utc).isoformat(),'datasets':{}}
    with threadpool_limits(limits=4):
        for name in ([args.dataset] if args.dataset else DATASETS):
            print('Validating '+name,flush=True)
            result['datasets'][name]=validate_dataset(name)
    result['passed']=True
    path=OUT/(f'validation_{args.dataset}.json' if args.dataset else 'validation.json')
    write_json(path,result)
    print('PASS: '+str(path),flush=True)


if __name__=='__main__':main()
