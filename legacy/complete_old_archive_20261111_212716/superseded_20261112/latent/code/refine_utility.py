"""Validation-only rescue when a fixed eight-dimensional bottleneck loses utility.

Retain the original comparisons; evaluate training-variance PCA at 95%, 99%, then
99.9% until the predeclared 0.02 validation Macro-F1 guard passes. Never use test
outcomes to choose dimension. Report the actual dimension instead of pretending
all data need the same width.
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from run_latent import (ROOT, HERE, OUT, DATASETS, SPLITS, load_data, evaluate_method,
                       fit_anchor_map, write_json, sha, log)


def refine(name):
    folder=OUT/name
    manifest=json.loads((folder/'manifest.json').read_text())
    initial=manifest['selection']['method']
    # An average over MIMIC tasks must not hide loss on an individual diagnosis.
    for method, record in manifest['methods'].items():
        metrics=record['metrics']
        changes={target:result['macro_f1']-manifest['uncompressed_probe'][target]['macro_f1']
                 for target,result in metrics['probe'].items()}
        metrics['per_target_macro_f1_change_vs_uncompressed']=changes
        metrics['clinical_utility_guard_passed']=min(changes.values())>=-.02
        write_json(folder/method/'metrics.json',metrics)
    manifest['selection']['clinical_utility_guard_passed']=manifest['methods'][initial]['metrics']['clinical_utility_guard_passed']
    if manifest['selection']['clinical_utility_guard_passed']:
        manifest['selection']['latent_dimensions']=manifest['methods'][initial]['artifacts']['train']['dimensions']
        write_json(folder/'manifest.json',manifest)
        return manifest
    data=load_data(name)
    norm=joblib.load(folder/'input_normalization.joblib')
    inputs={s:((x-norm['mean'])/norm['scale']).astype(np.float32) for s,x in data['X'].items()}
    attempts=[]
    for retained in (.95,.99,.999):
        method='PCA_'+str(retained*100).rstrip('0').rstrip('.')+'pct_variance'
        dest=folder/method;dest.mkdir(exist_ok=True)
        log(f'{name}: utility guard triggered; trying train-variance PCA {retained:.1%}')
        model=PCA(n_components=retained,svd_solver='full').fit(inputs['train'])
        raw={s:model.transform(x) for s,x in inputs.items()}
        scaler=StandardScaler().fit(raw['train'])
        z={s:scaler.transform(a).astype(np.float32) for s,a in raw.items()}
        dimension=z['train'].shape[1]
        joblib.dump(model,dest/'encoder.joblib');joblib.dump(scaler,dest/'latent_scaler.joblib')
        write_json(dest/'training.json',{'fit_rows':len(inputs['train']),'labels_used_for_encoder_training':False,
            'test_rows_used_for_fitting_or_selection':0,'explained_variance_ratio':model.explained_variance_ratio_,
            'requested_training_variance_retained':retained,'actual_dimensions':dimension})
        mapper,info=fit_anchor_map(z['train'],data['rows']['train'])
        joblib.dump(mapper,dest/'anchor_projection.joblib');write_json(dest/'anchor_info.json',info)
        artifacts={}
        for split in SPLITS:
            path=dest/f'Z_{split}.npz';np.savez_compressed(path,Z=z[split])
            np.savez_compressed(dest/f'Z_anchor_{split}.npz',Z=mapper.predict(z[split]).astype(np.float32))
            artifacts[split]={'rows':len(z[split]),'dimensions':dimension,'sha256':sha(path),
                'metadata_csv_sha256':manifest['metadata'][split]['csv_sha256']}
        metrics=evaluate_method(data,method,z,dest)
        metrics['validation_macro_f1_change_vs_uncompressed']=metrics['mean_validation_macro_f1']-manifest['uncompressed_mean_macro_f1']
        changes={target:result['macro_f1']-manifest['uncompressed_probe'][target]['macro_f1']
                 for target,result in metrics['probe'].items()}
        metrics['per_target_macro_f1_change_vs_uncompressed']=changes
        metrics['clinical_utility_guard_passed']=min(changes.values())>=-.02
        write_json(dest/'metrics.json',metrics);write_json(dest/'artifacts.json',artifacts)
        manifest['methods'][method]={'metrics':metrics,'artifacts':artifacts}
        attempts.append({'method':method,'dimensions':dimension,'guard_passed':metrics['clinical_utility_guard_passed']})
        log(f"{name} {method}: {dimension} features, validation macro-F1 {metrics['mean_validation_macro_f1']:.4f}, change {metrics['validation_macro_f1_change_vs_uncompressed']:+.4f}")
        if metrics['clinical_utility_guard_passed']:break
    eligible=[m for m,r in manifest['methods'].items() if r['metrics']['clinical_utility_guard_passed']]
    selected=max(eligible or list(manifest['methods']),key=lambda m:manifest['methods'][m]['metrics']['mean_validation_macro_f1'])
    result=manifest['methods'][selected]
    dimension=result['artifacts']['train']['dimensions']
    passed=result['metrics']['clinical_utility_guard_passed']
    manifest['selection']={'method':selected,'latent_dimensions':dimension,
        'rule':'Best validation Macro-F1 among candidates passing the per-target utility guard, after variance-retention rescue if required',
        'status':'ready' if passed else 'ready_with_utility_warning','clinical_utility_guard_passed':passed,
        'note':'Variable latent width is intentional; all initial eight-dimensional candidates remain available.'}
    manifest['dimension_refinement']={'reason':'initial best candidate lost >0.02 validation Macro-F1 on at least one target versus uncompressed',
        'initial_selected':initial,'attempts':attempts,'test_scoring_used':False,'script_sha256':sha(__file__)}
    for split in SPLITS:
        with np.load(folder/selected/f'Z_{split}.npz') as a:values=a['Z']
        path=folder/'selected'/f'{name}_{split}_latent.csv'
        frame=pd.concat([pd.DataFrame(values,columns=[f'z{i+1:02}' for i in range(dimension)]),data['rows'][split]],axis=1)
        frame.to_csv(path,index=False)
        np.savez_compressed(folder/'selected'/f'Z_{split}.npz',Z=values)
        columns=[f'z{i+1:02}' for i in range(dimension)]
        np.testing.assert_allclose(pd.read_csv(path,usecols=columns),values,atol=1e-7,rtol=1e-7)
        manifest['selection'][split]={'csv':str(path.relative_to(ROOT)),'sha256':sha(path),'rows':len(frame)}
    write_json(folder/'manifest.json',manifest)
    return manifest


if __name__=='__main__':
    with threadpool_limits(limits=4):
        for name in DATASETS:refine(name)
