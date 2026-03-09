"""Inline experiment engine; this text is embedded in the notebook, not imported."""
import hashlib, json, os, pickle, time, warnings
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.stats import wasserstein_distance, skew, kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import joblib

CONDITIONS = ['heart', 'respiratory', 'diabetes', 'hypertension']
CORE = [f'{v}_{s}' for v in ['heart_rate','resp_rate','spo2'] for s in ['mean','min','max']]
VITAL_NAMES = ['heart_rate','resp_rate','spo2','temperature','nibp_systolic','nibp_diastolic','nibp_mean',
               'arterial_systolic','arterial_diastolic','arterial_mean']
NUMERIC = [f'{v}_{s}' for v in VITAL_NAMES for s in ['mean','min','max']]
CATEGORICAL = ['age_group','sex']
GROUP_COLUMNS = ['sens_sex','sens_race_ethnicity','sens_age_group']
PHYS_BOUNDS = {'heart_rate': (0,300), 'resp_rate': (0,70), 'spo2': (0,100), 'temperature': (10,50),
               'nibp_systolic': (0,400), 'nibp_diastolic': (0,300), 'nibp_mean': (0,300),
               'arterial_systolic': (0,400), 'arterial_diastolic': (0,300), 'arterial_mean': (0,300)}


def file_sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def jsonable(value):
    if isinstance(value, dict): return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [jsonable(v) for v in value]
    if isinstance(value,np.ndarray): return jsonable(value.tolist())
    if isinstance(value,np.generic): return jsonable(value.item())
    if isinstance(value,float) and not np.isfinite(value): return None
    if isinstance(value,Path): return str(value)
    return value


def save_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(jsonable(value),indent=2,allow_nan=False),encoding='utf-8')


def progress(message):
    stamp=time.strftime('%Y-%m-%d %H:%M:%S')
    with (OUT/'run_progress.log').open('a',encoding='utf-8') as handle: handle.write(f'[{stamp}] {message}\n')
    print(message,flush=True)


def load_mimic():
    base=PROJECT/'cross_dataset/four_conditions/mimic'
    verified=json.loads((base/'preprocessing_summary.json').read_text())
    frames={}
    for condition in CONDITIONS:
        path=base/f'{condition}_model_ready.csv'
        assert file_sha(path)==verified['conditions'][condition]['sha256'], 'Changed source table'
        frames[condition]=pd.read_csv(path)
    first=frames['heart']
    reference=first.drop(columns='target')
    for condition,frame in frames.items():
        pd.testing.assert_frame_equal(reference,frame.drop(columns='target'))
        assert frame.target.isin([0,1]).all()
    clinical_path=base/'clinical_24h_features.csv'
    clinical_audit=json.loads((base/'clinical_preprocessing_validation.json').read_text())
    assert clinical_audit['passed'] and clinical_audit['chartevents_gzip_crc_verified_by_full_stream']
    assert file_sha(clinical_path)==clinical_audit['output_sha256'][clinical_path.name]
    clinical=pd.read_csv(clinical_path).set_index('stay_id')
    data=first.copy()
    data[NUMERIC]=clinical.loc[data.meta_stay_id,NUMERIC].to_numpy()
    for condition in CONDITIONS: data[f'y_{condition}']=frames[condition].target.to_numpy()
    data=data.drop(columns='target')
    assert data.meta_stay_id.is_unique
    assert data.groupby('meta_patient_id').meta_split.nunique().max()==1
    assert data.meta_domain_hospitalid.nunique()==1
    assert not np.isinf(data[NUMERIC].to_numpy()).any()
    parts={name:data.loc[data.meta_split.eq(name)].reset_index(drop=True) for name in ['train','validation','test']}
    # Test rows are loaded only for integrity/shape; no numerical test predictor or
    # test outcome is used in any comparison or choice until evaluate_final().
    return data,parts,{'input_sha256':{f'{c}_model_ready.csv':verified['conditions'][c]['sha256'] for c in CONDITIONS},
        'clinical_sha256':file_sha(clinical_path),'rows':len(data),'patients':data.meta_patient_id.nunique(),
        'split_rows':{k:len(v) for k,v in parts.items()},'features':NUMERIC+CATEGORICAL,
        'test_selection_rows':0,'external_eicu_files_opened':False}


def groups(frame): return frame[GROUP_COLUMNS].reset_index(drop=True)


def constrain_missing(raw,filled):
    """Apply physiological bounds/order to MISSING estimates only; never alter observations."""
    raw=np.asarray(raw,float); out=np.asarray(filled,float).copy(); missing=np.isnan(raw)
    for vital in VITAL_NAMES:
        indices=[NUMERIC.index(f'{vital}_{s}') for s in ['mean','min','max']]
        mean,low,high=indices; lower,upper=PHYS_BOUNDS[vital]
        for j in indices:
            use=missing[:,j];out[use,j]=np.clip(out[use,j],lower+1e-6,upper-1e-6)
        # Observed summaries from the same hourly source already obey ordering.
        use=missing[:,low];out[use,low]=np.minimum(out[use,low],out[use,mean])
        use=missing[:,high];out[use,high]=np.maximum(out[use,high],out[use,mean])
        use=missing[:,mean];out[use,mean]=np.clip(out[use,mean],out[use,low],out[use,high])
    assert np.array_equal(out[~missing],raw[~missing]), 'Observed measurement changed'
    assert np.isfinite(out).all()
    return out


def fit_probe(X,y,weights=None):
    model=LogisticRegression(C=1.,solver='lbfgs',max_iter=600,tol=1e-4,random_state=SEED)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        with threadpool_limits(limits=2): model.fit(X,y,sample_weight=weights)
    model.comparison_converged_=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
    return model


def encode_matrix(numeric,frame,encoder,scaler=None,mask=None):
    N=np.asarray(numeric,float)
    if scaler is not None: N=scaler.transform(N)
    C=encoder.transform(frame[CATEGORICAL].astype(str))
    if mask is None: mask=frame[NUMERIC].isna().to_numpy(dtype=float)
    return np.column_stack([N,C,np.asarray(mask,float)])


def probe_conditions(train,val,Ntrain,Nval,encoder,scaler,weights=None):
    Xt=encode_matrix(Ntrain,train,encoder,scaler); Xv=encode_matrix(Nval,val,encoder,scaler)
    rows=[]
    for condition in CONDITIONS:
        model=fit_probe(Xt,train[f'y_{condition}'].to_numpy(),weights)
        p=model.predict_proba(Xv)[:,1]
        result=classification_metrics(val[f'y_{condition}'].to_numpy(),p,groups(val))
        result.update(condition=condition,converged=model.comparison_converged_)
        rows.append(result)
    return pd.DataFrame(rows)


def utility_guard(row,baseline,auc_loss=.01,ba_loss=.015,brier_increase=.02,eo_increase=.03):
    needed=['auroc','balanced_accuracy','brier']
    if not all(np.isfinite(row.get(k,np.nan)) and np.isfinite(baseline.get(k,np.nan)) for k in needed): return False
    valid=(row['auroc']>=baseline['auroc']-auc_loss and row['balanced_accuracy']>=baseline['balanced_accuracy']-ba_loss
           and row['brier']<=baseline['brier']+brier_increase and bool(row.get('converged',True)))
    old,new=baseline.get('max_equalized_odds_gap',np.nan),row.get('max_equalized_odds_gap',np.nan)
    if np.isfinite(old) and np.isfinite(new): valid=valid and new<=old+eo_increase
    return bool(valid)


def group_gap_vectors(y,p,frame):
    tab=subgroup_metrics(y,p,groups(frame))
    selected=tab.loc[tab.attribute.ne('age') & tab.joint_eligible]
    return pd.Series(selected.tpr.to_numpy(),index=selected.attribute.astype(str)+' / '+selected.group.astype(str))


def run_duplicate_audit(train):
    duplicate_keys=train.meta_stay_id.duplicated(keep='first')
    payload=train[NUMERIC+CATEGORICAL].duplicated(keep='first')
    # Unique verified stay keys are identity evidence. Equal physiology alone is not.
    status='detected' if duplicate_keys.any() else 'not_detected'
    metrics=pd.Series({'Duplicate stay IDs':float(duplicate_keys.sum()),'Identical payload excess':float(payload.sum())})
    comparison=pd.DataFrame([{'method':'Verified stay-key audit','family':'established','status':'measured',
        'duplicate_rate':duplicate_keys.mean(),'redundancy_reduction':0.,'pair_precision':np.nan,'pair_recall':np.nan},
        {'method':'LSH candidate retrieval','family':'candidate_retrieval','status':'not_applied_no_verified_duplicates',
        'duplicate_rate':duplicate_keys.mean(),'redundancy_reduction':0.,'pair_precision':np.nan,'pair_recall':np.nan}])
    comparison.to_csv(OUT/'comparison_duplicate.csv',index=False)
    return {'status':status,'selected':'No change','before':metrics,'after':metrics.copy(),'comparison':comparison,
            'note':'Equal feature tuples across different stay IDs are not verified duplicates. Valid repeated patient stays remain.'}


def run_missing_comparison(train,val,encoder):
    progress('Missing-data comparison: fitting five frozen inductive imputers.')
    rawt=train[NUMERIC].to_numpy(float); rawv=val[NUMERIC].to_numpy(float)
    rng=np.random.default_rng(SEED)
    fitrows=np.sort(rng.choice(len(train),min(IMPUTER_FIT_CAP,len(train)),replace=False))
    benchrows=np.sort(rng.choice(len(val),min(MASK_BENCHMARK_ROWS,len(val)),replace=False))
    truth=rawv[benchrows].copy(); mask=np.isfinite(truth)&(rng.random(truth.shape)<.10)
    masked=truth.copy();masked[mask]=np.nan
    scale=np.nanstd(rawt,axis=0);scale[~np.isfinite(scale)|(scale<1e-8)]=1.
    candidates=['median','knn','iterative_bayes','iterative_extra_trees','adaptive_inductive']
    tables=[];imputers={};cache={};utilities={}
    for method in candidates:
        started=time.monotonic()
        try:
            options={'fit_cap':IMPUTER_FIT_CAP}
            if method not in ['adaptive_inductive']: options.update(max_iter=3,n_estimators=16)
            imputer=make_imputer(method,seed=SEED,**options).fit(rawt[fitrows])
            nt=constrain_missing(rawt,imputer.transform(rawt));nv=constrain_missing(rawv,imputer.transform(rawv))
            reconstructed=constrain_missing(masked,imputer.transform(masked))
            metrics=reconstruction_metrics(truth,reconstructed,mask,scale)
            # Paired physical-unit standardized distributions; no invented truth at natural missing cells.
            w=[]
            for j in range(len(NUMERIC)):
                take=mask[:,j]&np.isfinite(truth[:,j])
                if take.any(): w.append(wasserstein_distance(truth[take,j]/scale[j],reconstructed[take,j]/scale[j]))
            metrics['wasserstein_normalized']=np.mean(w)
            reference=np.where(np.isnan(truth),np.nanmedian(rawt,axis=0),truth)
            repaired=reference.copy();repaired[mask]=reconstructed[mask]
            active=np.std(reference,axis=0)>1e-8
            metrics['correlation_change']=np.nanmean(np.abs(np.corrcoef(reference[:,active],rowvar=False)-np.corrcoef(repaired[:,active],rowvar=False)))
            probe_scaler=StandardScaler().fit(nt)
            utility=probe_conditions(train,val,nt,nv,encoder,probe_scaler)
            utility['method']=method
            utilities[method]=utility.set_index('condition')
            utility.to_csv(OUT/f'imputation_validation_{method}.csv',index=False)
            row={'method':method,'family':METHOD_METADATA[method]['family'],'status':'measured',
                 'seconds':time.monotonic()-started,**metrics,**utility.select_dtypes('number').mean().to_dict(),
                 'converged':utility.converged.all(),'fit_rows':len(fitrows),'crps':np.nan,'nll':np.nan}
            tables.append(row);imputers[method]=imputer;cache[method]=(nt,nv)
            save_json(OUT/f'imputer_fit_{method}.json',imputer.diagnostics_)
            progress(f'  {method}: masked NRMSE={metrics.get("normalized_rmse",np.nan):.4f}')
        except Exception as error:
            tables.append({'method':method,'family':METHOD_METADATA[method]['family'],'status':'failed',
                           'error':f'{type(error).__name__}: {error}'})
    comparison=pd.DataFrame(tables).set_index('method',drop=False)
    assert 'median' in cache,'Baseline failed; stop rather than invent a result.'
    baseline=comparison.loc['median']
    comparison['eligible']=comparison.apply(lambda row:row.status=='measured' and utility_guard(row,baseline),axis=1)
    for method in utilities:
        comparison.loc[method,'eligible']=all(utility_guard(utilities[method].loc[c],utilities['median'].loc[c]) for c in CONDITIONS)
    objective=['normalized_rmse','normalized_mae','wasserstein_normalized','correlation_change']
    valid=comparison.loc[comparison.eligible].copy()
    assert valid[objective].notna().all().all()
    comparison['mean_quality_rank']=np.nan
    comparison.loc[valid.index,'mean_quality_rank']=valid[objective].rank(method='min').mean(axis=1)
    selected=comparison.loc[comparison.eligible].sort_values(['mean_quality_rank','normalized_rmse','seconds']).iloc[0]['method']
    comparison['selected']=comparison.index==selected
    comparison.to_csv(OUT/'comparison_missing_data.csv',index=False)
    nt,nv=cache[selected]
    # Original missingness patterns remain separate model inputs for every method.
    missing_before=train[NUMERIC].isna().mean()*100
    missing_after=pd.Series(np.isnan(nt).mean(axis=0)*100,index=NUMERIC)
    result={'status':'detected' if missing_before.max()>0 else 'not_detected','selected':selected,
       'before':missing_before,'after':missing_after,'comparison':comparison.reset_index(drop=True),
       'note':'Percent missing per numeric feature. Reconstruction scores use deliberately masked observed validation cells; natural missing truth is unknown.'}
    del cache
    return imputers[selected],nt,nv,result,comparison


def run_scale_comparison(train,val,nt,nv,encoder):
    progress('Feature-scale comparison: five genuine fitted statistical transforms.')
    rows=[];fitted={}
    base_sd=np.std(nt,axis=0);positive=base_sd[base_sd>1e-10]
    detected=len(positive)>1 and positive.max()/positive.min()>10
    for name in ['standard','robust','minmax','quantile_normal','yeo_johnson']:
        try:
            scaler=make_scaler(name,SEED).fit(nt)
            diagnostics=scaling_diagnostics(nt,scaler.transform(nt))
            metrics=probe_conditions(train,val,nt,nv,encoder,scaler)
            rows.append({'method':name,'family':METHOD_METADATA[name]['family'],'status':'measured',**diagnostics,
                         **metrics.select_dtypes('number').mean().to_dict(),'converged':metrics.converged.all()})
            fitted[name]=scaler
        except Exception as error: rows.append({'method':name,'status':'failed','error':str(error)})
    comparison=pd.DataFrame(rows).set_index('method',drop=False)
    base=comparison.loc['standard']
    comparison['eligible']=comparison.apply(lambda row:row.status=='measured' and utility_guard(row,base),axis=1)
    selected='standard'
    if detected:
        eligible=comparison.loc[comparison.eligible]
        selected=eligible.sort_values(['balanced_accuracy','auroc'],ascending=False).iloc[0]['method']
    comparison['selected']=comparison.index==selected
    comparison.to_csv(OUT/'comparison_feature_scale.csv',index=False)
    after=np.std(fitted[selected].transform(nt),axis=0) if detected else base_sd.copy()
    stage={'status':'detected' if detected else 'not_detected','selected':selected if detected else 'No change',
      'before':pd.Series(base_sd,index=NUMERIC),'after':pd.Series(after,index=NUMERIC),
      'comparison':comparison.reset_index(drop=True),
      'note':'Numeric feature SDs in the represented coordinate systems. Scaling changes units; it is not proof of fairness. Yeo-Johnson dates to 2000.'}
    return fitted[selected],stage


def make_state(train,val,nt,nv,encoder,scaler,condition):
    X=encode_matrix(nt,train,encoder,scaler);V=encode_matrix(nv,val,encoder,scaler)
    return {'X':X,'V':V,'y':train[f'y_{condition}'].to_numpy(int),'groups':groups(train),
      'meta':train[[c for c in train if c.startswith('meta_')]].copy(),
      'weights':np.ones(len(train)), 'condition':condition,'projection':None,'original_rows':np.arange(len(train)),
      'numeric':nt,'cats':train[CATEGORICAL].copy(),'missing':train[NUMERIC].isna().to_numpy(float),
      'synthetic':np.zeros(len(train),dtype=bool)}


def score_state(state,val):
    model=fit_probe(state['X'],state['y'],state['weights'])
    probability=model.predict_proba(state['V'])[:,1]
    metrics=classification_metrics(val[f'y_{state["condition"]}'].to_numpy(),probability,groups(val))
    metrics['converged']=model.comparison_converged_
    return metrics,model,probability


def state_copy(state): return {k:(v.copy() if hasattr(v,'copy') else v) for k,v in state.items()}


def run_outlier_comparison(state,val,scaled_numeric):
    progress(f'{state["condition"]}: training anomaly-screen comparison (downweight, never delete validation/test).')
    baseline,_,_=score_state(state,val)
    rows=[{'method':'No change','family':'reference','status':'measured',**baseline,'downweighted_rows':0}]
    candidates={'No change':state};scores={};extreme=np.zeros(len(state['y']),bool)
    center=np.median(scaled_numeric,axis=0);spread=np.quantile(scaled_numeric,.75,axis=0)-np.quantile(scaled_numeric,.25,axis=0)
    extreme=(np.abs((scaled_numeric-center)/np.where(spread>1e-8,spread,1))>6).any(axis=1)
    detected=extreme.mean()>.01
    if detected:
        for name in ['iqr','mad','isolation_forest','lof','one_class_svm','ecod','copod']:
            try:
                if name not in OUTLIER_CACHE:
                    OUTLIER_CACHE[name]=fit_outlier_detector(name,scaled_numeric,contamination=.02,seed=SEED,fit_cap=10000)
                detector=OUTLIER_CACHE[name]
                flag=detector.predict_outliers(scaled_numeric)
                candidate=state_copy(state);candidate['weights']=state['weights']*np.where(flag,.5,1.)
                result,_,_=score_state(candidate,val)
                # Score-stability: training split-fit ranking overlap, not true anomaly accuracy.
                scores[name]=detector.training_scores_
                rows.append({'method':name,'family':METHOD_METADATA[name]['family'],'status':'measured',**result,
                             'downweighted_rows':int(flag.sum()),'outlier_truth_auroc':np.nan,'precision_at_k':np.nan,
                             'outlier_truth_f1':np.nan,'outlier_brier':np.nan})
                candidates[name]=candidate
                save_json(OUT/f'{state["condition"]}_outlier_{name}_fit.json',detector.diagnostics_)
            except Exception as error: rows.append({'method':name,'status':'failed','error':str(error)})
    comparison=pd.DataFrame(rows)
    comparison['eligible']=comparison.apply(lambda row:row.status=='measured' and utility_guard(row,baseline),axis=1)
    best=comparison.loc[comparison.eligible].sort_values(['balanced_accuracy','auroc'],ascending=False).iloc[0]
    selected=best.method if detected and best.balanced_accuracy>=baseline['balanced_accuracy']+.002 else 'No change'
    comparison['selected']=comparison.method.eq(selected)
    comparison.to_csv(OUT/f'{state["condition"]}_comparison_outlier.csv',index=False)
    after=extreme.astype(float)*candidates[selected]['weights']
    before=pd.Series({'Statistical-tail rate (%)':100*extreme.mean(),'Effective tail weight (%)':100*extreme.mean()})
    after_vec=pd.Series({'Statistical-tail rate (%)':100*extreme.mean(),'Effective tail weight (%)':100*after.mean()})
    stage={'status':'detected' if detected else 'not_detected','selected':selected,'before':before,'after':after_vec,
       'comparison':comparison,'note':'Statistical tails are not confirmed clinical errors. Selected training weights may change; raw physiology and all held-out patients remain.'}
    return candidates[selected],stage


def sensitive_design(frame):
    return pd.get_dummies(frame[['sens_sex','sens_race_ethnicity']].fillna('Unknown').astype(str),dtype=float)


def run_demographic_comparison(state,val,feature_names):
    progress(f'{state["condition"]}: measured sex/race/intersectional disparities and preprocessing.')
    base,_,p0=score_state(state,val)
    rows=[{'method':'No change','family':'reference','status':'measured',**base}];candidates={'No change':state}
    detected=bool(np.isfinite(base.get('max_equalized_odds_gap',np.nan)) and base['max_equalized_odds_gap']>.10)
    if detected:
        for attribute in ['sens_sex','sens_race_ethnicity','intersection']:
            name='Reweighing_'+attribute
            group=(state['groups'].sens_sex.astype(str)+'|'+state['groups'].sens_race_ethnicity.astype(str)
                   if attribute=='intersection' else state['groups'][attribute].astype(str))
            candidate=state_copy(state)
            candidate['weights']=state['weights']*reweighing_weights(state['y'],group)
            result,_,_=score_state(candidate,val)
            rows.append({'method':name,'family':'established_reweighing','status':'measured',**result});candidates[name]=candidate
        for alpha in [.25,.5,1.]:
            name=f'Linear_correlation_projection_{alpha}'
            # Remove explicit sex predictor columns before projection. Age remains a clinical predictor.
            keep=np.array([not str(k).startswith('sex_') for k in feature_names])
            projection=CorrelationProjection(alpha=alpha).fit(state['X'][:,keep],sensitive_design(state['groups']))
            candidate=state_copy(state)
            candidate['X']=projection.transform(state['X'][:,keep],sensitive_design(state['groups']))
            candidate['V']=projection.transform(state['V'][:,keep],sensitive_design(groups(val)))
            candidate['projection']=(projection,keep)
            result,_,_=score_state(candidate,val)
            rows.append({'method':name,'family':'established_linear_projection','status':'measured',**result});candidates[name]=candidate
    comparison=pd.DataFrame(rows)
    comparison['eligible']=comparison.apply(lambda row:row.status=='measured' and utility_guard(row,base,auc_loss=.02,ba_loss=.02,eo_increase=0),axis=1)
    available=comparison.loc[comparison.eligible & comparison.max_equalized_odds_gap.notna()]
    selected='No change'
    if detected and len(available):
        winner=available.sort_values(['max_equalized_odds_gap','balanced_accuracy'],ascending=[True,False]).iloc[0]
        if winner.max_equalized_odds_gap<base['max_equalized_odds_gap']-.01: selected=winner.method
    metrics,_,p1=score_state(candidates[selected],val)
    comparison['selected']=comparison.method.eq(selected)
    comparison.to_csv(OUT/f'{state["condition"]}_comparison_demographic.csv',index=False)
    before=group_gap_vectors(val[f'y_{state["condition"]}'],p0,val)
    after=group_gap_vectors(val[f'y_{state["condition"]}'],p1,val).reindex(before.index)
    if selected=='No change': after=before.copy()
    stage={'status':'detected' if detected else 'not_detected','selected':selected,'before':before,'after':after,
        'comparison':comparison,'note':'Validation subgroup TPRs. Detection: eligible maximum sex/race/intersection EO gap >0.10; an observed disparity is not proof of causal discrimination. Age is audited separately.'}
    return candidates[selected],stage


def noise_diagnostic(train,X,condition):
    """Cross-fitted training-only disagreement; never claim to identify clinical mislabels."""
    y=train[f'y_{condition}'].to_numpy(int);oof=np.zeros(len(y));folds=GroupKFold(n_splits=3)
    for fit,check in folds.split(X,y,train.meta_patient_id):
        # Every preprocessing fit is confined to the inner training patients.
        imputer=make_imputer('median').fit(train.iloc[fit][NUMERIC].to_numpy(float))
        inner_train=imputer.transform(train.iloc[fit][NUMERIC].to_numpy(float))
        inner_check=imputer.transform(train.iloc[check][NUMERIC].to_numpy(float))
        scaler=StandardScaler().fit(inner_train)
        encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(train.iloc[fit][CATEGORICAL].astype(str))
        xf=encode_matrix(inner_train,train.iloc[fit],encoder,scaler)
        xc=encode_matrix(inner_check,train.iloc[check],encoder,scaler)
        model=fit_probe(xf,y[fit]);oof[check]=model.predict_proba(xc)[:,1]
    from cleanlab.filter import find_label_issues
    probabilities=np.column_stack([1-oof,oof])
    issues=find_label_issues(y,probabilities,filter_by='prune_by_noise_rate',n_jobs=1)
    discrepancy=(oof>=.5)!=y
    comparison=pd.DataFrame([
       {'method':'Cross-fitted disagreement','family':'established_diagnostic','flag_rate':discrepancy.mean(),
        'label_error_detection_f1':np.nan,'status':'diagnostic_no_ground_truth'},
       {'method':'Official Cleanlab confident learning','family':'modern_2021','flag_rate':issues.mean(),
        'label_error_detection_f1':np.nan,'status':'diagnostic_no_ground_truth'}])
    comparison.to_csv(OUT/f'{condition}_comparison_data_noise.csv',index=False)
    pd.DataFrame({'meta_stay_id':train.meta_stay_id,'out_of_fold_probability':oof,'disagreement_flag':discrepancy,
                  'unverified_cleanlab_issue':issues}).to_csv(OUT/f'{condition}_noise_flags_training_only.csv',index=False)
    metrics=pd.Series({'Model-disagreement flags (%)':100*discrepancy.mean(),'Unverified Cleanlab flags (%)':100*issues.mean()})
    return {'status':'not_assessable','selected':'No label changes','before':metrics,'after':metrics.copy(),
       'comparison':comparison,'note':'No adjudicated label-error ground truth. Cross-fitted flags are review candidates, not confirmed mistakes. No label is flipped or patient deleted; temporal filtering is unavailable for stay-level summaries.'}


def simple_domain_stage():
    metrics=pd.Series({'Observed MIMIC hospital domains':1.,'One-hot encoder refits on held-out rows':0.})
    comparison=pd.DataFrame([{'method':'Frozen training one-hot encoding','family':'established','status':'measured'},
       {'method':'CORAL / OTDA / DANN / IRM','family':'domain_methods','status':'not_applicable_single_domain',
        'mmd':np.nan,'domain_classifier_auc':np.nan,'target_domain_auroc':np.nan}])
    return {'status':'not_assessable','selected':'Training one-hot encoding; no hospital adaptation','before':metrics,'after':metrics.copy(),
       'comparison':comparison,'note':'MIMIC contains BIDMC only. No hospital-domain comparison can be inferred. eICU stays unopened and cannot influence preprocessing selection.'}


def cluster_bootstrap(y,p0,p1,patient_ids,draws=200):
    """Paired patient-cluster resampling preserves dependence across repeated stays."""
    y=np.asarray(y);p0=np.asarray(p0);p1=np.asarray(p1);ids=np.asarray(patient_ids)
    unique,inverse=np.unique(ids,return_inverse=True);rng=np.random.default_rng(SEED)
    collected=[]
    for _ in range(draws):
        counts=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))
        row_weight=counts[inverse]
        # Weighted confusion-derived balanced accuracy avoids materializing repeated rows.
        def ba(p):
            pred=p>=.5;positive=np.sum(row_weight*(y==1));negative=np.sum(row_weight*(y==0))
            return .5*(np.sum(row_weight*(y==1)*pred)/positive+np.sum(row_weight*(y==0)*(~pred))/negative)
        if np.sum(row_weight*(y==1)) and np.sum(row_weight*(y==0)): collected.append(ba(p1)-ba(p0))
    return {'metric':'balanced_accuracy_after_minus_before','patient_cluster_bootstrap_draws':len(collected),
            'ci_2_5':float(np.quantile(collected,.025)),'ci_97_5':float(np.quantile(collected,.975)),
            'modern_superiority_not_assumed':True}
