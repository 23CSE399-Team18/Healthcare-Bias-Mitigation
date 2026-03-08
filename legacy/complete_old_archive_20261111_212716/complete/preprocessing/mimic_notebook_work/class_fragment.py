"""Inline genuine CTGAN, categorical-aware and numeric-arm class comparisons."""

def fit_ctgan_training(train,nt):
    from ctgan import CTGAN
    import torch
    started=time.monotonic();rng=np.random.default_rng(SEED)
    fitted=np.sort(rng.choice(len(train),min(CTGAN_FIT_CAP,len(train)),replace=False))
    table=pd.DataFrame(index=np.arange(len(train)))
    # Reparameterization guarantees min <= mean <= max after generation. CTGAN
    # itself is the official package; this input/output constraint policy is ours.
    for vital in VITAL_NAMES:
        mean=nt[:,NUMERIC.index(f'{vital}_mean')];lo=nt[:,NUMERIC.index(f'{vital}_min')];hi=nt[:,NUMERIC.index(f'{vital}_max')]
        table[vital+'__mean']=mean
        table[vital+'__log_delta_low']=np.log1p(np.maximum(mean-lo,0))
        table[vital+'__log_delta_high']=np.log1p(np.maximum(hi-mean,0))
    for col in CATEGORICAL+['sens_race_ethnicity']:
        table[col]=train[col].astype(str).to_numpy()
    for col in NUMERIC: table['missing__'+col]=train[col].isna().astype(int).astype(str).to_numpy()
    for condition in CONDITIONS: table['y_'+condition]=train['y_'+condition].astype(str).to_numpy()
    discrete=[c for c in table if c not in [v+'__'+suffix for v in VITAL_NAMES for suffix in ['mean','log_delta_low','log_delta_high']]]
    generator=CTGAN(epochs=CTGAN_EPOCHS,embedding_dim=64,generator_dim=(128,128),discriminator_dim=(128,128),
                   batch_size=500,pac=10,enable_gpu=torch.cuda.is_available(),verbose=False)
    generator.set_random_state(SEED)
    progress(f'Official CTGAN joint training: {len(fitted):,} training rows, {CTGAN_EPOCHS} epochs.')
    generator.fit(table.iloc[fitted].reset_index(drop=True),discrete_columns=discrete)
    if hasattr(generator,'loss_values'): generator.loss_values.to_csv(OUT/'ctgan_training_losses.csv',index=False)
    generator.save(str(OUT/'ctgan_training_only.pkl'))
    save_json(OUT/'ctgan_fit_audit.json',{'package':'ctgan','epochs':CTGAN_EPOCHS,'fit_rows':len(fitted),
      'fit_partition':'train_only','seed':SEED,'seconds':time.monotonic()-started,'cuda':torch.cuda.is_available(),
      'joint_targets_used_in_generator_not_as_classifier_predictors':CONDITIONS,
      'condition_not_guaranteed_by_api':'Generated target is filtered explicitly',
      'constraint_policy':'Mean plus nonnegative log-deltas; clip generated physiology to declared bounds',
      'not_a_convergence_or_SOTA_claim':True})
    return generator,table


def ctgan_rows(generator,condition,wanted,label,train):
    accepted=[];drawn=0;attempts=0
    categories={c:set(train[c].astype(str)) for c in CATEGORICAL+['sens_race_ethnicity']}
    while sum(len(a) for a in accepted)<wanted and attempts<8:
        need=wanted-sum(len(a) for a in accepted)
        count=min(max(need*2,500),30000)
        generated=generator.sample(count,condition_column='y_'+condition,condition_value=str(label))
        drawn+=len(generated);attempts+=1
        valid=generated['y_'+condition].astype(str).eq(str(label))
        for c,levels in categories.items(): valid&=generated[c].astype(str).isin(levels)
        for c in NUMERIC: valid&=generated['missing__'+c].astype(str).isin(['0','1'])
        accepted.append(generated.loc[valid].head(need))
    if sum(len(a) for a in accepted)<wanted:
        raise MethodNotApplicable(f'CTGAN conditional generation accepted fewer than {wanted} requested rows within bounded attempts')
    generated=pd.concat(accepted,ignore_index=True).head(wanted)
    numeric=np.empty((wanted,len(NUMERIC)))
    clipped=0
    for vital in VITAL_NAMES:
        lo,hi=PHYS_BOUNDS[vital]
        mean=generated[vital+'__mean'].to_numpy(float)
        clipped+=int(((mean<=lo)|(mean>=hi)).sum())
        mean=np.clip(mean,lo+1e-5,hi-1e-5)
        delta_low=np.expm1(np.clip(generated[vital+'__log_delta_low'].to_numpy(float),0,20))
        delta_high=np.expm1(np.clip(generated[vital+'__log_delta_high'].to_numpy(float),0,20))
        numeric[:,NUMERIC.index(vital+'_mean')]=mean
        numeric[:,NUMERIC.index(vital+'_min')]=np.clip(mean-delta_low,lo+1e-5,mean)
        numeric[:,NUMERIC.index(vital+'_max')]=np.clip(mean+delta_high,mean,hi-1e-5)
    mask=generated[['missing__'+c for c in NUMERIC]].astype(int).to_numpy(float)
    frame=pd.DataFrame({'age_group':generated.age_group.astype(str),'sex':generated.sex.astype(str),
      'sens_age_group':generated.age_group.astype(str),'sens_sex':generated.sex.astype(str),
      'sens_race_ethnicity':generated.sens_race_ethnicity.astype(str)})
    return numeric,mask,frame,{'requested':wanted,'drawn':drawn,'accepted':wanted,'attempts':attempts,
                             'mean_bound_clippings':clipped,'generated_protected_groups_are_synthetic':True}


def make_resampled_state(method,state,train,encoder,scaler,generator=None):
    counts=np.bincount(state['y'],minlength=2);minority=int(np.argmin(counts))
    if method=='class_weights':
        candidate=state_copy(state)
        factors=len(state['y'])/(2*counts)
        candidate['weights']=state['weights']*factors[state['y']]
        return candidate,{'effective_class_counts':np.bincount(state['y'],weights=factors[state['y']],minlength=2).tolist(),
                          'synthetic_rows':0,'weighting_only':True}
    if method=='ctgan':
        if generator is None: raise MethodNotApplicable('CTGAN did not train successfully')
        needed=int(counts.max()-counts.min())
        N,M,G,info=ctgan_rows(generator,state['condition'],needed,minority,train)
        syntheticX=encode_matrix(N,G,encoder,scaler,mask=M)
        candidate=state_copy(state)
        candidate['X']=np.vstack([state['X'],syntheticX]);candidate['y']=np.r_[state['y'],np.full(needed,minority)]
        candidate['weights']=np.r_[state['weights'],np.ones(needed)]
        candidate['groups']=pd.concat([state['groups'],G[GROUP_COLUMNS]],ignore_index=True)
        meta=pd.DataFrame({c:pd.NA for c in state['meta']},index=np.arange(needed))
        meta['meta_patient_id']=[f'synthetic:{state["condition"]}:{i}' for i in range(needed)]
        meta['meta_stay_id']=[f'synthetic:{state["condition"]}:{i}' for i in range(needed)]
        meta['meta_source']='mimic_ctgan_generated';meta['meta_split']='train';meta['meta_domain_hospitalid']='synthetic_training'
        candidate['meta']=pd.concat([state['meta'],meta],ignore_index=True)
        candidate['synthetic']=np.r_[state['synthetic'],np.ones(needed,bool)]
        return candidate,{**info,'synthetic_rows':needed,'effective_class_counts':np.bincount(candidate['y'],minlength=2).tolist()}
    if method in ['random_oversampling','nearmiss']:
        xr,yr,info=resample_training(method,state['X'],state['y'],seed=SEED,sampling_strategy=1.)
        indices=info['original_row_indices'];candidate=state_copy(state)
        candidate['X']=xr;candidate['y']=yr;candidate['weights']=state['weights'][indices]
        candidate['groups']=state['groups'].iloc[indices].reset_index(drop=True)
        candidate['meta']=state['meta'].iloc[indices].reset_index(drop=True)
        candidate['synthetic']=state['synthetic'][indices]
        info['effective_class_counts']=np.bincount(yr,minlength=2).tolist()
        return candidate,info
    if method=='smotenc':
        # Resampling space includes categorical audit metadata explicitly. Race
        # helps define generated categories but is removed from classifier X.
        category_cols=CATEGORICAL+['sens_race_ethnicity']
        levels={col:sorted(train[col].astype(str).unique()) for col in category_cols}
        codes=np.column_stack([train[c].astype(str).map({v:i for i,v in enumerate(levels[c])}) for c in category_cols])
        continuous=scaler.transform(state['numeric'])
        mask=state['missing']
        mixed=np.column_stack([continuous,mask,codes])
        cat_indices=list(range(len(NUMERIC),mixed.shape[1]))
        xr,yr,info=resample_training('smotenc',mixed,state['y'],seed=SEED,sampling_strategy=1.,categorical_indices=cat_indices)
        generated=xr[len(train):]
        G=pd.DataFrame({c:np.asarray(levels[c])[generated[:,2*len(NUMERIC)+i].astype(int)] for i,c in enumerate(category_cols)})
        G['sens_age_group']=G.age_group;G['sens_sex']=G.sex
        generated_numeric=scaler.inverse_transform(generated[:,:len(NUMERIC)])
        # Project generated, never original, triples to valid physiological order.
        corrected=0
        for vital in VITAL_NAMES:
            ix=[NUMERIC.index(vital+'_'+s) for s in ['mean','min','max']]
            raw=generated_numeric[:,ix].copy();ordered=np.sort(raw,axis=1)
            generated_numeric[:,ix[1]]=ordered[:,0];generated_numeric[:,ix[0]]=ordered[:,1];generated_numeric[:,ix[2]]=ordered[:,2]
            corrected+=int((np.abs(raw-generated_numeric[:,ix])>1e-8).any(axis=1).sum())
        newx=encode_matrix(generated_numeric,G,encoder,scaler,mask=generated[:,len(NUMERIC):2*len(NUMERIC)])
        candidate=state_copy(state);candidate['X']=np.vstack([state['X'],newx]);candidate['y']=yr
        n=len(G);candidate['weights']=np.r_[state['weights'],np.ones(n)]
        candidate['groups']=pd.concat([state['groups'],G[GROUP_COLUMNS]],ignore_index=True)
        meta=pd.DataFrame({c:pd.NA for c in state['meta']},index=np.arange(n))
        meta['meta_patient_id']=[f'synthetic_smotenc:{state["condition"]}:{i}' for i in range(n)]
        meta['meta_stay_id']=meta.meta_patient_id;meta['meta_split']='train';meta['meta_source']='mimic_smotenc_generated';meta['meta_domain_hospitalid']='synthetic_training'
        candidate['meta']=pd.concat([state['meta'],meta],ignore_index=True);candidate['synthetic']=np.r_[state['synthetic'],np.ones(n,bool)]
        return candidate,{'synthetic_rows':n,'generated_order_projection_count':corrected,
          'categorical_audit_fields_used_for_sampling_not_classifier':category_cols,
          'effective_class_counts':np.bincount(yr,minlength=2).tolist()}
    raise MethodNotApplicable('Unknown primary class method '+method)


def numeric_resampling_arm(state,val,nt,nv):
    """All numeric-only samplers use the SAME predictor ablation; never rank against mixed primary arm."""
    scaler=StandardScaler().fit(nt);xt=scaler.transform(nt);xv=scaler.transform(nv)
    y=state['y'];rows=[]
    for name in ['No change','smote','adasyn','borderline_smote','nearmiss','kmeans_smote']:
        try:
            if name=='No change':xr,yr,info=xt,y,{'synthetic_rows':0}
            else:xr,yr,info=resample_training(name,xt,y,SEED,sampling_strategy=1.)
            model=fit_probe(xr,yr);p=model.predict_proba(xv)[:,1]
            result=classification_metrics(val['y_'+state['condition']],p,groups(val))
            rows.append({'method':name,'family':METHOD_METADATA.get(name,{}).get('family','reference'),
                'arm':'numeric_only_ablation_not_primary_selection','status':'measured',**result,
                **class_balance_metrics(yr),'converged':model.comparison_converged_,
                'synthetic_rows':info.get('synthetic_rows',0),'eligible_for_primary_selection':False})
        except Exception as error:rows.append({'method':name,'status':'failed','error':str(error),'eligible_for_primary_selection':False})
    table=pd.DataFrame(rows);table.to_csv(OUT/f'{state["condition"]}_comparison_class_numeric_ablation.csv',index=False)
    return table


def run_class_comparison(state,train,val,nt,nv,encoder,scaler,generator):
    progress(f'{state["condition"]}: class imbalance comparison on unchanged validation patients.')
    baseline,_,_=score_state(state,val);balance=class_balance_metrics(state['y'])
    counts=np.bincount(state['y'],minlength=2);detected=counts.max()/counts.min()>1.5
    rows=[{'method':'No change','family':'reference','status':'measured',**baseline,**balance,'synthetic_rows':0}]
    candidates={'No change':state};infos={'No change':{'effective_class_counts':counts.tolist()}}
    if detected:
        for name in ['class_weights','random_oversampling','nearmiss','smotenc','ctgan']:
            try:
                candidate,info=make_resampled_state(name,state,train,encoder,scaler,generator)
                result,_,_=score_state(candidate,val)
                rows.append({'method':name,'family':'modern_2019_generative' if name=='ctgan' else METHOD_METADATA.get(name,{}).get('family','established_weighting'),
                             'status':'measured',**result,**class_balance_metrics(candidate['y']),
                             'synthetic_rows':info.get('synthetic_rows',0)})
                candidates[name]=candidate;infos[name]=info
            except Exception as error:rows.append({'method':name,'status':'failed','error':f'{type(error).__name__}: {error}'})
        numeric_resampling_arm(state,val,nt,nv)
    comparison=pd.DataFrame(rows)
    comparison['eligible']=comparison.apply(lambda row:row.status=='measured' and utility_guard(row,baseline),axis=1)
    best=comparison.loc[comparison.eligible].sort_values(['balanced_accuracy','auroc'],ascending=False).iloc[0]
    selected=best.method if detected and best.balanced_accuracy>=baseline['balanced_accuracy']+.002 else 'No change'
    comparison['selected']=comparison.method.eq(selected)
    comparison.to_csv(OUT/f'{state["condition"]}_comparison_class_imbalance.csv',index=False)
    for name,info in infos.items():save_json(OUT/f'{state["condition"]}_class_{name}_audit.json',{k:v for k,v in info.items() if k!='original_row_indices'})
    before=pd.Series(counts,index=['Not documented (0)','Recorded condition (1)'])
    after=pd.Series(infos[selected]['effective_class_counts'],index=before.index,dtype=float)
    if selected=='No change':after=before.copy()
    stage={'status':'detected' if detected else 'not_detected','selected':selected,'before':before,'after':after,
      'comparison':comparison,'note':'Training counts (effective weighted counts for weighting). Trigger: majority/minority ratio >1.5. Held-out labels and prevalence never resampled. Numeric-only SMOTE variants are reported separately.'}
    return candidates[selected],stage
