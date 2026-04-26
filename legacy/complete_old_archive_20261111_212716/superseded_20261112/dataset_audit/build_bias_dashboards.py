"""One interactive HTML per bias; aggregate-only plots from frozen artifacts."""
from pathlib import Path
import argparse,copy,hashlib,html,json,os,sys,warnings
import numpy as np
import pandas as pd
import joblib
from plotly.offline import get_plotlyjs
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'inprocessing/code'),str(ROOT/'latent/code')]
from run_inprocessing import JOBS,load
ATTRS=('sens_sex','sens_age_group','sens_race_ethnicity')
PRE={'missing_data':'Missing data bias','class_imbalance':'Class imbalance bias','outlier':'Outlier screen','feature_scale':'Feature scale bias','demographic':'Demographic outcome differences','duplicate':'Duplicate record bias','noise':'Data noise assessment','encoding_domain':'Encoding / domain shift'}
IN={'demographic':'Demographic prediction dependence','intersectional':'Intersectional / subgroup unfairness','miscalibration':'Predictive miscalibration','cross_hospital':'Cross-hospital generalization risk'}
LAT={'information_preservation':'Clinical information preservation','latent_structure':'Latent structure','alignment':'Input / latent alignment','domain_invariance':'Domain association'}
FAMILIES={'demographic':['AdversarialFair_tabular','DANN_source_domains','GroupDRO_sex_label','FairERM_softEO'], 'intersectional':['FOMO_weight_search','Intersectional_adversary','Constrained_softEO','GroupDRO_intersection_label'], 'miscalibration':['AvUC_deterministic','CRL_history','UWG_Brier','Brier_regularized'], 'cross_hospital':['Fishr_last_layer','VREx','EQRM_empirical','Fish_first_order']}
REGISTRY=[]

def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def clean(o):
    if isinstance(o,dict):return {str(k):clean(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)):return [clean(v) for v in o]
    if isinstance(o,np.ndarray):return clean(o.tolist())
    if isinstance(o,np.generic):return clean(o.item())
    if isinstance(o,float) and not np.isfinite(o):return None
    return o
def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(clean(obj),allow_nan=False,separators=(',',':')),encoding='utf-8')
def state(value,limit,rule,available=True):
    return {'label':'Not applicable' if not available else 'Insufficient evidence' if value is None else 'Detected by rule' if value>limit else 'Not detected by rule','value':value,'limit':limit,'rule':rule}
def unavailable(rule):return state(None,0,rule)
def bars(title,labels,series,ylabel='',horizontal=False):
    colors=['#77ade0','#83b657','#a980c2','#e1a559']
    traces=[]
    for i,(name,values) in enumerate(series.items()):
        traces.append({'type':'bar','name':name,'marker':{'color':colors[i%len(colors)]},'x':values if horizontal else labels,'y':labels if horizontal else values,'orientation':'h' if horizontal else 'v','hovertemplate':'%{x}: %{y}<extra>%{fullData.name}</extra>'})
    return {'title':title,'data':traces,'layout':{'barmode':'group','yaxis':{'title':{'text':ylabel if not horizontal else ''}},'xaxis':{'title':{'text':ylabel if horizontal else ''}, 'automargin':True},'height':max(360,24*len(labels)+180) if horizontal else 400}}
def line(title,traces,xlabel,ylabel):return {'title':title,'data':traces,'layout':{'xaxis':{'title':xlabel},'yaxis':{'title':ylabel},'height':380}}
def table(title,rows):return {'title':title,'table':clean(rows)}
def known(v):return str(v) not in ('Unknown','nan','<NA>','None','')
def rows(path):
    cols=pd.read_csv(path,nrows=0).columns
    return pd.read_csv(path,dtype={c:'string' for c in cols if c.startswith('meta_')})
def index(df,ids):return pd.MultiIndex.from_frame(df[ids])

def source_icu(source):
    frame=None
    for task in ('heart','respiratory','diabetes','hypertension'):
        p=ROOT/f'cross_dataset/four_conditions/{source}/{task}_model_ready.csv';d=rows(p)
        if frame is None:frame=d.drop(columns='target').copy()
        else:pd.testing.assert_frame_equal(frame[d.columns.drop('target')],d.drop(columns='target'))
        frame['target_'+task]=d.target.to_numpy()
    return frame

def variables(before,after,names):
    result={};qs=np.linspace(0,1,101)
    for i,name in enumerate(names):
        a=before[:,i];a=a[np.isfinite(a)];b=after[:,i];b=b[np.isfinite(b)]
        both=np.r_[a,b]
        edges=np.histogram_bin_edges(both,bins=24) if len(both) else np.linspace(0,1,25)
        ac=np.histogram(a,edges)[0];bc=np.histogram(b,edges)[0]
        assert ac.sum()==len(a) and bc.sum()==len(b)
        result[name]={'edges':edges,'before_counts':ac,'after_counts':bc,'before_n':len(a),'after_n':len(b),'before_quantiles':np.quantile(a,qs) if len(a) else None,'after_quantiles':np.quantile(b,qs) if len(b) else None}
    return result

def pre_branch(name,base,source,external=None,original=False):
    print('Preprocessing bias diagnostics: '+name,flush=True)
    prep=joblib.load(base/'preprocessor.joblib');features=list(prep.feature_names_in_)
    nums=list(prep.transformers_[0][2]);cats=list(prep.transformers_[1][2]);num=prep.named_transformers_['numeric'];cat=prep.named_transformers_['categorical']
    ids=['meta_domain_state','meta_record_id'] if name=='heart' else ['meta_record_id'] if name in ('diabetes','hypertension') else ['meta_patientunitstayid'] if 'respiratory' in name else ['meta_stay_id']
    sources={False:source.copy(),True:external.copy() if external is not None else None}
    for is_ext,d in sources.items():
        if d is not None:d.index=index(d,ids);assert d.index.is_unique
    parts={};matrices={};metadata={};fingerprints={}
    splits=['train','validation','test']+(['external'] if external is not None else [])
    for s in splits:
        rp=(ROOT/'latent/outputs'/name/f'rows_{s}.csv') if original else base/f'rows_{s}.csv'
        metadata[s]=rows(rp);d=sources[s=='external'].loc[index(metadata[s],ids)].reset_index(drop=True)
        with np.load(base/f'X_{s}.npz') as a:matrices[s]=a['X']
        frame=d[features].copy()
        for c in cats:frame[c]=frame[c].astype(object).where(frame[c].notna(),np.nan)
        expected=prep.transform(frame)
        np.testing.assert_allclose(expected,matrices[s],atol=2e-6,rtol=2e-6)
        targets=[c for c in d if c=='target' or c.startswith('target_')]
        for target in targets:np.testing.assert_array_equal(d[target],metadata[s][target])
        parts[s]=d;fingerprints[s]={'X_sha256':sha(base/f'X_{s}.npz'),'rows_sha256':sha(rp),'transform_reproduced':True}
    train=parts['train'];continuous=[c for c in nums if train[c].nunique()>=20 and train[c].quantile(.75)>train[c].quantile(.25)]
    fences={c:(float(train[c].quantile(.25)-1.5*(train[c].quantile(.75)-train[c].quantile(.25))),float(train[c].quantile(.75)+1.5*(train[c].quantile(.75)-train[c].quantile(.25)))) for c in continuous}
    allstates={bias:{} for bias in PRE};stats={}
    train_imputed=num['imputer'].transform(train[nums])[:,:len(nums)]
    trainmean=train_imputed.mean(0);trainsd=train_imputed.std(0);trainsd[trainsd==0]=1
    for split,d in parts.items():
        a=d[nums].to_numpy(float);b=num['imputer'].transform(d[nums])[:,:len(nums)];x=matrices[split];n=len(d)
        v=variables(a,b,nums);labels=[c for c in d if c=='target' or c.startswith('target_')]
        missing=d[features].isna().sum();allmissing=int(missing.sum());after_missing=int((~np.isfinite(x)).sum())
        labelcounts=[{'target':c,'positive':int(d[c].sum()),'negative':int(n-d[c].sum()),'minority_share':float(min(d[c].mean(),1-d[c].mean()))} for c in labels]
        rates_a=[];rates_b=[]
        for c in continuous:
            j=nums.index(c);lo,hi=fences[c];observed=a[:,j][np.isfinite(a[:,j])]
            rates_a.append(float(np.mean((observed<lo)|(observed>hi))) if len(observed) else None)
            rates_b.append(float(np.mean((b[:,j]<lo)|(b[:,j]>hi))))
        continuous_sd=[float(np.nanstd(a[:,nums.index(c)])) for c in continuous];scaled_sd=[float(x[:,nums.index(c)].std()) for c in continuous]
        def ratio(vals):
            finite=[z for z in vals if z>1e-10 and np.isfinite(z)]
            return max(finite)/min(finite) if len(finite)>=2 else None
        demographic=[];gaps=[]
        for attr in ATTRS:
            for target in labels:
                valid=[]
                for group,sub in d.groupby(attr,dropna=False):
                    supported=len(sub)>=30 and known(group);rate=float(sub[target].mean())
                    demographic.append({'attribute':attr,'target':target,'group':str(group),'rows':len(sub),'prevalence':rate,'supported':supported})
                    if supported:valid.append(rate)
                if len(valid)>=2:gaps.append(max(valid)-min(valid))
        duplicate_ids=int(index(d,ids).duplicated().sum())
        duplicates_before=int(d[features].duplicated().sum());duplicates_after=int(pd.DataFrame(x).duplicated().sum())
        ambiguity=[]
        # Hash grouping is a diagnostic for repeated measured feature vectors,
        # not identity resolution. Use exact feature-value groupby to avoid hash collisions.
        for target in labels:
            grouped=d.groupby(features,dropna=False,observed=True)[target].agg(['min','max','size'])
            conflict=grouped['min'].ne(grouped['max']);ambiguity.append({'target':target,'conflicting_patterns':int(conflict.sum()),'rows_in_conflicting_patterns':int(grouped.loc[conflict,'size'].sum())})
        ci=cat['imputer'].transform(d[cats].astype(object).where(d[cats].notna(),np.nan))
        unseen=[{'feature':c,'unseen_rows':int((~np.isin(ci[:,i],cat['encode'].categories_[i])).sum())} for i,c in enumerate(cats)]
        shift=np.abs((b.mean(0)-trainmean)/trainsd);domaincol=next((c for c in ('meta_domain_hospitalid','meta_domain_state','meta_domain_region') if c in d),None)
        domains=[] if domaincol is None else [{'domain':str(k),'rows':int(z)} for k,z in d[domaincol].value_counts(dropna=False).items()]
        shared={'rows':n,'numeric_features':len(nums),'input_features':len(features),'variables':v,'before_label':'Model-ready (before)','after_label':'Imputed, original units (after)'}
        def put(bias,before,after,panels,note,vars=True):
            allstates[bias][split]={'default':{**shared,'before':before,'after':after,'panels':panels,'note':note,'variables':v if vars else {}}}
        ordered=list(missing.sort_values(ascending=False).index)
        put('missing_data',state(allmissing,0,'Any missing predictor cell'),state(after_missing,0,'Any nonfinite encoded output cell'),[bars('Missing input cells by variable',ordered,{'Before':missing[ordered].tolist(),'After imputation':[0]*len(ordered)},'Missing cells'),table('Full partition reconciliation',[{'rows':n,'missing_input_cells':allmissing,'nonfinite_saved_output_cells':after_missing}])],'Imputation uses source-training statistics. Distributions compare observed values with imputed values on the same row cohort; histogram denominators differ when observations were missing.')
        minor=min(r['minority_share'] for r in labelcounts)
        put('class_imbalance',state(.2-minor,0,'Minority class share <20%'),state(.2-minor,0,'Minority class share <20%'),[bars('Minority class share: before and after',[r['target'] for r in labelcounts],{'Before':[100*r['minority_share'] for r in labelcounts],'After':[100*r['minority_share'] for r in labelcounts]},'Minority share (%)'),table('Unchanged target counts',labelcounts)],'The verified baseline does not resample. Counts and labels are unchanged; this page does not claim imbalance was removed.',False)
        put('outlier',state(max((r for r in rates_a if r is not None),default=None),.01,'Any eligible variable with >1% outside training IQR fences'),state(max(rates_b,default=None),.01,'Same training fences after imputation'),[bars('Statistical tail rates (training-fitted fences)',continuous,{'Before observed':[None if r is None else 100*r for r in rates_a],'After imputed':[100*r for r in rates_b]},'Outside fences (%)',True),table('Training-only fences',[{'feature':c,'lower':fences[c][0],'upper':fences[c][1]} for c in continuous])],'The fixed baseline does not remove outliers. These are statistical tail flags, not confirmed invalid clinical measurements. Missing values are excluded from the before denominator.')
        put('feature_scale',state(ratio(continuous_sd),10,'Largest/smallest positive SD >10 for eligible continuous inputs'),state(ratio(scaled_sd),10,'Same variables in saved standardized matrix'),[bars('Standard deviations: raw versus standardized',continuous,{'Before (raw units)':continuous_sd,'After (standardized)':scaled_sd},'SD (different units)',True)],'This compares numeric scales, not patient distributions in a common unit. Histogram/box/Q-Q context remains in original imputed units; the scale change is shown in the SD chart.')
        dg=max(gaps,default=None)
        supported=[r for r in demographic if r['supported']]
        put('demographic',state(dg,.1,'Supported group outcome-prevalence range >0.10'),state(dg,.1,'Labels and group metadata unchanged'),[bars('Supported group outcome prevalence',[f'{r["target"]}: {r["attribute"]} / {r["group"]}' for r in supported],{'Before':[r['prevalence'] for r in supported],'After':[r['prevalence'] for r in supported]},'Recorded outcome prevalence',True),table('All groups with support flags',demographic)],'This is descriptive outcome/case-mix disparity, not model fairness or proof of discrimination. Minimum 30 rows; unknown groups do not enter the screen.',False)
        put('duplicate',state(duplicate_ids,0,'Repeated record identifiers within this partition'),state(duplicate_ids,0,'Record IDs retained by preprocessing'),[bars('Duplicates: identity check versus repeated feature patterns',['Repeated record IDs','Repeated feature vectors beyond first'],{'Before':[duplicate_ids,duplicates_before],'After':[duplicate_ids,duplicates_after]},'Rows'),table('Interpretation',[{'record_ID_duplicates':duplicate_ids,'repeated_features_before':duplicates_before,'repeated_features_after':duplicates_after,'rows_removed':0}])],'No deduplication occurred. Repeated measurements/features can belong to different people. Imputation/encoding may create additional identical feature vectors; those are not automatically duplicate records.')
        put('noise',unavailable('No clean reference or adjudicated labels'),unavailable('No denoising validation / clean reference'),[bars('Ambiguous repeated feature patterns',[r['target'] for r in ambiguity],{'Conflicting patterns':[r['conflicting_patterns'] for r in ambiguity],'Affected rows':[r['rows_in_conflicting_patterns'] for r in ambiguity]},'Counts'),table('Label ambiguity diagnostics',ambiguity)],'Noise presence/absence cannot be established from these artifacts. Different outcomes at identical coarse predictors can be legitimate. No denoising or noise removal is claimed.')
        signals=max(float(shift.max()/.5),1.01 if any(r['unseen_rows'] for r in unseen) else 0.) if split!='train' else None
        order=np.argsort(shift)[::-1]
        put('encoding_domain',state(signals,1,'Any unseen category OR |numeric standardized mean shift| >0.5',split!='train'),state(signals,1,'Same shift remains after linear scaling; unknown categories are ignored',split!='train'),[bars('Mean shift from source training after imputation',[nums[i] for i in order],{'Absolute standardized shift':shift[order]},'Absolute standardized mean shift',True),bars('Unseen categorical values',[r['feature'] for r in unseen],{'Unseen rows':[r['unseen_rows'] for r in unseen]},'Rows'),table('Partition domains',domains)],'Unknown categories use the saved encoder handle_unknown=ignore. Linear scaling is not domain adaptation. States/regions are not hospitals. Training self-comparison is not an evaluation of domain shift.',False)
        stats[split]={'rows':n,'missing_input_cells':allmissing,'nonfinite_output_cells':after_missing,'record_id_duplicates':duplicate_ids,'label_counts':labelcounts,'transform_and_labels_checked':True}
    outputs=[]
    for bias,title in PRE.items():
        outputs.append(save_page('preprocessing',name,bias,{'stage':'preprocessing','dataset':name,'title':title,'kind':bias,'states':allstates[bias],'default_split':'train','methods':{'default':'Saved fixed baseline'},'default_method':'default','protocol':'No fitting or changes to patient rows. Before = model-ready rows; after = saved preprocessing.','input_checks':fingerprints}))
    write(ROOT/f'preprocessing/outputs/bias_dashboards/{name}/verification.json',{'passed':True,'partitions':stats,'pages':outputs})

def build_pre():
    icu={}
    for name in ('heart','respiratory','diabetes','hypertension'):
        pre_branch(name,ROOT/f'preprocessing/outputs/verified_baseline/{name}',rows(ROOT/f'model_ready/{name}_model_ready.csv'),original=True)
    for source in ('mimic','eicu'):icu[source]=source_icu(source)
    pre_branch('mimic',ROOT/'preprocessing/outputs/cross_dataset_baseline/mimic_to_eicu',icu['mimic'])
    pre_branch('respiratory_hospital',ROOT/'preprocessing/outputs/hospital_baseline/respiratory',rows(ROOT/'model_ready/respiratory_model_ready.csv'))
    pre_branch('eicu_hospital',ROOT/'preprocessing/outputs/cross_dataset_baseline/eicu_to_mimic',icu['eicu'],icu['mimic'])
    pre_branch('mimic_transfer',ROOT/'preprocessing/outputs/cross_dataset_baseline/mimic_to_eicu',icu['mimic'],icu['eicu'])

def ip_status(kind,m,validation,groups,dataset,split):
    if kind=='demographic':
        vals=[m['group_disparities'][a]['equalized_odds_gap'] for a in ATTRS];v=max((x for x in vals if x is not None),default=None)
        return state(v,.1,'Largest supported age/sex/ethnicity EO gap >0.10')
    if kind=='intersectional':return state(m['group_disparities']['intersection']['equalized_odds_gap'],.1,'Supported intersectional EO gap >0.10')
    if kind=='miscalibration':return state(m['ece_10_bins'],.05,'10-bin ECE >0.05 (all rows)')
    applicable=dataset in ('eicu_hospital','respiratory_hospital') and split=='test' or dataset in ('eicu_hospital','mimic_transfer') and split=='external'
    supported=groups[(groups.attribute=='domain') & groups.supported]
    spread=float(supported.brier.max()-supported.brier.min()) if len(supported)>=2 else None
    drop=validation['auroc']-m['auroc'] if validation['auroc'] is not None and m['auroc'] is not None else None
    v=max((x for x in (spread,drop) if x is not None),default=None)
    return state(v,.05,'AUROC drop from source validation >0.05 OR supported hospital Brier range >0.05',applicable)

def build_in():
    for dataset,task in JOBS:
        print('In-processing bias dashboards: '+dataset+'/'+task,flush=True)
        base=ROOT/'inprocessing/outputs'/dataset/task;decision=read(str((base/'selection_frozen.json').relative_to(ROOT)));complete=read(str((base/'COMPLETE.json').relative_to(ROOT)))
        original_hash=sha(base/'selection_frozen.json');assert original_hash==complete['selection_sha256']
        _,y,_,_=load(dataset,task);cache={}
        for method in complete['methods']:
            cache[method]={}
            for split in ('validation','test','external'):
                if split not in complete['splits']:continue
                folder=base/method;m=json.loads((folder/f'{split}_metrics.json').read_text());g=pd.read_csv(folder/f'{split}_groups.csv');bins=json.loads((folder/f'{split}_reliability.json').read_text())
                with np.load(folder/f'{split}_predictions.npz') as a:p=a['probability']
                edges=np.linspace(0,1,25);hist={f'Outcome {label}':np.histogram(p[y[split]==label],edges)[0].tolist() for label in (0,1)}
                assert sum(sum(v) for v in hist.values())==len(p)==m['rows']
                # Recompute plotting inputs independently of saved reliability bins.
                binid=np.minimum((p.astype(float)*10).astype(int),9);ece=0
                for bin in bins:
                    mask=binid==bin['bin'];assert int(mask.sum())==bin['rows']
                    if mask.any():
                        np.testing.assert_allclose(float(p[mask].astype(float).mean()),bin['mean_probability'],atol=1e-7)
                        np.testing.assert_allclose(float(y[split][mask].mean()),bin['event_rate'])
                        ece+=mask.mean()*abs(p[mask].astype(float).mean()-y[split][mask].mean())
                np.testing.assert_allclose(ece,m['ece_10_bins'],atol=1e-7)
                cache[method][split]={'m':m,'groups':g,'bins':bins,'hist':hist,'edges':edges.tolist()}
        for bias,title in IN.items():
            methods=list(dict.fromkeys([decision['selected'],'ERM']+[m for m in FAMILIES[bias] if m in cache]));states={}
            for split in ('test','external','validation'):
                if split not in cache['ERM']:continue
                states[split]={};before=cache['ERM'][split]
                for method in methods:
                    after=cache[method][split];a=before['m'];b=after['m'];panels=[]
                    if bias in ('demographic','intersectional'):
                        attrs=ATTRS if bias=='demographic' else ('intersection',)
                        panels.append(bars('Supported equalized-odds gaps',list(attrs),{'ERM baseline':[a['group_disparities'][k]['equalized_odds_gap'] for k in attrs],method:[b['group_disparities'][k]['equalized_odds_gap'] for k in attrs]},'EO gap'))
                        for attr in attrs:
                            ga=before['groups'].query('attribute == @attr').set_index('group');gb=after['groups'].query('attribute == @attr').set_index('group');assert set(ga.index)==set(gb.index)
                            for rate,flag in [('tpr','tpr_supported'),('fpr','fpr_supported')]:
                                eligible=ga[ga[flag]];labels=list(eligible.index)
                                panels.append(bars(attr+' - '+rate.upper()+' (supported groups)',labels,{'ERM baseline':ga.loc[labels,rate].tolist(),method:gb.loc[labels,rate].tolist()},rate.upper(),True))
                            panels.append(table(attr+' - all groups and support',after['groups'].query('attribute == @attr').to_dict('records')))
                    elif bias=='miscalibration':
                        traces=[{'type':'scatter','mode':'lines','name':'Perfect calibration','x':[0,1],'y':[0,1],'line':{'dash':'dash','color':'#777'}}]
                        for label,d,color in [('ERM baseline',before,'#77ade0'),(method,after,'#83b657')]:
                            bins=[z for z in d['bins'] if z['rows']]
                            traces.append({'type':'scatter','mode':'lines+markers','name':label,'x':[z['mean_probability'] for z in bins],'y':[z['event_rate'] for z in bins],'customdata':[z['rows'] for z in bins],'hovertemplate':'p=%{x:.3f}<br>observed=%{y:.3f}<br>bin n=%{customdata}<extra>%{fullData.name}</extra>','line':{'color':color}})
                        panels+=[line('Reliability curve (hover for bin counts)',traces,'Mean probability','Observed event rate'),bars('Calibration errors',['Brier','ECE (10 bins)'],{'ERM baseline':[a['brier'],a['ece_10_bins']],method:[b['brier'],b['ece_10_bins']]},'Error'),bars('Reliability-bin support',[f'{i/10:.1f}-{(i+1)/10:.1f}' for i in range(10)],{'ERM baseline':[z['rows'] for z in before['bins']],method:[z['rows'] for z in after['bins']]},'Rows')]
                        centers=(np.array(after['edges'][:-1])+np.array(after['edges'][1:]))/2
                        panels.append(bars('Comparison-model probabilities by actual outcome',centers.tolist(),after['hist'],'Rows'))
                    else:
                        g=before['groups'];h=after['groups'];g=g[(g.attribute=='domain')&g.supported].set_index('group');h=h[h.attribute=='domain'].set_index('group')
                        labels=list(g.index);panels.append(bars('Domain Brier scores (supported domains)',labels,{'ERM baseline':g.brier.tolist(),method:h.loc[labels,'brier'].tolist()},'Brier',True))
                        available=list(cache[method]);panels.append(bars('Source validation and holdout AUROC',available,{'ERM baseline':[cache['ERM'][s]['m']['auroc'] for s in available],method:[cache[method][s]['m']['auroc'] for s in available]},'AUROC'))
                        panels.append(table('Domain support and group errors',after['groups'].query("attribute == 'domain'").to_dict('records')))
                    panels.append(bars('Predictive utility tradeoffs',['Macro-F1','AUROC','Positive recall'],{'ERM baseline':[a[k] for k in ('macro_f1','auroc','positive_recall')],method:[b[k] for k in ('macro_f1','auroc','positive_recall')]},'Score'))
                    metric='ece_10_bins' if bias=='miscalibration' else 'demographic_eo_gap' if bias=='demographic' else 'intersection_eo_gap' if bias=='intersectional' else 'auroc'
                    def family_value(z):
                        mm=cache[z][split]['m']
                        if bias=='intersectional':return mm['group_disparities']['intersection']['equalized_odds_gap']
                        if bias=='demographic':return max((mm['group_disparities'][k]['equalized_odds_gap'] for k in ATTRS if mm['group_disparities'][k]['equalized_odds_gap'] is not None),default=None)
                        return mm[metric]
                    panels.append(bars('Family methods on this partition (descriptive)',methods,{metric:[family_value(z) for z in methods]},metric))
                    states[split][method]={'rows':b['rows'],'numeric_features':None,'input_features':None,'before_label':'ERM baseline','after_label':method,'before':ip_status(bias,a,cache['ERM']['validation']['m'],before['groups'],dataset,split),'after':ip_status(bias,b,cache[method]['validation']['m'],after['groups'],dataset,split),'panels':panels,'variables':{},'note':'The default is the original overall validation-selected model. Changing this dropdown only changes the display, not the saved model selection. Methods are documented adaptations. Unweighted estimates; unsupported groups excluded from disparity comparisons.','thresholds':{'ERM':a['threshold'],'comparison':b['threshold']}}
            save_page('inprocessing',dataset+'/'+task,bias,{'stage':'inprocessing','dataset':dataset+' / '+task,'title':title,'kind':bias,'states':states,'default_split':'test','methods':{m:m+(' (frozen project selection)' if m==decision['selected'] else '') for m in methods},'default_method':decision['selected'],'protocol':'Before = ERM baseline; after = chosen saved comparison model. Test/external results do not select a new winner.','selection_sha256':original_hash})
        assert sha(base/'selection_frozen.json')==original_hash

def build_latent():
    for name in ('heart','respiratory','diabetes','hypertension','mimic'):
        m=read(f'latent/outputs/{name}/manifest.json');selected=m['selection']['method'];methods=list(m['methods']);states={}
        for bias,title in LAT.items():
            panels=[];note='Validation diagnostics from the original saved manifests; no model refitting or test-label selection.'
            if bias=='information_preservation':
                targets=m['targets'];values=m['methods'][selected]['metrics']['probe'];drops=[m['uncompressed_probe'][t]['macro_f1']-values[t]['macro_f1'] for t in targets]
                before=unavailable('Uncompressed reference, not a bias test');after=state(max(drops),.02,'Validation Macro-F1 decrease >0.02 versus uncompressed reference')
                panels.append(bars('Saved validation utility guard',targets,{'Uncompressed':[m['uncompressed_probe'][t]['macro_f1'] for t in targets],selected:[values[t]['macro_f1'] for t in targets]},'Macro-F1'))
                for t in targets:panels.append(bars(t+' - all candidate probes',methods,{'Validation Macro-F1':[m['methods'][k]['metrics']['probe'][t]['macro_f1'] for k in methods]},'Macro-F1'))
            else:
                before=unavailable('No universal binary cutoff for this latent metric');after=copy.deepcopy(before)
                keys={'latent_structure':['silhouette'],'alignment':['same_row_input_latent_cka'],'domain_invariance':['domain_cluster_nmi','domain_cluster_ari']}[bias]
                for key in keys:
                    if key=='silhouette':
                        for t in m['targets']:panels.append(bars(t+' - validation label silhouette',methods,{'Silhouette':[m['methods'][k]['metrics']['probe'][t]['silhouette'] for k in methods]},'Silhouette'))
                    else:panels.append(bars(key,methods,{key:[m['methods'][k]['metrics'].get(key) for k in methods]},key))
                if bias=='alignment':note+=' Same-row input/latent CKA is not cross-hospital generalization or proof of cross-dataset clinical equivalence.'
                if bias=='domain_invariance':note+=' NMI/ARI can be NA for one source institution or absent domain labels. Low values do not establish universal domain invariance.'
            states={'validation':{'default':{'rows':m['metadata']['validation']['rows'],'input_features':None,'numeric_features':m['selection']['latent_dimensions'],'before_label':'Reference','after_label':selected,'before':before,'after':after,'panels':panels,'variables':{},'note':note}}}
            save_page('latent',name,bias,{'stage':'latent','dataset':name,'title':title,'kind':bias,'states':states,'default_split':'validation','methods':{'default':selected},'default_method':'default','protocol':'Latent metrics are individual diagnostic pages; they are not all bias-presence tests.'})
    for name in ('respiratory_hospital','eicu_hospital','mimic_transfer'):
        m=read(f'latent/outputs/hospital/{name}/manifest.json');encoder=joblib.load(ROOT/f'latent/outputs/hospital/{name}/encoder.joblib')
        diagnostic=hospital_diagnostics(name,m)
        for bias,title in LAT.items():
            panels=[];note='Frozen source PCA and saved latent matrices. Fixed diagnostic probes are fitted on source training data only; no encoder/classifier or saved selection is changed.'
            if bias=='information_preservation':
                tasks=m['targets'];drops=[diagnostic['uncompressed'][t]-diagnostic['latent'][t] for t in tasks]
                panels=[bars('Validation utility guard',tasks,{'Uncompressed':[diagnostic['uncompressed'][t] for t in tasks],'Saved PCA':[diagnostic['latent'][t] for t in tasks]},'Macro-F1')]
                after=state(max(drops),.02,'Any validation Macro-F1 decrease >0.02 relative to the uncompressed probe')
            elif bias=='latent_structure':
                panels=[bars('Validation recorded-label silhouette',m['targets'],{'Saved PCA':[diagnostic['silhouette'][t] for t in m['targets']]},'Silhouette'),table('Silhouette sample sizes',[{'target':t,'rows':diagnostic['silhouette_rows'][t]} for t in m['targets']])]
                after=unavailable('No universal silhouette cutoff establishes clinical usefulness or bias')
            elif bias=='alignment':
                panels=[bars('Input / latent CKA on the same validation rows',['Same-row CKA'],{'Saved PCA':[diagnostic['cka']]},'Linear CKA'),line('Training PCA variance retained',[{'x':list(range(1,len(encoder.explained_variance_ratio_)+1)),'y':np.cumsum(encoder.explained_variance_ratio_).tolist(),'type':'scatter','mode':'lines','name':'Training cumulative variance'}],'PCA component','Cumulative explained variance')];after=unavailable('CKA and variance retention do not prove cross-source clinical alignment')
            else:
                panels=[bars('Training-fitted clusters versus validation hospital IDs',['NMI','ARI'],{'Saved PCA':[diagnostic['domain_nmi'],diagnostic['domain_ari']]},'Association score'),table('Applicability',[{'status':diagnostic['domain_status'],'validation_rows':m['partitions']['validation']['rows']}])]
                after=unavailable(diagnostic['domain_status'])
            panel={'rows':m['partitions']['validation']['rows'],'input_features':None,'numeric_features':m['partitions']['validation']['dimensions'],'before_label':'Reference','after_label':'Source PCA','before':unavailable('No separate reference metric in this record'),'after':after,'panels':panels,'variables':{},'note':note}
            save_page('latent',name,bias,{'stage':'latent','dataset':name,'title':title,'kind':bias,'states':{'validation':{'default':panel}},'default_split':'validation','methods':{'default':'Source PCA'},'default_method':'default','protocol':'Missing measurements are shown as unavailable, not as zero or successful mitigation.'})


def hospital_diagnostics(name,manifest):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import f1_score,silhouette_score,normalized_mutual_info_score,adjusted_rand_score
    from run_latent import linear_cka,sample_indices,SEED
    print('Hospital latent diagnostic plots: '+name,flush=True)
    folder=ROOT/'latent/outputs/hospital'/name;source=ROOT/manifest['source']
    x,z,y,metadata={},{},{},{}
    for split in ('train','validation'):
        with np.load(source/f'X_{split}.npz') as a:x[split]=a['X'].astype(np.float32)
        with np.load(folder/f'Z_{split}.npz') as a:z[split]=a['Z']
        with np.load(folder/f'y_{split}.npz') as a:y[split]={t:a[t] for t in a.files}
        metadata[split]=rows(folder/f'rows_{split}.csv')
    def scores(data):
        scaler=StandardScaler().fit(data['train']);tr=scaler.transform(data['train']);va=scaler.transform(data['validation']);result={}
        for task in manifest['targets']:
            model=LogisticRegression(C=1,class_weight='balanced',max_iter=1000).fit(tr,y['train'][task])
            result[task]=float(f1_score(y['validation'][task],model.predict(va),average='macro'))
        return result
    result={'uncompressed':scores(x),'latent':scores(z),'silhouette':{},'silhouette_rows':{},'encoders_refitted':False,'selection_changed':False,'diagnostic_probes_fit_partition':'train'}
    for task in manifest['targets']:
        ix=sample_indices(y['validation'][task],2000)
        result['silhouette'][task]=float(silhouette_score(z['validation'][ix],y['validation'][task][ix]));result['silhouette_rows'][task]=len(ix)
    ix=np.random.default_rng(SEED).choice(len(z['validation']),min(2000,len(z['validation'])),replace=False)
    result['cka']=linear_cka(x['validation'][ix],z['validation'][ix]);result['cka_rows']=len(ix)
    domains=metadata['train'].meta_domain_hospitalid.astype(str)
    result.update(domain_nmi=None,domain_ari=None,domain_status='Not applicable: one source institution')
    if domains.nunique()>1:
        clustering=MiniBatchKMeans(n_clusters=min(8,domains.nunique()),random_state=SEED,n_init=5,batch_size=2048).fit(z['train'])
        pred=clustering.predict(z['validation']);truth=metadata['validation'].meta_domain_hospitalid.astype(str)
        result.update(domain_nmi=float(normalized_mutual_info_score(truth,pred)),domain_ari=float(adjusted_rand_score(truth,pred)),domain_status='Training-fitted clusters versus validation hospitals; no universal bias cutoff')
    if manifest['validation_refinement']:
        for task,value in manifest['validation_refinement'][-1]['validation_macro_f1'].items():np.testing.assert_allclose(result['latent'][task],value,atol=1e-8)
    result['all_target_utility_guards_pass']=all(result['latent'][t]>=result['uncompressed'][t]-.02 for t in manifest['targets'])
    write(folder/'dashboard_diagnostics.json',result)
    return result

def save_page(stage,dataset,bias,payload):
    outfile=ROOT/stage/'outputs/bias_dashboards'/dataset/(bias+'.json');write(outfile,payload)
    dest=ROOT/stage/'plots/biases'/dataset/(bias+'_bias_dashboard.html');dest.parent.mkdir(parents=True,exist_ok=True)
    publish_page(dest,payload)
    REGISTRY.append({'stage':stage,'dataset':dataset,'bias':bias,'title':payload['title'],'html':dest.relative_to(ROOT).as_posix(),'data':outfile.relative_to(ROOT).as_posix()})
    return outfile.relative_to(ROOT).as_posix()

def publish_page(dest,payload):
    def relative(path):return os.path.relpath(ROOT/path,dest.parent).replace('\\','/')
    stage=payload['stage'];prefix=relative(stage+'/plots/assets')
    payload=copy.deepcopy(payload)
    if stage=='latent':
        for split in payload['states'].values():
            for value in split.values():
                value['note']+=' Original silhouette and CKA comparisons use up to 2,000 validation rows; probe scores use all validation rows. See saved method metrics for sample counts.'
    raw=json.dumps(clean(payload),allow_nan=False,separators=(',',':')).replace('<','\\u003c')
    text='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TITLE</title><link rel="stylesheet" href="ASSETS/bias-dashboard.css"></head><body><header><h1 id="title"></h1><p id="subtitle"></p></header><div class="shell"><aside><a href="INDEX">All individual bias files</a><section><label for="split">1. Select partition</label><select id="split"></select><label for="method">2. Comparison model</label><select id="method"></select></section><section><label for="view">3. Select plot type</label><select id="view"><option value="summary">Bias-specific plots</option><option value="hist">Histograms</option><option value="box">Box plots</option><option value="qq">Q-Q comparison</option></select><label for="variable">4. Select variable</label><select id="variable"></select></section><section><h2>Bias status</h2><div id="before"></div><div id="after"></div><p class="small">Not detected means below this page's rule, not proven absence.</p></section><section><h2>Dataset information</h2><div id="info"></div><p class="small">All calculations use the full saved partition. No patient-level rows are embedded.</p></section><section><h2>Legend</h2><p><i class="swatch before"></i> Before / ERM baseline</p><p><i class="swatch after"></i> After / comparison model</p><a href="RULES">Detection rules and limitations</a><button id="download">Download plotted data</button></section></aside><main><section id="explanation"></section><div id="plots"></div></main></div><script id="payload" type="application/json">DATA</script><script src="ASSETS/plotly.min.js"></script><script src="ASSETS/bias-dashboard.js"></script></body></html>'''
    text=text.replace('<option value="box">','<option value="density">Binned probability density</option><option value="box">')
    text=text.replace('All calculations use the full saved partition. No patient-level rows are embedded.','Plot-specific populations and support rules are stated on this page. No patient-level rows are embedded.')
    text=text.replace('TITLE',html.escape(payload['dataset']+' | '+payload['title'])).replace('ASSETS',prefix).replace('INDEX',relative(stage+'/plots/index.html')).replace('RULES',relative('dataset_audit/BIAS_DASHBOARD_RULES.md')).replace('DATA',raw)
    dest.write_text(text,encoding='utf-8')

def publish_saved():
    entries=[]
    for stage in ('preprocessing','latent','inprocessing'):
        assets=ROOT/stage/'plots/assets';assets.mkdir(parents=True,exist_ok=True)
        library=assets/'plotly.min.js'
        if not library.exists():library.write_text(get_plotlyjs(),encoding='utf-8')
        for suffix in ('js','css'):(assets/f'bias-dashboard.{suffix}').write_text((ROOT/f'dataset_audit/dashboard_assets/bias-dashboard.{suffix}').read_text(encoding='utf-8'),encoding='utf-8')
        for data in sorted((ROOT/stage/'outputs/bias_dashboards').rglob('*.json')):
            if data.name=='verification.json':continue
            payload=json.loads(data.read_text());dataset=data.parent.relative_to(ROOT/stage/'outputs/bias_dashboards').as_posix();bias=data.stem;dest=ROOT/stage/'plots/biases'/dataset/(bias+'_bias_dashboard.html');dest.parent.mkdir(parents=True,exist_ok=True);publish_page(dest,payload)
            entries.append({'stage':stage,'dataset':dataset,'bias':bias,'title':payload['title'],'html':dest.relative_to(ROOT).as_posix(),'data':data.relative_to(ROOT).as_posix()})
    publish_indexes(entries)
    write(ROOT/'dataset_audit/bias_dashboard_manifest.json',{'completed':True,'individual_pages':len(entries),'entries':entries,'no_refitting':True,'rules':'dataset_audit/BIAS_DASHBOARD_RULES.md'})
    return entries

def publish_indexes(entries):
    def index_page(dest,title,subset):
        dest.parent.mkdir(parents=True,exist_ok=True);body=[]
        for dataset in sorted(set(r['dataset'] for r in subset)):
            body.append('<section><h2>'+html.escape(dataset)+'</h2><ul>')
            for e in subset:
                if e['dataset']==dataset:body.append('<li><a href="'+os.path.relpath(ROOT/e['html'],dest.parent).replace('\\','/')+'">'+html.escape(Path(e['html']).name)+'</a> <span>'+html.escape(e['title'])+'</span></li>')
            body.append('</ul></section>')
        dest.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>body{font:16px system-ui;margin:0;background:#eef1f8;color:#14204a}header{background:#111e4c;color:white;padding:26px 5%}main{max-width:1250px;margin:30px auto;padding:0 20px}section{background:white;border:1px solid #dce1ee;border-radius:12px;padding:22px;margin:22px 0}li{padding:12px 0;border-bottom:1px solid #e7eaf2}a{color:#234f9d}span{display:block;color:#566582;margin-top:4px}</style><header><h1>'+html.escape(title)+'</h1><p>One file per bias or latent metric. Open an individual file below.</p></header><main>'+''.join(body)+'</main></html>',encoding='utf-8')
    for stage in ('preprocessing','latent','inprocessing'):
        sub=[r for r in entries if r['stage']==stage]
        if not sub:continue
        index_page(ROOT/stage/'plots/index.html',stage.title()+' - individual bias files',sub)
        index_page(ROOT/stage/'index.html',stage.title()+' - individual bias files',sub)
        for dataset in set(r['dataset'] for r in sub):
            selection=[r for r in sub if r['dataset']==dataset]
            index_page(ROOT/stage/'plots/biases'/dataset/'index.html',dataset+' - individual files',selection)
            if stage=='inprocessing':
                index_page(ROOT/stage/'plots'/dataset/'index.html',dataset+' - individual bias files',selection)
                branch=dataset.split('/')[0]
                index_page(ROOT/stage/'plots'/branch/'index.html',branch+' - individual bias files',[r for r in sub if r['dataset'].split('/')[0]==branch])
            elif stage=='preprocessing':
                legacy=('baseline/'+dataset) if dataset in ('heart','respiratory','diabetes','hypertension','mimic') else 'hospital/'+{'respiratory_hospital':'respiratory','eicu_hospital':'eicu','mimic_transfer':'mimic'}[dataset]
                index_page(ROOT/stage/'plots'/legacy/'index.html',dataset+' - individual bias files',selection)
            elif stage=='latent':
                legacy=dataset if dataset in ('heart','respiratory','diabetes','hypertension','mimic') else 'hospital/'+dataset
                index_page(ROOT/stage/'plots'/legacy/'index.html',dataset+' - individual metric files',selection)
        if stage=='latent':index_page(ROOT/stage/'plots/hospital/index.html','Hospital latent metrics',[r for r in sub if r['dataset'] in ('respiratory_hospital','eicu_hospital','mimic_transfer')])
    lines=['# Individual bias dashboards','','Each link opens one bias/metric file, not a combined dashboard. Screens and their limits are declared in [the rules](dataset_audit/BIAS_DASHBOARD_RULES.md).','','| Stage | Dataset/task | Individual page |','|---|---|---|']
    for r in entries:lines.append(f'| {r["stage"]} | {r["dataset"]} | [{r["title"]}]({r["html"]}) |')
    (ROOT/'BIAS_DASHBOARDS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main(stage='all'):
    with threadpool_limits(limits=2):
        if stage in ('all','preprocessing'):build_pre()
        if stage in ('all','inprocessing'):build_in()
        if stage in ('all','latent'):build_latent()
    entries=publish_saved();print(f'Published {len(entries)} separate bias/metric pages.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('all','preprocessing','latent','inprocessing','publish'),default='all');args=parser.parse_args()
    if args.stage=='publish':publish_saved()
    else:main(args.stage)
