"""Build readable comparisons and complete selected-prediction exports."""
from pathlib import Path
import json, html, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
from run_inprocessing import ROOT,HERE,OUT,JOBS,load,sha,write

def chart(dataset,task,decision):
    folder=OUT/dataset/task; selected=decision['selected']
    plotfolder=HERE/'plots'/dataset/task;plotfolder.mkdir(parents=True,exist_ok=True)
    splits=['test']+(['external'] if (folder/selected/'external_metrics.json').exists() else [])
    for split in splits:
        metrics={m:json.loads((folder/m/f'{split}_metrics.json').read_text()) for m in ('ERM',selected)}
        before=metrics['ERM'];after=metrics[selected]
        fig,axes=plt.subplots(2,3,figsize=(18,10),constrained_layout=True)
        colors=['#66788a','#20876c'];labels=['ERM baseline',selected.replace('_',' ')]
        x=np.arange(3)
        for offset,m,label,color in zip((-.18,.18),(before,after),labels,colors):
            values=[m[k] for k in ('macro_f1','auroc','positive_recall')]
            axes[0,0].bar(x+offset,values,.36,label=label,color=color)
        axes[0,0].set_xticks(x,['Macro-F1','AUROC','Recall']);axes[0,0].set(ylim=(0,1),title='Predictive utility');axes[0,0].legend(fontsize=8)
        attributes=['sens_sex','sens_age_group','sens_race_ethnicity','intersection']
        for offset,m,color in zip((-.18,.18),(before,after),colors):
            values=[m['group_disparities'][a]['equalized_odds_gap'] for a in attributes]
            for j,v in enumerate(values):
                if v is not None:axes[0,1].bar(j+offset,v,.36,color=color)
                else:axes[0,1].text(j+offset,.02,'NA',rotation=90,fontsize=8)
        axes[0,1].set_xticks(range(4),['Sex','Age','Ethnicity','Intersection']);axes[0,1].set(ylim=(0,1),title='Equalized-odds gaps (lower is better)')
        axes[0,1].tick_params(axis='x',rotation=20)
        for m,label,color in zip(('ERM',selected),labels,colors):
            bins=json.loads((folder/m/f'{split}_reliability.json').read_text());bins=[b for b in bins if b['rows']]
            axes[0,2].plot([b['mean_probability'] for b in bins],[b['event_rate'] for b in bins],marker='o',label=label,color=color)
        axes[0,2].plot([0,1],[0,1],'--',color='gray');axes[0,2].set(xlim=(0,1),ylim=(0,1),title='Reliability: nonempty probability bins',xlabel='Mean predicted probability',ylabel='Observed event rate')
        for offset,m,color in zip((-.18,.18),(before,after),colors):axes[1,0].bar(np.arange(2)+offset,[m['brier'],m['ece_10_bins']],.36,color=color)
        axes[1,0].set_xticks([0,1],['Brier','ECE (10 bins)']);axes[1,0].set(title='Calibration error (lower is better)',ylim=(0,max(before['brier'],after['brier'],before['ece_10_bins'],after['ece_10_bins'])*1.25+.01))
        a=pd.read_csv(folder/'ERM'/f'{split}_groups.csv');b=pd.read_csv(folder/selected/f'{split}_groups.csv')
        for ax,attribute,title in [(axes[1,1],'intersection','Intersectional TPR: up to 10 lowest ERM groups'),(axes[1,2],'domain','Domain Brier: up to 10 highest ERM groups')]:
            field='tpr' if attribute=='intersection' else 'brier'
            subset=a[(a.attribute==attribute)&(a.tpr_supported if attribute=='intersection' else a.supported)]
            subset=subset.sort_values(field,ascending=attribute=='intersection').head(10)
            if subset.empty:
                ax.axis('off');ax.text(.5,.5,'No supported domain/group comparison',ha='center',transform=ax.transAxes);continue
            values=b[b.attribute==attribute].set_index('group').loc[subset.group,field].to_numpy()
            pos=np.arange(len(subset));ax.barh(pos-.18,subset[field],.36,color=colors[0]);ax.barh(pos+.18,values,.36,color=colors[1])
            ax.set_yticks(pos,subset.group);ax.tick_params(axis='y',labelsize=7);ax.set(title=title,xlabel=field)
            if attribute=='intersection':ax.set_xlim(0,1)
        title=f'{dataset} / {task} | {split}: '+('external database evaluation' if split=='external' else ('held-out hospitals' if dataset in ('eicu_hospital','respiratory_hospital') else 'patient/record holdout'))
        fig.suptitle(title+'\nSelection frozen on validation; support rules apply; unweighted estimates',fontsize=14)
        fig.savefig(plotfolder/f'{split}_comparison.png',dpi=140,bbox_inches='tight');plt.close(fig)
    values=decision['validation']; methods=list(values)
    fig,axes=plt.subplots(1,4,figsize=(17,max(6,len(methods)*.32)),constrained_layout=True)
    for ax,key,title in zip(axes,['macro_f1','auroc','max_equalized_odds_gap','brier'],['Validation Macro-F1','Validation AUROC','Validation maximum EO gap','Validation Brier']):
        vals=[values[m][key] for m in methods]
        ax.barh(range(len(methods)),[v if v is not None else np.nan for v in vals],color=['#20876c' if m==selected else '#66788a' for m in methods])
        ax.set_yticks(range(len(methods)),[m.replace('_',' ') for m in methods] if ax==axes[0] else []);ax.set_title(title,fontsize=10);ax.invert_yaxis()
    fig.suptitle(f'{dataset} / {task}: full method comparison; green = validation-selected')
    fig.savefig(plotfolder/'method_comparison.png',dpi=140,bbox_inches='tight');plt.close(fig)

def main(plots_only=False):
    cards=[];records=[];summary=[];external_summary=[];exports={}
    for dataset,task in JOBS:
        folder=OUT/dataset/task
        complete=json.loads((folder/'COMPLETE.json').read_text());assert complete['passed']
        decision=json.loads((folder/'selection_frozen.json').read_text());assert sha(folder/'selection_frozen.json')==complete['selection_sha256']
        chart(dataset,task,decision)
        selected=decision['selected'];test=json.loads((folder/selected/'test_metrics.json').read_text())
        base=json.loads((folder/'ERM/test_metrics.json').read_text())
        summary.append((dataset,task,selected,test['macro_f1'],test['auroc'],test['max_equalized_odds_gap'],base['max_equalized_odds_gap']))
        if 'external' in complete['splits']:
            ext=json.loads((folder/selected/'external_metrics.json').read_text())
            external_summary.append((dataset,task,selected,ext['macro_f1'],ext['auroc'],ext['max_equalized_odds_gap']))
        for method in complete['methods']:
            for split in complete['splits']:
                if split=='train':continue
                m=json.loads((folder/method/f'{split}_metrics.json').read_text())
                records.append({'dataset':dataset,'task':task,'method':method,'split':split,'selected':method==selected,**{k:v for k,v in m.items() if k!='group_disparities'}})
    if plots_only:
        print('Refreshed all result plots without changing models or exports.',flush=True)
        return
    for dataset in sorted({d for d,t in JOBS}):
        tasks=[t for d,t in JOBS if d==dataset];x,y,rows,_=load(dataset,tasks[0]);out=OUT/dataset/'selected';out.mkdir(exist_ok=True);exports[dataset]={}
        for split in x:
            frame=pd.concat([pd.DataFrame(x[split],columns=[f'z{i+1:02}' for i in range(x[split].shape[1])]),rows[split]],axis=1)
            if dataset in ('heart','diabetes','hypertension','respiratory_hospital'):frame['target']=y[split]
            for task in tasks:
                folder=OUT/dataset/task;decision=json.loads((folder/'selection_frozen.json').read_text());method=decision['selected'];threshold=decision['validation'][method]['threshold']
                with np.load(folder/method/f'{split}_predictions.npz') as a:p=a['probability']
                frame[f'probability_{task}']=p;frame[f'prediction_{task}']=(p>=threshold).astype(np.int8)
            frame['meta_evaluation_split']=split
            path=out/f'{dataset}_{split}_predictions.csv';frame.to_csv(path,index=False)
            exports[dataset][split]={'csv':str(path.relative_to(ROOT)),'rows':len(frame),'sha256':sha(path)}
    pd.DataFrame(records).to_csv(OUT/'method_comparison.csv',index=False)
    write(OUT/'manifest.json',{'completed':True,'tasks':len(JOBS),'selected_exports':exports,'test_used_for_selection':False,'external_used_for_selection':False})
    lines=['# In-processing results','','Training, validation selection and locked test/external evaluation are complete. Use the [plots](../plots/index.html) and [all method metrics](method_comparison.csv).','', '| Dataset branch | Task | Selected method | Test Macro-F1 | Test AUROC | Test max EO gap | ERM max EO gap |','|---|---|---|---:|---:|---:|---:|']
    for values in summary:lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else ('NA' if v is None else str(v)) for v in values)+' |')
    lines+=['','## Frozen external-database evaluation','','| Source branch | Task | Selected method | External Macro-F1 | External AUROC | External max EO gap |','|---|---|---|---:|---:|---:|']
    for values in external_summary:lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else ('NA' if v is None else str(v)) for v in values)+' |')
    lines+=['','Selection uses validation only, not the test values in this table. A selected method can perform worse than ERM on the test set. The full method tables and support counts are retained.','',
        '## Cross-hospital scope','',
        '- Respiratory mortality uses hospital-disjoint preprocessing and a newly fitted latent encoder: 137 training, 34 validation and 29 test hospitals. Cross-partition patients are excluded and recorded.',
        '- eICU recorded-diagnosis tasks use 141 training, 34 validation and 31 test hospitals, with patient purging across partitions. The frozen eICU models are also evaluated on MIMIC.',
        '- MIMIC models use a single source institution and are evaluated externally on eICU. Its internal test is patient-held-out. Anonymous eICU hospital IDs do not establish institutional non-overlap with BIDMC across databases.',
        '- Survey heart, diabetes and hypertension benchmarks have no hospital IDs. Their results are not cross-hospital evidence.',
        '- Mortality and recorded respiratory diagnosis are different targets. MIMIC discharge coding and eICU ICU/history labels have ascertainment differences. These are retrospective recorded-label evaluations, not future disease prediction.','',
        '## Files and interpretation','',
        'Selected complete CSVs are in `outputs/<branch>/selected/`. They contain the original latent columns, labels, metadata and task-specific probabilities/predictions. In-processing trains classifiers; these files are prediction results, not newly cleaned patient data. Each task directory retains every fitted model, scaler, training history, frozen selection record, probabilities, group counts and reliability bins.',
        'The methods are documented adaptations from the supplied image. FairCare graph learning is unavailable in flat latent inputs; the tabular adversarial model must not be called an exact FairCare replication. FOMO is bounded weight search, Fish is an ERM-plus-first-order variant, and EQRM uses empirical quantiles. Methods needing multiple source domains are NA for single-domain MIMIC and the no-domain diabetes survey.',
        'Group rates are unweighted. Raw counts/rates and support flags are retained in group tables; unsupported groups are excluded from disparity calculations and comparison plots. Undefined rates or insufficient comparisons remain NA. Age and sex are present in source features as documented upstream. The group gap metric is descriptive, not a causal discrimination estimate. Single-seed comparisons and fixed budgets do not establish clinical safety, statistical significance or universal generalization. Independent replication and uncertainty analysis remain necessary before publication or deployment.','',
        '## Reproduce','', '`python inprocessing/code/run_inprocessing.py` trains unfinished tasks. Completed tasks are preserved. `python inprocessing/code/make_report.py` refreshes plots and prediction CSVs. `python inprocessing/code/validate.py` checks saved results.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import sys
    sys.path.insert(0,str(ROOT/'dataset_audit'))
    from publish_layout import main as publish_layout
    publish_layout()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plots-only',action='store_true');args=parser.parse_args()
    main(plots_only=args.plots_only)
