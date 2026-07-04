"""Publish verified hospital CSVs, diagrams and explicit navigation."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import joblib,nbformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
BRANCHES=('respiratory_hospital','eicu_hospital','mimic_transfer')
titles={'respiratory_hospital':'eICU respiratory mortality: held-out hospitals','eicu_hospital':'eICU four diagnoses: held-out hospitals + external MIMIC','mimic_transfer':'MIMIC four diagnoses: patient holdout + external eICU'}

def rows(path):
    cols=pd.read_csv(path,nrows=0).columns
    return pd.read_csv(path,dtype={c:'string' for c in cols if c.startswith('meta_')})

def main():
    check=json.loads((ROOT/'latent/outputs/hospital/independent_validation.json').read_text());assert check['passed']
    sections=[];guide=['# Hospital preprocessing and latent outputs','','The three survey benchmarks remain separate. Use these branches for ICU evaluation. Training and validation/test hospitals are disjoint within eICU; MIMIC has one source institution. External source-to-target transformations are frozen.','', '| Branch | Train / validation / test hospitals | Input width | Latent width | Preprocessing outputs | Latent outputs |','|---|---|---:|---:|---|---|']
    exports={}
    for name in BRANCHES:
        folder=ROOT/'latent/outputs/hospital'/name
        manifest=json.loads((folder/'manifest.json').read_text());source=ROOT/manifest['source'];splits=list(manifest['partitions'])
        pre=ROOT/'preprocessing/outputs/hospital_baseline'/('respiratory' if name=='respiratory_hospital' else 'eicu' if name=='eicu_hospital' else 'mimic')
        pre.mkdir(parents=True,exist_ok=True)
        features=json.loads((source/'feature_names.json').read_text());features=features.get('features',features.get('output_features'))
        metadata={s:rows(source/f'rows_{s}.csv') for s in splits}
        exports[name]={};prelinks=[];latentlinks=[]
        for split in splits:
            with np.load(source/f'X_{split}.npz') as a:x=a['X']
            prefix='respiratory' if name=='respiratory_hospital' else 'eicu' if name=='eicu_hospital' else 'mimic'
            target=pre/f'{prefix}_{split}_preprocessed.csv'
            if name=='mimic_transfer' and split!='external':target=ROOT/'preprocessing/outputs/verified_baseline/mimic'/target.name
            elif name!='respiratory_hospital':pd.concat([pd.DataFrame(x,columns=features),metadata[split]],axis=1).to_csv(target,index=False)
            read=rows(target)
            np.testing.assert_allclose(read[features],x,atol=1e-6,rtol=1e-6)
            pd.testing.assert_frame_equal(read[list(metadata[split])],metadata[split],check_dtype=False)
            with np.load(source/f'y_{split}.npz') as a:
                for task in a.files:np.testing.assert_array_equal(read['target' if name=='respiratory_hospital' else 'target_'+task],a[task])
            exports[name][split]={'csv':target.relative_to(ROOT).as_posix(),'rows':len(x),'features':len(features),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
            prelinks.append(f'[{split}](../../{target.relative_to(ROOT).as_posix()})')
            latentlinks.append(f'[{split}]({name}/{name}_{split}_latent.csv)')
        plot=ROOT/'preprocessing/plots/hospital'/pre.name/'preprocessing_checks.png';plot.parent.mkdir(parents=True,exist_ok=True)
        fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
        counts=[len(metadata[s]) for s in splits];hosp=[int(metadata[s].meta_domain_hospitalid.nunique()) for s in splits]
        for ax,values,title in [(axes[0,0],counts,'Rows in each partition'),(axes[0,1],hosp,'Hospitals in each partition')]:
            bars=ax.bar(splits,values,color=['#387aab','#d29a37','#20876c','#885aa2'][:len(splits)])
            ax.bar_label(bars,padding=3,fontsize=9);ax.set(title=title,ylim=(0,max(values)*1.18));ax.tick_params(axis='x',rotation=10)
        tasks=manifest['targets'];pos=np.arange(len(tasks));width=.8/len(splits)
        for i,split in enumerate(splits):
            with np.load(source/f'y_{split}.npz') as labels:rates=[float(labels[t].mean()) for t in tasks]
            axes[1,0].bar(pos-.4+width*(i+.5),rates,width,label=split)
        axes[1,0].set_xticks(pos,['Mortality' if name=='respiratory_hospital' else t.title() for t in tasks]);axes[1,0].set(title='Recorded target prevalence',ylim=(0,1));axes[1,0].legend(fontsize=8)
        axes[1,1].axis('off');axes[1,1].text(.03,.95,'Verified preprocessing\n\n'+f'{len(features)} numeric output columns\nTraining-only fitted imputation and scaling\nFinite predictors; labels and IDs aligned\nComplete CSV values checked against saved matrices\n\n'+('Disjoint hospitals and patients across internal splits' if name!='mimic_transfer' else 'Disjoint patients within one source institution')+'\nExternal records never fit source transformations',va='top',fontsize=11,linespacing=1.6)
        fig.suptitle(titles[name],fontsize=14);fig.savefig(plot,dpi=140,bbox_inches='tight');plt.close(fig)
        encoder=joblib.load(folder/'encoder.joblib')
        fig,axes=plt.subplots(1,3,figsize=(16,4.5),constrained_layout=True)
        cumulative=np.cumsum(encoder.explained_variance_ratio_);axes[0].plot(np.arange(1,len(cumulative)+1),cumulative,color='#20876c');axes[0].set(xlabel='Retained PCA components',ylabel='Cumulative training variance',ylim=(0,1.01),title=f'{len(features)} inputs to {len(cumulative)} latent components')
        for split in splits:
            with np.load(folder/f'Z_{split}.npz') as a:z=a['Z']
            axes[1].hist(z[:,0],bins=60,density=True,histtype='step',label=split)
            axes[2].plot(np.arange(1,z.shape[1]+1),z.mean(0),label=split,linewidth=1)
        axes[1].set(title='First latent component distribution',xlabel='Standardized component 1',ylabel='Density');axes[1].legend(fontsize=8)
        axes[2].set(title='Partition mean by component',xlabel='Latent component',ylabel='Mean standardized component');axes[2].legend(fontsize=8)
        fig.suptitle(titles[name]+'\nPCA and latent scaling fitted to source training rows only',fontsize=12);(ROOT/'latent/plots/hospital'/name).mkdir(parents=True,exist_ok=True);fig.savefig(ROOT/'latent/plots/hospital'/name/'latent_checks.png',dpi=140,bbox_inches='tight');plt.close(fig)
        numbers='/'.join(str(hosp[i]) for i in range(3))
        guide.append(f'| {name} | {numbers} | {len(features)} | {len(cumulative)} | '+', '.join(prelinks)+' | '+', '.join(latentlinks)+' |')
        prepath=plot.relative_to(ROOT).as_posix()
        sections.append(f'<section><h2>{titles[name]}</h2><img src="../../{prepath}" alt="Preprocessing verification"><img src="{name}/latent_checks.png" alt="Training-fitted latent diagnostics"></section>')
    (ROOT/'preprocessing/outputs/hospital_baseline/export_validation.json').write_text(json.dumps({'passed':True,'exports':exports},indent=2),encoding='utf-8')
    import sys
    sys.path.insert(0,str(ROOT/'dataset_audit'))
    from publish_layout import main as publish_layout
    publish_layout()

if __name__=='__main__':main()
