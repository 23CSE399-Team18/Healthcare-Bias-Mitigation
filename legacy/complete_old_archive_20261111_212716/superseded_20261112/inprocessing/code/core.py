"""Binary prediction and support-aware audit metrics; no test-based fitting."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score, log_loss

ATTRS=('sens_sex','sens_age_group','sens_race_ethnicity')
METHODS=['ERM','AdversarialFair_tabular','DANN_source_domains','GroupDRO_sex_label','FairERM_softEO',
         'Intersectional_adversary','Constrained_softEO','GroupDRO_intersection_label','AvUC_deterministic',
         'CRL_history','UWG_Brier','Brier_regularized','Fishr_last_layer','VREx','EQRM_empirical','Fish_first_order']
DOMAIN_METHODS={'DANN_source_domains','Fishr_last_layer','VREx','EQRM_empirical','Fish_first_order'}

def grouping(rows):
    groups={a:rows[a].fillna('Unknown').astype(str).to_numpy() for a in ATTRS}
    groups['intersection']=np.array(['|'.join(x) for x in zip(*(groups[a] for a in ATTRS))])
    col=next((c for c in ('meta_domain_hospitalid','meta_domain_state','meta_domain_region') if c in rows),None)
    if col:groups['domain']=rows[col].fillna('Unknown').astype(str).to_numpy()
    return groups

def known(value):
    return not any(v in ('Unknown','nan','<NA>','None','') for v in str(value).split('|'))

def code_groups(groups,y):
    result={};levels={}
    expanded=dict(groups)
    expanded['sex_label']=np.array([f'{s}|{int(t)}' for s,t in zip(groups[ATTRS[0]],y)])
    expanded['intersection_label']=np.array([f'{s}|{int(t)}' for s,t in zip(groups['intersection'],y)])
    for key,values in expanded.items():
        counts=pd.Series(values).value_counts()
        labels=sorted(s for s,n in counts.items() if n>=50 and known(s))
        mapping={v:i for i,v in enumerate(labels)}
        result[key]=np.array([mapping.get(v,-1) for v in values],dtype=np.int64);levels[key]=labels
    return result,levels

def reliability(y,p):
    bins=np.minimum((p*10).astype(int),9); records=[];ece=0.
    for b in range(10):
        mask=bins==b;count=int(mask.sum())
        avg=float(p[mask].mean()) if count else None; rate=float(y[mask].mean()) if count else None
        if count:ece+=count/len(y)*abs(avg-rate)
        records.append({'bin':b,'lower':b/10,'upper':(b+1)/10,'rows':count,'mean_probability':avg,'event_rate':rate})
    return float(ece),records

def evaluate(y,p,groups,threshold):
    y=np.asarray(y,int);p=np.asarray(p,float);pred=p>=threshold
    assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    ece,bins=reliability(y,p)
    metrics={'rows':len(y),'positives':int(y.sum()),'threshold':float(threshold),
        'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
        'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'auroc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
        'average_precision':float(average_precision_score(y,p)) if y.sum() else None,
        'brier':float(np.mean((p-y)**2)),'log_loss':float(log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1])),
        'ece_10_bins':ece,'positive_recall':float(pred[y==1].mean()) if y.sum() else None}
    table=[];disparities={}
    for attribute,values in groups.items():
        collection=[]
        for label in sorted(set(values)):
            mask=values==label;n=int(mask.sum());n1=int(y[mask].sum());n0=n-n1
            tp=int(np.sum(pred[mask] & (y[mask]==1)));fp=int(np.sum(pred[mask] & (y[mask]==0)))
            item={'attribute':attribute,'group':label,'rows':n,'positives':n1,'negatives':n0,'tp':tp,'fp':fp,
                  'selection_rate':float(pred[mask].mean()),'tpr':tp/n1 if n1 else None,'fpr':fp/n0 if n0 else None,
                  'brier':float(np.mean((p[mask]-y[mask])**2)),'known':known(label),
                  'supported':n>=30 and known(label),'tpr_supported':n>=30 and n1>=10 and known(label),
                  'fpr_supported':n>=30 and n0>=10 and known(label)}
            table.append(item);collection.append(item)
        def gap(field,flag):
            vals=[a[field] for a in collection if a[flag]]
            return float(max(vals)-min(vals)) if len(vals)>=2 else None
        tpr=gap('tpr','tpr_supported');fpr=gap('fpr','fpr_supported')
        disparities[attribute]={'selection_rate_gap':gap('selection_rate','supported'),'tpr_gap':tpr,'fpr_gap':fpr,
                               'equalized_odds_gap':max(tpr,fpr) if tpr is not None and fpr is not None else None}
    gaps=[v['equalized_odds_gap'] for k,v in disparities.items() if k!='domain' and v['equalized_odds_gap'] is not None]
    metrics['max_equalized_odds_gap']=max(gaps) if gaps else None
    metrics['group_disparities']=disparities
    return metrics,table,bins

def threshold_for(y,p):
    choices=np.arange(.05,.951,.025)
    return float(max(choices,key=lambda t:(f1_score(y,p>=t,average='macro',zero_division=0),-abs(t-.5))))

class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):return x.view_as(x)
    @staticmethod
    def backward(ctx,gradient):return -gradient

class Network(nn.Module):
    def __init__(self,dimension,head_sizes):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(dimension,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU())
        self.output=nn.Linear(16,1)
        self.adversaries=nn.ModuleDict({str(i):nn.Linear(16,size) for i,size in enumerate(head_sizes)})
    def forward(self,x):
        h=self.body(x);return self.output(h).squeeze(-1),h

@torch.no_grad()
def probabilities(model,x):
    model.eval();result=[]
    for start in range(0,len(x),4096):result.append(torch.sigmoid(model(torch.from_numpy(x[start:start+4096]))[0]).numpy())
    return np.concatenate(result)

def group_means(values,codes,minimum=4):
    labels=[int(g) for g in torch.unique(codes) if g>=0 and (codes==g).sum()>=minimum]
    return labels,torch.stack([values[codes==g].mean(0) for g in labels]) if labels else None

def soft_eo(p,y,codes):
    penalties=[]
    for groups in codes:
        for label in (0,1):
            mask=y==label
            _,means=group_means(p[mask],groups[mask])
            if means is not None and len(means)>1:penalties.append(means.var(unbiased=False))
    return torch.stack(penalties).mean() if penalties else p.sum()*0

def loss_for(method,model,x,y,codes,state,indices,epoch):
    logits,h=model(x);p=torch.sigmoid(logits);bce=F.binary_cross_entropy_with_logits(logits,y,reduction='none');base=bce.mean()
    if method in ('AdversarialFair_tabular','Intersectional_adversary','DANN_source_domains'):
        keys=ATTRS if method=='AdversarialFair_tabular' else (('intersection',) if method=='Intersectional_adversary' else ('domain',))
        extra=[]
        for i,key in enumerate(keys):
            mask=codes[key]>=0
            if mask.any():extra.append(F.cross_entropy(model.adversaries[str(i)](Reverse.apply(h[mask])),codes[key][mask]))
        return base+.1*torch.stack(extra).mean() if extra else base
    if method.startswith('GroupDRO'):
        key='sex_label' if method=='GroupDRO_sex_label' else 'intersection_label'
        labels,risks=group_means(bce,codes[key])
        if risks is None:return base
        q=state['q'];q[labels]*=torch.exp(.01*risks.detach());q/=q.sum()
        weights=q[labels];weights=weights/weights.sum()
        return .1*base+.9*(weights*risks).sum()
    if method in ('FairERM_softEO','Constrained_softEO'):
        penalty=soft_eo(p,y,[codes[a] for a in ATTRS]+([codes['intersection']] if method=='Constrained_softEO' else []))
        if method=='Constrained_softEO':
            state['dual']=max(0.,min(100.,state['dual']+.1*(penalty.detach().item()-.0025)))
            return base+state['dual']*(penalty-.0025)
        return base+10*penalty
    if method=='Brier_regularized':return base+((p-y)**2).mean()
    if method=='UWG_Brier':return (bce*(.05+2*(p.detach()-y)**2)).mean()
    if method=='CRL_history':
        confidence=torch.maximum(p,1-p);hist=state['correct'][indices]/max(epoch,1)
        other=torch.roll(torch.arange(len(y)),1);sign=torch.sign(hist-hist[other]);mask=sign!=0
        rank=F.relu(.05-sign*(confidence-confidence[other]))[mask].mean() if mask.any() else base*0
        state['correct'][indices]+=(p.detach().ge(.5)==y.bool()).float()
        return base+.1*rank
    if method=='AvUC_deterministic':
        entropy=-(p*torch.log(p+1e-7)+(1-p)*torch.log(1-p+1e-7));confidence=torch.maximum(p,1-p)
        if epoch==0:
            state['ent_sum']+=entropy.detach().sum().item();state['ent_n']+=len(y);return base
        accurate=(p.detach()>=.5)==y.bool();certain=entropy.detach()<=state['ent_sum']/state['ent_n'];u=torch.tanh(entropy)
        good=(confidence*(1-u)*(accurate&certain)+(1-confidence)*u*(~accurate&~certain)).sum()
        bad=(confidence*u*(accurate&~certain)+(1-confidence)*(1-u)*(~accurate&certain)).sum()
        return base+.1*torch.log1p(bad/(good+1e-7))
    if method in ('VREx','EQRM_empirical','Fishr_last_layer'):
        labels,risks=group_means(bce,codes['domain'])
        if risks is None or len(risks)<2:return base
        if method=='VREx':return risks.mean()+10*risks.var(unbiased=False)
        if method=='EQRM_empirical':return torch.quantile(risks,.9)
        grads=(p-y)[:,None]*torch.cat([h,torch.ones(len(h),1)],dim=1)
        variances=torch.stack([grads[codes['domain']==g].var(0,unbiased=False) for g in labels])
        return base+10*variances.var(0,unbiased=False).mean()
    return base
