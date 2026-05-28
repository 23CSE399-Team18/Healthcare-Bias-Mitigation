import unittest
import numpy as np
import torch
from core import *

class Contracts(unittest.TestCase):
    def test_gradient_reversal_has_unit_negative_derivative(self):
        x=torch.tensor([1.,2.,3.],requires_grad=True)
        (Reverse.apply(x)*torch.tensor([2.,3.,4.])).sum().backward()
        torch.testing.assert_close(x.grad,torch.tensor([-2.,-3.,-4.]))

    def test_fishr_analytic_final_layer_gradient_matches_autograd(self):
        torch.manual_seed(9);model=Network(5,[]);x=torch.randn(7,5);y=torch.arange(7).remainder(2).float()
        logits,h=model(x);analytic=(torch.sigmoid(logits)-y)[:,None]*torch.cat([h,torch.ones(len(h),1)],1)
        expected=[]
        for i in range(len(x)):
            grads=torch.autograd.grad(torch.nn.functional.binary_cross_entropy_with_logits(logits[i],y[i]),(model.output.weight,model.output.bias),retain_graph=True)
            expected.append(torch.cat([g.reshape(-1) for g in grads]))
        torch.testing.assert_close(analytic,torch.stack(expected))

    def test_soft_eo_conditions_on_true_label_and_excludes_unknown(self):
        y=torch.tensor([0.]*16+[1.]*16)
        groups=torch.tensor(([0]*8+[1]*8)*2)
        p=torch.where(y==1,.8,.2)
        self.assertAlmostEqual(soft_eo(p,y,[groups]).item(),0.)
        p[groups==1]+=.1
        self.assertAlmostEqual(soft_eo(p,y,[groups]).item(),.0025,places=7)
        groups[:]=-1
        self.assertAlmostEqual(soft_eo(p,y,[groups]).item(),0.)

    def test_ece_includes_endpoints_and_empty_bins_are_undefined(self):
        ece,bins=reliability(np.array([0,1,0,1]),np.array([0.,1.,.2,.8]))
        self.assertAlmostEqual(ece,.1)
        self.assertEqual(sum(b['rows'] for b in bins),4)
        self.assertEqual(bins[9]['rows'],1)
        self.assertIsNone(bins[5]['mean_probability'])

    def test_losses_are_finite_and_backpropagate(self):
        torch.set_num_threads(1);torch.manual_seed(17)
        x=torch.randn(128,8);y=(torch.arange(128)%2).float()
        codes={k:(torch.arange(128)//2)%2 for k in (*ATTRS,'intersection','domain','sex_label','intersection_label')}
        for method in METHODS:
            heads=[2]*3 if method=='AdversarialFair_tabular' else ([2] if method in ('Intersectional_adversary','DANN_source_domains') else [])
            model=Network(8,heads);state={'q':torch.ones(2)/2,'dual':0.,'correct':torch.ones(128),'ent_sum':60.,'ent_n':128}
            loss=loss_for(method,model,x,y,codes,state,np.arange(128),1);loss.backward()
            self.assertTrue(torch.isfinite(loss),method)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),method)
    def test_metric_support_and_known_rates(self):
        y=np.tile([0,1],80);groups={'sex':np.array(['A']*80+['B']*80)}
        p=np.r_[np.tile([.1,.9],40),np.tile([.9,.1],40)]
        m,table,bins=evaluate(y,p,groups,.5)
        self.assertEqual(m['group_disparities']['sex']['equalized_odds_gap'],1.)
        self.assertEqual(sum(b['rows'] for b in bins),len(y))
        tiny,_,_=evaluate(y[:6],p[:6],{'sex':np.array(['A']*3+['B']*3)},.5)
        self.assertIsNone(tiny['max_equalized_odds_gap'])
    def test_inference_is_independent_of_other_query_rows(self):
        rng=np.random.default_rng(0);x=rng.normal(size=(40,8)).astype(np.float32);model=Network(8,[])
        a=probabilities(model,x)
        b=np.concatenate([probabilities(model,row[None,:]) for row in x])
        np.testing.assert_allclose(a,b,atol=1e-7)
    def test_sensitive_columns_never_become_predictors(self):
        rows=pd.DataFrame({'sens_sex':['Female']*60,'sens_age_group':['40-64']*60,'sens_race_ethnicity':['Unknown']*60})
        codes,levels=code_groups(grouping(rows),np.zeros(60))
        self.assertEqual(levels['intersection'],[])
        self.assertTrue((codes['intersection']==-1).all())

if __name__=='__main__':unittest.main(verbosity=2)
