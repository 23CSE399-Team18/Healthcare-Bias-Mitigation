import unittest
import numpy as np
import torch
from models import MODELS, encode_all, fit_gcca_projection
from run_latent import linear_cka, fit_anchor_map, feature_groups
import pandas as pd


class LatentContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.rng = np.random.default_rng(17)
        self.x = self.rng.normal(size=(128, 12)).astype(np.float32)
        self.groups = [[i] for i in range(12)]
        self.views = [list(range(6)), list(range(6,12))]

    def test_cka_requires_corresponding_observations(self):
        q, _ = np.linalg.qr(self.rng.normal(size=(12,12)))
        self.assertAlmostEqual(linear_cka(self.x,self.x@q*3+4),1.,places=10)
        self.assertLess(linear_cka(self.x,self.x[self.rng.permutation(len(self.x))]),.3)
        self.assertIsNone(linear_cka(np.ones((20,2)),np.ones((20,3))))

    def test_all_models_finite_gradients_and_inductive_inference(self):
        for method, cls in MODELS.items():
            with self.subTest(method=method):
                model=cls(12,self.views,self.groups,self.x[:16])
                loss=model.objective(torch.from_numpy(self.x[:64]),True)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                proj=fit_gcca_projection(model,self.x) if method=='DGCCA' else None
                a=encode_all(model,self.x[80:100],proj,batch_size=20)
                b=encode_all(model,self.x[80:100],proj,batch_size=1)
                np.testing.assert_allclose(a,b,atol=3e-5,rtol=3e-5)
                reverse=encode_all(model,self.x[80:100][::-1].copy(),proj,batch_size=7)[::-1]
                np.testing.assert_allclose(a,reverse,atol=3e-5,rtol=3e-5)
                self.assertEqual(a.shape,(20,8))

    def test_feature_views_cover_predictors_once(self):
        data={'name':'diabetes','raw_features':['age','male','bmi','smoking_status'],
              'features':['numeric__age','numeric__male','numeric__bmi','numeric__missingindicator_bmi',
                          'categorical__smoking_status_never','categorical__smoking_status_former']}
        groups,views,_=feature_groups(data)
        self.assertEqual(sorted(sum(groups,[])),list(range(6)))
        self.assertEqual(sorted(sum(views,[])),list(range(6)))
        self.assertTrue([2,3] in groups)

    def test_anchor_does_not_read_labels_or_race(self):
        rows=[]
        for age in ['18-39','40-64','65-79','80+']:
            for sex in ['Female','Male']:
                rows += [{'sens_age_group':age,'sens_sex':sex}]*25
        frame=pd.DataFrame(rows)
        z=self.rng.normal(size=(200,8))
        a,info=fit_anchor_map(z,frame)
        frame['target']=self.rng.integers(0,2,len(frame))
        frame['sens_race_ethnicity']='changed'
        b,_=fit_anchor_map(z,frame)
        np.testing.assert_array_equal(a.coef_,b.coef_)
        self.assertEqual(info['dimensions'],4)


if __name__=='__main__':
    unittest.main(verbosity=2)
