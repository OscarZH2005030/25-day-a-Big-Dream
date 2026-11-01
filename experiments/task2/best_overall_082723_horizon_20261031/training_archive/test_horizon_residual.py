"""Verify LightGBM external initial scores compose correctly after model saving."""
import tempfile,unittest
from pathlib import Path
import numpy as np
import lightgbm as lgb

class ExternalScoreChecks(unittest.TestCase):
    def test_initial_score_is_external_for_binary_and_custom_objective(self):
        rng=np.random.default_rng(17);x=rng.normal(size=(160,4));y=(x[:,0]+.3*x[:,1]>0).astype(float);initial=.7*x[:,0]-.2;sigmoid=lambda z:np.exp(-np.logaddexp(0.,-z))
        for custom in [False,True]:
            captured=[]
            def metric(p,data):captured.append(p.copy());return 'check',0.,False
            def obj(z,data):p=sigmoid(z);return p-data.get_label(),p*(1-p)
            data=lgb.Dataset(x,label=y,init_score=initial,free_raw_data=False);model=lgb.train(dict(objective=obj if custom else 'binary',metric='None',verbosity=-1,num_threads=2,num_leaves=7,min_data_in_leaf=5,learning_rate=.05),data,num_boost_round=5,valid_sets=[data],feval=metric)
            raw=model.predict(x,raw_score=True,num_threads=2);expected=initial+raw if custom else sigmoid(initial+raw);np.testing.assert_allclose(captured[-1],expected,atol=1e-12)
            with tempfile.TemporaryDirectory() as folder:
                p=Path(folder)/'residual.txt';model.save_model(str(p));loaded=lgb.Booster(model_file=str(p));np.testing.assert_array_equal(loaded.predict(x,raw_score=True,num_threads=2),raw)

if __name__=='__main__':unittest.main()
