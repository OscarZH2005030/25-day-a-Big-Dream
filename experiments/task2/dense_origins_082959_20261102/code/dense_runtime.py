"""Label-free portable ongoing inference, with explicit retained parent chain."""
from pathlib import Path
import numpy as np
import lightgbm as lgb
SEEDS=[20261030,20261031,20261101]

class Portable:
    def __init__(self,root):
        self.root=Path(root);self.models={};self.base=self.root.parent/'best_overall_082639_jaccard_20261031/models';self.prior=self.root.parent/'best_overall_082723_horizon_20261031/models'

    def preload(self,panels):
        for panel in panels:
            for seed in SEEDS:
                self.models[panel,seed,'base']=lgb.Booster(model_file=str(self.base/f'final_{seed}/{panel}.txt'))
                for h in range(5,31,5):
                    for kind,folder in [('prior','prior_residuals'),('new','models')]:
                        self.models[panel,seed,kind,h]=lgb.Booster(model_file=str((self.prior if kind=='prior' else self.root/'models')/f'final_{seed}/{panel}-h{h}.txt'))

    def predict(self,panel,x274,r127_probability,alpha=1.):
        x=np.asarray(x274,dtype='float32');parent=np.asarray(r127_probability)
        assert x.ndim==2 and x.shape[1]==274 and parent.shape==(len(x),6)
        assert alpha in [0.,.5,1.]
        if (panel,SEEDS[0],'base') not in self.models:self.preload([panel])
        ps=[]
        for seed in SEEDS:
            z=np.empty((len(x),6),dtype='float64')
            for i,h in enumerate(range(5,31,5)):
                xx=np.column_stack([x,np.full(len(x),h)])
                z[:,i]=self.models[panel,seed,'base'].predict(xx,raw_score=True,num_threads=4)+self.models[panel,seed,'prior',h].predict(xx,raw_score=True,num_threads=4)
                if alpha:
                    tx=np.column_stack([x,np.full((len(x),30),np.nan),np.full(len(x),h)]).astype('float32')
                    z[:,i]+=alpha*self.models[panel,seed,'new',h].predict(tx,raw_score=True,num_threads=4)
            ps.append(np.exp(-np.logaddexp(0.,-z)))
        a=np.clip(parent,1e-6,1-1e-6);b=np.clip(np.mean(ps,axis=0).astype('float32'),1e-6,1-1e-6)
        z=.5*(np.log(a)-np.log1p(-a)+np.log(b)-np.log1p(-b))
        return np.exp(-np.logaddexp(0.,-z)).astype('float32')
