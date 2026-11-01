"""Label-free single-window replay of frozen safe and pending shadow models.
Adds a standalone inference adapter; never edits frozen parent/ancestor files.
"""
from pathlib import Path
import numpy as np
from core import ROOT,r
from origin_stress_inference import FrozenPipeline,prediction_input
from propagation_sequence import features as observed
from shadow_rescue_precision import risk_features,shadow_prediction
from shadow_rescue_trajectory import context_features
from suggested_hazard import hazard_features as query
from pending_specialist import PendingSpecialist
from same_cause_views import shared_cause_probability

class FrozenShadow:
    def __init__(self,args,backend):
        self.pipeline=FrozenPipeline(args,backend);self.committee=PendingSpecialist('full');self.prior=dict(np.load(ROOT/'work/task2-cascade/suggested-feature-cache/full-hazard.npz'))
    def predict(self,sample):
        s=prediction_input(sample);old,d=self.pipeline.predict(s);p={n:old[n].copy() for n in ['R127','retained']};p['safe_shadow']=p['retained'].copy();p['pending_shadow']=p['retained'].copy();d['shadow_features']={};d['shadow_scores']={}
        if s['condition']!=0:return p,d
        b=d['b'];ep,q,_=self.pipeline.gen.predict(s,b);ri,hi=d['raw_ri'],d['raw_hi'];take=np.flatnonzero(d['rescue'][hi,ri]);x=d['rawX'];margin=self.pipeline.margin.model.predict(x,num_threads=4)-x[:,3] if len(x) else np.zeros(0);raw=observed(s)[0];date=str(s['origin'])[:10];origin=str(s['origin']);slot=int(origin[11:13])*12+int(origin[14:16])//5;prior=query(self.prior,s['panel'],date,slot,False);unit=old['retained']&~old['R127']&~d['rescue'];veto={n:np.zeros_like(b,bool) for n in ['safe_shadow','pending_shadow']}
        if len(take):
            rows=[]
            for j in take:
                link,h=int(ri[j]),int(hi[j]);t,semantic=context_features(b[:,link],q[:,link],ep[link],d['first'][link],d['before'][link],d['after'][link],d['conditional'][:,link],d['joint'][:,link],d['rescue'][:,link],h);f=np.r_[risk_features(x[j],margin[j],raw[link],h,'observed'),t,semantic,prior[link]];assert f.shape==(571,);rows.append(f);d['shadow_features'][link,h]=f
            full=np.array(rows);legacy=full[:,:565];compact=np.column_stack([legacy[:,:220],legacy[:,469:530]]);safe=np.array([shared_cause_probability(a.predict(compact,num_threads=4),z.predict(legacy,num_threads=4)) for a,z in zip(self.committee.compact,self.committee.safe)]).min(0);pending=self.committee.predict(full)
            for row,j in enumerate(take):
                link,h=int(ri[j]),int(hi[j]);veto['safe_shadow'][h,link]=safe[row]>=.5;veto['pending_shadow'][h,link]=pending[row]>=.5;d['shadow_scores'][link,h]=[float(safe[row]),float(pending[row])]
        for name in veto:p[name]=shadow_prediction(old['R127'],d['rescue'],unit,veto[name]);assert not (old['R127']&~p[name]).any() and not (unit&~p[name]).any()
        return p,d
