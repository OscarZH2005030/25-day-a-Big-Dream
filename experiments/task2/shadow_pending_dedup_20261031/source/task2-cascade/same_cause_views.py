"""Posthoc shadow rule: both views must support the SAME non-queue cause.
No truth, date, identity, class boosts, or changed probability cutoffs in rule.
"""
import argparse,json
from pathlib import Path
import numpy as np
import lightgbm as lgb
from core import ROOT,r
import shadow_rescue_precision as base
import shadow_rescue_trajectory as traj
import rescue_explanations as exp
HERE=ROOT/'work/task2-cascade/same-cause-cache';HERE.mkdir(exist_ok=True)

def shared_cause_probability(compact,observed):
    a,b=np.asarray(compact),np.asarray(observed)
    assert a.shape==b.shape and a.shape[-1]==3
    return np.minimum(a[...,1:],b[...,1:]).max(-1)

class SharedCause:
    def __init__(self,compact,observed):self.compact=compact;self.observed=observed
    def predict(self,x,num_threads=4):
        assert x.shape[1]==565
        # Rich layout = original469 + trajectory61 + semantic35.
        compact=np.column_stack([x[:,:220],x[:,469:530]])
        a=self.compact.predict(compact,num_threads=num_threads);b=self.observed.predict(x,num_threads=num_threads)
        return shared_cause_probability(a,b)

def models(name):
    bags=[SharedCause(lgb.Booster(model_file=str(exp.HERE/f'{name}-legacy-{k}.txt')),lgb.Booster(model_file=str(exp.HERE/f'{name}-observed-{k}.txt'))) for k in [-1,0,1,2]]
    return dict(observed_binary=bags)

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();protocol=r.read(args.output/'task2_origin_stress_frozen.json');base.check_models(protocol);backend=r.setup(args);base.HERE=HERE;base.risk_features=traj.enriched
    reports={}
    for pop in ['november','december','january','february_new','february_old']:
        regime='early' if pop=='november' else 'full';name='probe' if pop=='november' else 'full';traj.LOOKUP.clear();traj.prepare(pop,regime,name);reports[pop]=base.evaluate(backend,pop,regime,name,models(name))
        r.write(args.output/'task2_same_cause_views.json',dict(reports=reports,policy='same non-current cause >=.5 in both views, consensus over three fixed date models',posthoc_design=True,shadow_only=True,official_truth_available=False,repeated_development=True,default_unchanged=True))
        print(json.dumps(dict(pop=pop,scores={k:{j:v['scores'][j] for j in ['proxy_IoU','onset_macro','TP','FP','lost_retained_TP','removed_rescue_FP']} for k,v in reports[pop].items()})),flush=True)
    base.check_models(protocol)
if __name__=='__main__':main()
