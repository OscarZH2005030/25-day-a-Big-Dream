"""Posthoc fixed pending-arrival specialist plus frozen safe shadow veto.
Require SAME pending cause in all folds; never union ambiguous fold causes.
"""
import argparse,json
from pathlib import Path
import numpy as np
import lightgbm as lgb
from core import ROOT,r
import suggested_feature_experiment as experiment
import suggested_hazard as hazard
import partial_explanations as partial
import rescue_explanations as exp
import shadow_rescue_precision as base
import shadow_rescue_trajectory as traj
from same_cause_views import shared_cause_probability
HERE=ROOT/'work/task2-cascade/pending-specialist-cache';HERE.mkdir(exist_ok=True)

def combine_votes(safe_votes,pending_votes):
    return np.maximum(np.asarray(safe_votes).min(0),np.asarray(pending_votes).min(0))

class PendingSpecialist:
    def __init__(self,name):
        folder=ROOT/'work/task2-cascade/suggested-ablation-hazard'
        self.timing=[partial.Probabilities(lgb.Booster(model_file=str(folder/f'{name}-{k}.txt'))) for k in [0,1,2]]
        self.safe=[partial.Probabilities(lgb.Booster(model_file=str(partial.HERE/f'{name}-{k}.txt'))) for k in [0,1,2]]
        self.compact=[lgb.Booster(model_file=str(exp.HERE/f'{name}-legacy-{k}.txt')) for k in [0,1,2]]
    def predict(self,x,num_threads=4):
        old=x[:,:565];compact=np.column_stack([old[:,:220],old[:,469:530]])
        baseline=np.array([shared_cause_probability(a.predict(compact,num_threads=num_threads),b.predict(old,num_threads=num_threads)) for a,b in zip(self.compact,self.safe)])
        pending=np.array([m.predict(x,num_threads=num_threads)[:,1] for m in self.timing])
        return combine_votes(baseline,pending)

def rich(legacy,margin,raw,h,view):
    full=experiment.rich(legacy,margin,raw,h,view);return np.r_[full[:565],full[-6:]]

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();protocol=r.read(args.output/'task2_origin_stress_frozen.json');base.check_models(protocol);backend=r.setup(args);base.HERE=HERE;base.risk_features=rich;experiment.HERE=HERE;experiment.PROFILE_DIR=ROOT/'work/task2-cascade/suggested-strict-date-cache';reports={}
    for pop in ['november','december','january','february_new','february_old']:
        name='probe' if pop=='november' else 'full';regime='early' if pop=='november' else 'full';prior=hazard.profile(backend,['october'] if name=='probe' else ['october','november'],name);experiment.EXTRA.clear();experiment.prepare_features(backend,pop,regime,name,prior);traj.LOOKUP.clear();traj.prepare(pop,regime,name);committee=PendingSpecialist(name);reports[pop]=base.evaluate(backend,pop,regime,name,dict(observed_binary=[committee]*4));r.write(args.output/'task2_pending_specialist.json',dict(reports=reports,posthoc_design=True,default_unchanged=True,official_truth_available=False,threshold=.5,policy='frozen safe veto OR pending probability>=.5 in every date fold'))
        print(json.dumps(dict(pop=pop,scores={k:{j:v['scores'][j] for j in ['proxy_IoU','TP','FP','lost_retained_TP','removed_rescue_FP']} for k,v in reports[pop].items()})),flush=True)
    base.check_models(protocol)
if __name__=='__main__':main()
