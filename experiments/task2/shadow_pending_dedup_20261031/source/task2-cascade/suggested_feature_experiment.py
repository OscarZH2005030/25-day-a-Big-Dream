"""Bounded screenshot feature experiment on partial-label rescue risk head."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from core import ROOT,r
import partial_explanations as partial
import shadow_rescue_precision as base
import shadow_rescue_trajectory as traj
import suggested_hazard as hazard
from suggested_geometry import geometry
PROFILE_DIR=hazard.HERE
HERE=ROOT/'work/task2-cascade/suggested-strict-date-cache';HERE.mkdir(exist_ok=True)
EXTRA={}
RESULT_PREFIX='task2_suggested_strict_features'

class FlexibleShared:
    def __init__(self,compact,observed):self.compact=compact;self.observed=observed
    def predict(self,x,num_threads=4):
        old=x[:,:565];small=np.column_stack([old[:,:220],old[:,469:530]])
        from same_cause_views import shared_cause_probability
        return shared_cause_probability(self.compact.predict(small,num_threads=num_threads),self.observed.predict(x,num_threads=num_threads))

def rich(legacy,margin,raw,h,view):return np.r_[traj.enriched(legacy,margin,raw,h,'observed'),EXTRA[np.asarray(legacy).tobytes()][h]]

def prepare_features(backend,pop,regime,name,prior,training=False):
    meta,z=base.load(pop,regime);dest=HERE/f'{name}-{regime}-{pop}-geometry.npz';cache=dict(np.load(dest)) if dest.exists() else dict(np.load(PROFILE_DIR/dest.name)) if (PROFILE_DIR/dest.name).exists() else {};fds={};requests={};changed=False
    if training:accept,_,_=base.oof_models(name)
    else:
        import lightgbm as lgb
        margin=lgb.Booster(model_file=str(ROOT/'work/task2-cascade'/f'margin-{name}.txt'))
    for i,e in enumerate(meta):
        if e['condition']!=0:continue
        x=z[f'{i}_Z'];a=accept[f'{pop}_{i}'] if training else (margin.predict(x,num_threads=4)-x[:,3]<=0 if len(x) else np.zeros(0,bool));ids=np.flatnonzero(a)
        if not len(ids):continue
        selected=np.unique(z[f'{i}_ri'][ids]);missing=[link for link in selected if f'{i}_{link}' not in cache]
        slot=int(e['origin'][11:13])*12+int(e['origin'][14:16])//5
        if missing:
            panel=e['panel']
            if panel not in fds:
                path=ROOT/'work/data/kaggle_public/corridors'/panel/'network/fd_parameters.csv';fd=pd.read_csv(path).set_index('link_id').reindex(backend[3][panel]['link_ids']);fds[panel]={k:fd[k].to_numpy(float) for k in ['critical_density','lanes']}
            s=backend[1](dict(e,origin_slot=slot));f,names=geometry(s,fds[panel]);poison=dict(s,y=1-s['y'],m=~s['m']);assert np.array_equal(f,geometry(poison,fds[panel])[0])
            for link in missing:cache[f'{i}_{link}']=f[link]
            changed=True
        prior_features=hazard.hazard_features(prior,e['panel'],e['origin'][:10],slot,training)
        for j in ids:
            link=int(z[f'{i}_ri'][j]);ff=np.column_stack([cache[f'{i}_{link}'],np.tile(prior_features[link],(6,1))]);EXTRA[z[f'{i}_Z'][j].tobytes()]=ff
    if changed:np.savez_compressed(dest,**cache);r.write(HERE/'geometry-schema.json',dict(columns=names+['conditional_hazard_band_kind','conditional_hazard_log_count','conditional_hazard_present','conditional_hazard_alltime','conditional_hazard_alltime_log_count','conditional_hazard_alltime_present'],input='history/maskedT and real published neighbors, train-only prior',no_future_feature=True))

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();protocol=r.read(args.output/'task2_origin_stress_frozen.json');base.check_models(protocol);backend=r.setup(args);partial.HERE=HERE;base.HERE=HERE;partial.rich=rich;base.risk_features=rich;partial.same.SharedCause=FlexibleShared
    prior=hazard.profile(backend,['october'],'probe');EXTRA.clear();prepare_features(backend,'october','early','probe',prior,True);probe=partial.fit(['october'],backend,'probe');EXTRA.clear();prepare_features(backend,'november','early','probe',prior);traj.LOOKUP.clear();traj.prepare('november','early','probe');internal=base.evaluate(backend,'november','early','probe',probe);r.write(args.output/f'{RESULT_PREFIX}_internal.json',dict(reports=internal,shadow_only=True));print(json.dumps(dict(pop='november',scores={k:{j:v['scores'][j] for j in ['TP','FP','lost_retained_TP','removed_rescue_FP']} for k,v in internal.items()})),flush=True)
    prior=hazard.profile(backend,['october','november'],'full');EXTRA.clear()
    for pop in ['october','november']:prepare_features(backend,pop,'early','full',prior,True)
    full=partial.fit(['october','november'],backend,'full');reports={}
    for pop in ['december','january','february_new','february_old']:
        EXTRA.clear();prepare_features(backend,pop,'full','full',prior);traj.LOOKUP.clear();traj.prepare(pop,'full','full');reports[pop]=base.evaluate(backend,pop,'full','full',full)
        r.write(args.output/f'{RESULT_PREFIX}_external.json',dict(reports=reports,shadow_only=True,official_truth_available=False,repeated_development=True,default_unchanged=True));print(json.dumps(dict(pop=pop,scores={k:{j:v['scores'][j] for j in ['proxy_IoU','TP','FP','lost_retained_TP','removed_rescue_FP']} for k,v in reports[pop].items()})),flush=True)
    base.check_models(protocol)
if __name__=='__main__':main()
