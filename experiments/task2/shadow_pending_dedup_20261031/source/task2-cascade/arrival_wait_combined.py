"""Exploratory union of explicit-next-slot and conservative ordinal vetoes.

This composition was defined AFTER development-month ablations were inspected.
Its comparison is descriptive, not a frozen-policy independent validation.
"""
import argparse,json
from pathlib import Path
import numpy as np
import lightgbm as lgb
from core import r
import arrival_wait as wait
import arrival_ordinal_wait as ordinal

HERE=wait.HERE/'combined';HERE.mkdir(exist_ok=True)

def load(name):
    explicit=lgb.Booster(model_file=str(wait.HERE/f'{name}-multi.txt'))
    bags=[lgb.Booster(model_file=str(ordinal.HERE/f'{name}-{k}.txt')) for k in range(3)]
    class Union:
        def predict(self,x,**kw):
            return np.maximum(explicit.predict(x,num_threads=4)[:,1],np.array([m.predict(x,num_threads=4) for m in bags]).min(0))
    class Multi:
        def predict(self,x,**kw):
            p=Union().predict(x);return np.column_stack([1-p,p,np.zeros(len(x))])
    return dict(multi=Multi(),pair=Union(),pair_consensus=[Union()])

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();backend=r.setup(args)
    models={name:load(name) for name in ['probe','full']};previous=wait.HERE;wait.HERE=HERE;reports={}
    try:
        for pop in ['november','december','january','february_new','february_old']:
            name='probe' if pop=='november' else 'full';regime='early' if pop=='november' else 'full'
            v=wait.evaluate(backend,pop,regime,name,models[name]);reports[pop]={k:v[k] for k in ['base','margin','committee_advance']};reports[pop]['combined']=v['multi']
            r.write(args.output/'task2_arrival_wait_combined.json',dict(reports=reports,policy_defined_after_development_ablations=True,independent_validation=False,official_truth_available=False,formal_R127_unchanged=True,thresholds=dict(onset=.5,ongoing=.25,wait=.5)))
            print(json.dumps(dict(pop=pop,scores={k:{j:vv['scores'][j] for j in ['proxy_IoU','onset_macro','TP','FP','lost_original_positive_cells','lost_rescue_positive_cells']} for k,vv in reports[pop].items()})),flush=True)
    finally:wait.HERE=previous

if __name__=='__main__':main()
