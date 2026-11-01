"""Directional/public-position context for advance vs delay decisions."""
import argparse,json
from pathlib import Path
import numpy as np
from core import r
import arrival_prefix as prefix
from onset_phase import HERE as PHASEDIR
from boundary_context import directed_features
from spatial_arrival import field_features

HERE=PHASEDIR/'contextual_prefix';HERE.mkdir(exist_ok=True)
SOURCE_CACHE=prefix.cache

def cache(backend,pop,regime):
    previous=prefix.HERE;prefix.HERE=PHASEDIR
    try:meta,z,c=SOURCE_CACHE(pop,regime)
    finally:prefix.HERE=previous
    dest=HERE/f'{regime}-{pop}.npz'
    if dest.exists():return meta,z,dict(np.load(dest))
    out={k:v.copy() for k,v in c.items()}
    for i,e in enumerate(meta):
        if e['condition']!=0:continue
        ids=c[f'{i}_ri']
        if len(ids):
            slot=int(e['origin'][11:13])*12+int(e['origin'][14:16])//5;s=backend[1](dict(e,origin_slot=slot));d,dnames=directed_features(s,z[f'{i}_base']);f,fnames=field_features(s,backend[2](s)[:,-359:]);newfirst=c[f'{i}_pred'].argmax(1);extra=np.column_stack([d[ids].mean(1),f[ids].mean(1),d[ids,newfirst]])
        else:extra=None
        out[f'{i}_Z']=np.column_stack([c[f'{i}_Z'],extra]) if extra is not None else None
        if (i+1)%512==0:print(json.dumps(dict(stage='arrival_context',pop=pop,done=i+1)),flush=True)
    width=next(v.shape[1] for k,v in out.items() if k.endswith('_Z') and v is not None)
    for k,v in out.items():
        if v is None:out[k]=np.zeros((0,width),dtype='float32')
    np.savez_compressed(dest,**out);r.write(HERE/'context-schema.json',dict(directed=dnames,position_field=fnames,features=width,inputs='history/masked T and existing model predictions only',parent_rank_positive_group=.5));return meta,z,out

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();backend=r.setup(args);prefix.HERE=HERE;prefix.cache=lambda pop,regime:cache(backend,pop,regime);models=prefix.fit(['october'],'context-probe');internal=prefix.population(backend,'november','early',models);b=internal['base']['scores'];eligible=[k for k,v in internal.items() if k!='base' and v['scores']['onset_macro']>b['onset_macro'] and v['scores']['FP']<=b['FP'] and v['scores']['TP']>=b['TP'] and v['scores']['negative_false_any']<=b['negative_false_any']];selected=max(eligible,key=lambda k:internal[k]['scores']['onset_macro']) if eligible else 'base';r.write(args.output/'task2_arrival_context_internal.json',dict(reports=internal,selected=selected,later_month_selection=False,threshold=.5,all_evaluations_reused_development=True));r.write(HERE/'frozen-policy.json',dict(selected=selected,thresholds=dict(onset=.5,ongoing=.25,verifier=.5),later_month_selection=False));models=prefix.fit(['october','november'],'context-full');reports={}
    for pop in ['december','january','february_new','february_old']:
        reports[pop]=prefix.population(backend,pop,'full',models);r.write(args.output/'task2_arrival_context_external.json',dict(reports=reports,frozen_selection=selected,all_evaluations_reused_development=True,official_truth_available=False));print(json.dumps(dict(population=pop,selected=selected,scores={k:{j:v['scores'][j] for j in ['proxy_IoU','onset_macro','TP','FP','FN']} for k,v in reports[pop].items()})),flush=True)

if __name__=='__main__':main()
