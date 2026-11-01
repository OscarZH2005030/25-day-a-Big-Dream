"""Align early correction teachers with the inherited history-model family.

R10/R15 probe weights fit June-September. The R127 history routing is inherited
from an already selected model, so the internal check is development, not a
historically blind reconstruction. Exact external parent remains R127.
"""
import argparse,json
from pathlib import Path
import numpy as np
import lightgbm as lgb
from core import ROOT,r,Context
import ongoing as o
import phase_experiment as phase
from queue_phase import phase_features

ORIGINAL=o.HERE;ALIGNED=ORIGINAL/'history_aligned';ALIGNED.mkdir(exist_ok=True)

def build(args,backend,ctx,pop):
    dest=ALIGNED/f'early-{pop}.npz'
    if dest.exists():return
    meta=r.read(ORIGINAL/f'early-{pop}.json');z=np.load(ORIGINAL/f'early-{pop}.npz');out={};c={};a={}
    from coverage_data import matrix
    from history_repair import extras
    directory=args.package/'payload/outputs/task2-round15-ongoing-history';routes={e['panel']:e['alpha_new'] for e in r.read(directory/'routes.json')};alpha=r.read(directory/'frozen-calibration.json')['selected']['alpha_history'];models={p:lgb.Booster(model_file=str(directory/f'{p}-probe.txt')) for p in backend[3]}
    for i,e in enumerate(meta):
        slot=int(e['origin'][11:13])*12+int(e['origin'][14:16])//5;s=backend[1](dict(e,origin_slot=slot));old=z[f'{i}_base'];weight=alpha*routes[e['panel']]
        if weight:
            hp=models[e['panel']].predict(np.column_stack([matrix(s),extras(s)]),num_threads=4).reshape(6,-1).astype('float32');b=(1-weight)*old+weight*hp
        else:b=old.copy()
        new=z[f'{i}_binary'];speed=z[f'{i}_speed'];dynamic=z[f'{i}_dynamic'];ri,hi=np.nonzero((((b>=.25)!=(new>=.5))&dynamic[None]).T);values,_,_=o.features(s,ctx,'early');p,types,names=phase_features(s,b)
        prev=np.vstack([b[:1],b[:-1]]);nxt=np.vstack([b[1:],b[-1:]]);last=np.column_stack([b[hi,ri],new[hi,ri],speed[hi,ri],prev[hi,ri],nxt[hi,ri],b.max(0)[ri],b.mean(0)[ri],(b>=.25).mean(0)[ri],new.max(0)[ri],new.mean(0)[ri],b[hi,ri]>=.25]);x=np.column_stack([values[ri,hi],last])
        for k,v in dict(base=b,y=z[f'{i}_y'],m=z[f'{i}_m']).items():out[f'{i}_{k}']=v
        for k,v in dict(ri=ri,hi=hi,Z=x).items():c[f'{i}_{k}']=v
        a[f'{i}_A']=p[ri,hi];a[f'{i}_phase']=types[ri]
        if (i+1)%512==0:print(json.dumps(dict(stage='history_aligned_candidates',pop=pop,done=i+1)),flush=True)
    np.savez_compressed(dest,**out);np.savez_compressed(ALIGNED/f'gate-early-{pop}.npz',**c);np.savez_compressed(ALIGNED/f'phase-early-{pop}.npz',**a);r.write(dest.with_suffix('.json'),meta);r.write(ALIGNED/'teacher-provenance.json',dict(weights='June-September R10 and R15 history probes',routing=routes,alpha_history=alpha,inherited_routing_selected_later=True,internal_not_blind=True,external_parent='exact R127',thresholds=dict(parent=.25,proposal=.5,verifier=.5)))

def load_aligned(pop,regime):
    directory=ALIGNED if regime=='early' else ORIGINAL
    meta=r.read(directory/f'{regime}-{pop}.json');z=dict(np.load(directory/f'{regime}-{pop}.npz'));c=dict(np.load(directory/f'gate-{regime}-{pop}.npz'));return meta,z,c

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();backend=r.setup(args);ctx=Context(args,backend,o.V2/'profiles',r.read,r.write,lambda *a,**kw:None)
    for pop in ['october','november']:build(args,backend,ctx,pop)
    phase.HERE=ALIGNED;phase.load=load_aligned;models=phase.fit(['october'],'probe');meta,z,c=load_aligned('november','early');a=dict(np.load(ALIGNED/'phase-early-november.npz'));variants=['base','shared','phase_experts','core_guard','boundary_only','fresh_boundary'];internal={k:phase.evaluate(meta,z,c,a,backend,models,k)[0] for k in variants};b=internal['base'];eligible=[k for k in variants[1:] if internal[k]['ongoing_macro']>b['ongoing_macro'] and internal[k]['FP']<=b['FP'] and internal[k]['TP']>=b['TP']];selected=max(eligible,key=lambda k:internal[k]['ongoing_macro']) if eligible else 'base'
    r.write(args.output/'task2_history_aligned_phase_internal.json',dict(scores=internal,selected=selected,later_month_selection=False,probe_weights_fit='June-September',inherited_routing_selected_later=True,internal_not_blind=True));r.write(ALIGNED/'frozen-policy.json',dict(selected=selected,later_month_selection=False,thresholds=dict(parent=.25,proposal=.5,verifier=.5)));models=phase.fit(['october','november'],'full');reports={}
    for pop in ['december','january','february_new','february_old']:
        meta,z,c=load_aligned(pop,'full');a=dict(np.load(ORIGINAL/f'phase-full-{pop}.npz'));reports[pop]={}
        for k in variants:
            scores,rows=phase.evaluate(meta,z,c,a,backend,models,k);reports[pop][k]=dict(scores=scores,windows=rows)
        r.write(args.output/'task2_history_aligned_phase_external.json',dict(reports=reports,frozen_selection=selected,all_evaluations_reused_development=True,official_truth_available=False));print(json.dumps(dict(population=pop,selected=selected,scores={k:{j:v['scores'][j] for j in ['ongoing_macro','TP','FP','FN','known_changed_cells']} for k,v in reports[pop].items()})),flush=True)

if __name__=='__main__':main()
