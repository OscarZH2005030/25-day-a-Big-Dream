"""Phase-specific verification with quality-aware ramp supply evidence."""
import argparse,json
from pathlib import Path
import numpy as np
from core import r
import phase_experiment as phase
from history_aligned_phase import load_aligned,ALIGNED,ORIGINAL
from queue_supply import supply_features

HERE=ALIGNED/'supply';HERE.mkdir(exist_ok=True)

def augment(backend,pop,regime):
    dest=HERE/f'phase-{regime}-{pop}.npz'
    if dest.exists():return dict(np.load(dest))
    import ramp_features as rf
    meta,z,c=load_aligned(pop,regime);source=ALIGNED if regime=='early' else ORIGINAL;old=dict(np.load(source/f'phase-{regime}-{pop}.npz'));out={}
    for i,e in enumerate(meta):
        if e['condition']!=1:continue
        ri=c[f'{i}_ri'];hi=c[f'{i}_hi']
        if len(ri):
            slot=int(e['origin'][11:13])*12+int(e['origin'][14:16])//5;s=backend[1](dict(e,origin_slot=slot));h=rf.ramp_history(s);top=rf.topology(e['panel']);a,names=supply_features(s,h,top);new=a[ri,hi]
        else:new=np.zeros((0,37),dtype='float32')
        out[f'{i}_A']=np.column_stack([old[f'{i}_A'],new]);out[f'{i}_phase']=old[f'{i}_phase']
        if (i+1)%512==0:print(json.dumps(dict(stage='queue_supply',pop=pop,done=i+1)),flush=True)
    np.savez_compressed(dest,**out);r.write(HERE/'supply-schema.json',dict(names=names,history_cutoff='T-5 only',unknown_ramps_not_zero=True,pressure='partial observed balance proxy, external connectors explicitly unknown'));return out

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();backend=r.setup(args)
    for pop in ['october','november']:augment(backend,pop,'early')
    phase.HERE=HERE;phase.load=load_aligned;models=phase.fit(['october'],'probe');meta,z,c=load_aligned('november','early');a=augment(backend,'november','early');variants=['base','shared','phase_experts','core_guard','boundary_only','fresh_boundary'];internal={k:phase.evaluate(meta,z,c,a,backend,models,k)[0] for k in variants};b=internal['base'];eligible=[k for k in variants[1:] if internal[k]['ongoing_macro']>b['ongoing_macro'] and internal[k]['FP']<=b['FP'] and internal[k]['TP']>=b['TP']];selected=max(eligible,key=lambda k:internal[k]['ongoing_macro']) if eligible else 'base'
    r.write(args.output/'task2_supply_phase_internal.json',dict(scores=internal,selected=selected,later_month_selection=False,inherited_routing_selected_later=True,internal_not_blind=True));r.write(HERE/'frozen-policy.json',dict(selected=selected,thresholds=dict(parent=.25,proposal=.5,verifier=.5),later_month_selection=False));models=phase.fit(['october','november'],'full');reports={}
    for pop in ['december','january','february_new','february_old']:
        meta,z,c=load_aligned(pop,'full');a=augment(backend,pop,'full');reports[pop]={}
        for k in variants:
            scores,rows=phase.evaluate(meta,z,c,a,backend,models,k);reports[pop][k]=dict(scores=scores,windows=rows)
        r.write(args.output/'task2_supply_phase_external.json',dict(reports=reports,frozen_selection=selected,all_evaluations_reused_development=True,official_truth_available=False));print(json.dumps(dict(population=pop,selected=selected,scores={k:{j:v['scores'][j] for j in ['ongoing_macro','TP','FP','FN','known_changed_cells']} for k,v in reports[pop].items()})),flush=True)

if __name__=='__main__':main()
