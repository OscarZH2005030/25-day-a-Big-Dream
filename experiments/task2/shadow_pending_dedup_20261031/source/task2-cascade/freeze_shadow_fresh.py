"""Freeze genuinely new origin roster using history-only risk, not outcomes.
Same developed months; historical-risk sampling is explicitly non-natural.
"""
import json,hashlib
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import numpy as np
from core import ROOT,r
from episodes import load
from frozen_shadow_inference import FrozenShadow
from origin_stress_inference import check_models
from pending_specialist import HERE as PENDING_CACHE
HERE=ROOT/'work/task2-cascade/shadow-fresh';HERE.mkdir(exist_ok=True)
PROTOCOL=ROOT/'outputs/task2_shadow_fresh_frozen.json'

def causal_sample(values,net,panel,date,slot):
    from task2q.data import features,infer_condition,FIELDS,FEATURE_NAMES
    history={k:np.concatenate([values[k][slot-12:slot],values['origin_'+k][slot:slot+1]]) for k in FIELDS}
    return dict(x=features(history,net,values['times_ns'][slot-12:slot+1]),network=net,panel=panel,origin=f'{date}T{slot//12:02d}:{slot%12*5:02d}:00Z',condition=infer_condition(history,net),role='frozen_history_audit',feature_names=FEATURE_NAMES)

def observable_risk(s):
    x=s['x'];good=(x[:,:,3]>=.75)&(x[:,:,7]>.5)&(x[:,:,4]<.5);n=x.shape[1];latest=np.ones(n);seen=np.zeros(n,bool);past=np.ones(n);old=np.zeros(n,bool)
    for t in range(13):
        latest[good[t]]=x[t,good[t],0];seen|=good[t]
        if t<=9:past[good[t]]=x[t,good[t],0];old|=good[t]
    gap=latest-s['network']['threshold']/s['network']['free_speed'];drop=np.maximum(past-latest,0)*(seen&old)
    return float(drop.max()-np.min(np.abs(gap[seen]))) if seen.any() else -999.

def freeze(backend):
    if PROTOCOL.exists():return r.read(PROTOCOL)
    old=r.read(ROOT/'outputs/task2_origin_stress_frozen.json');exclude=defaultdict(list)
    for pop in ['december','january','february_new','february_old']:
        for e in load(pop,'full')[0]:exclude[e['panel'],e['origin'][:10]].append(int(e['origin'][11:13])*12+int(e['origin'][14:16])//5)
    for e in old['origins']:exclude[e['panel'],e['date']].append(e['origin_slot'])
    rng=np.random.default_rng(20261031);origins=[];dates={};panels=list(backend[3])
    for month in ['2030-12','2031-01','2031-02']:
        files=(ROOT/'work/data/kaggle_public/corridors'/panels[0]/'train/mainline_states'/('year_month='+month)).glob('*.parquet');available=sorted({p.stem.removeprefix('synthetic_mainline_').replace('_','-') for p in files}-set(old['selected_dates'][month]));selected=sorted(rng.choice(available,size=2,replace=False).tolist());dates[month]=selected
        for date in selected:
            for panel in panels:
                values=backend[1].load_day(panel,date);candidates=defaultdict(list)
                for slot in range(12,282,3):
                    if any(abs(slot-q)<=12 for q in exclude[panel,date]):continue
                    s=causal_sample(values,backend[3][panel],panel,date,slot);candidates[s['condition']].append((observable_risk(s),slot))
                for condition,cc in candidates.items():
                    cc=sorted(cc,key=lambda a:(-a[0],a[1]));top=cc[0];remaining=[v for v in cc if abs(v[1]-top[1])>12];chosen=[('highest_observed_risk',top)]
                    if remaining:chosen.append(('uniform_history_eligible',remaining[int(rng.integers(len(remaining)))]))
                    for rule,(risk,slot) in chosen:origins.append(dict(panel=panel,date=date,origin_slot=slot,origin=f'{date}T{slot//12:02d}:{slot%12*5:02d}:00Z',condition=int(condition),sampling_rule=rule,history_risk=risk))
    paths=[ROOT/'work/task2-cascade'/n for n in ['frozen_shadow_inference.py','pending_specialist.py','same_cause_views.py','shadow_rescue_trajectory.py','suggested_hazard.py','freeze_shadow_fresh.py']]
    for folder in ['partial-explanation-cache','suggested-ablation-hazard']:paths+=list((ROOT/'work/task2-cascade'/folder).glob('full-*.txt'))
    import rescue_explanations as exp
    paths+=list(exp.HERE.glob('full-legacy-*.txt'));paths.append(ROOT/'work/task2-cascade/suggested-feature-cache/full-hazard.npz')
    protocol=dict(seed=20261031,origins=origins,selected_dates=dates,roster_sha256=hashlib.sha256(json.dumps(origins,sort_keys=True).encode()).hexdigest(),model_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},policies=['R127','retained','safe_shadow','pending_shadow'],same_developed_months=True,independent_new_month=False,not_natural_prevalence=True,selection='two preselected dates/month excluding opened574 dates; per available history-condition select highest observed risk and one uniform origin >60min away; >60min from old and574 origins; no future label fields used',thresholds=dict(onset=.5,ongoing=.25,binary=.5),no_refitting=True,no_selection_using_future_results=True,official_truth_available=False,metric='original complete macro only if coverage exists; all-onset TP/FP/FN and negative-window alarms separately',sample_construction='explicit twelve preT frames plus released maskedT; full-day cache exists but no future slice accessed during selection/inference')
    r.write(PROTOCOL,protocol);return protocol

def main():
    assert r.read(ROOT/'outputs/task2_frozen_shadow_reconstruction.json')['passed'];cfg=r.read(ROOT/'outputs/task2_research_handoff.json')['args'];args=SimpleNamespace(**{k:Path(v) for k,v in cfg.items()});backend=r.setup(args);old=r.read(ROOT/'outputs/task2_origin_stress_frozen.json');check_models(old);protocol=freeze(backend);check_models(protocol);pipe=FrozenShadow(args,backend)
    for j,e in enumerate(protocol['origins']):
        path=HERE/f'prediction-{j:04d}.npz'
        if not path.exists():
            values=backend[1].load_day(e['panel'],e['date']);s=causal_sample(values,backend[3][e['panel']],e['panel'],e['date'],e['origin_slot']);assert s['condition']==e['condition'];pred,d=pipe.predict(s);np.savez_compressed(path,**pred);r.write(HERE/f'prediction-{j:04d}.json',dict(origin=e['origin'],panel=e['panel'],roster_sha256=protocol['roster_sha256'],targets_opened=False,rescues=int(d['rescue'].sum()),shadow_changed_cells=int((pred['retained']^pred['pending_shadow']).sum())))
        else:assert r.read(HERE/f'prediction-{j:04d}.json')['roster_sha256']==protocol['roster_sha256']
        if (j+1)%10==0:r.write(ROOT/'outputs/task2_shadow_fresh_progress.json',dict(status='frozen inference',completed=j+1,total=len(protocol['origins'])));print(j+1,len(protocol['origins']),flush=True)
    check_models(old);check_models(protocol);r.write(ROOT/'outputs/task2_shadow_fresh_progress.json',dict(status='all predictions frozen; scoring not yet opened',completed=len(protocol['origins']),total=len(protocol['origins']),next='Score frozen policies without fitting/selecting; report coverage and any newly lost knownTP honestly'))
if __name__=='__main__':main()
