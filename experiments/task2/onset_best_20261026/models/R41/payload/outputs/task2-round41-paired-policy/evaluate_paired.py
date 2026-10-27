import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import sys,json,hashlib,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent;PROJECT=ROOT.parents[1]
from paired_runtime import ConcentrationPredictor,normalized_mix
from gated_core import concentrated_adjust
from timing_core import features,adjust,unseen_queue
from run_round2 import sample,role_samples,PANELS,score,DATA
from task2q.cache import atomic_json
from task2q.data import released_sample
from task2q.metrics import window_iou
from task2q.submission import validate_queue
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(n,v):atomic_json(ROOT/n,v)
def progress(phase,**kw):
    v=dict(phase=phase,pid=os.getpid(),utc=pd.Timestamp.now(tz='UTC').isoformat(),**kw);save('development-progress.json',v);atomic_json(OUT/'overnight-status.json',dict(v,script=str(Path(__file__)),training=False));print(json.dumps(v),flush=True)
def metric(ss,ps,t):return float(np.mean([window_iou(s['y'],p>=t,s['m']) for s,p in zip(ss,ps)]))
def negative(ss,ps):
    groups={}
    for s,p in zip(ss,ps):
        fp=(p>=.5)&s['m'];groups.setdefault(s['panel'],[]).append([float(fp.any()),float(fp.sum()/s['m'].sum())])
    return dict(windows=len(ss),false_any_macro=float(np.mean([np.mean(v,axis=0)[0] for v in groups.values()])),false_eligible_fraction_macro=float(np.mean([np.mean(v,axis=0)[1] for v in groups.values()])))
def main():
    started=time.time();training=read(ROOT/'head-source-verification.json');assert training['passed']
    for n,h in training['hashes'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h
    from package_guard import guard
    before=guard('development-before');base=ConcentrationPredictor(OUT/'task2-round38-concentration-policy');anchor=base.anchor;bt=base.rule['thresholds'];head=lgb.Booster(model_file=str(ROOT/'conditional-first-full.txt'));gate=read(OUT/'task2-round40-paired-timing/training-gates.json')['roads'];tune=[sample(e) for e in read(OUT/'task2-round11-phase-validation/tune-phase-index.json')['samples'] if e['condition']==0];routes={};reports=[]
    save('development-protocol.json',dict(hypothesis='source-neutral paired timing views may improve cross-phase conditional predictions; new R40 head changes only majority-mode links, others retain frozen R38',new_models_in_policy_stage=0,head_source='R40 full600 trained on original+paired conditional targets; exact SHA; original X/y preserved; no coefficient search',anchor='R31 original event curve; low concentration retains current R34 curve, no repeated CDF on already corrected curves',December='only R40-gated roads and >=4 windows per phase; alpha.5/1, mean gain>=.004/worstphase>=-.015',January='single global Onsetalpha0/.5/1; normalized blend preserves unmasked peaks/known queue links; fixed thresholds .5/.25 and negative false-any+.03/eligibleFP+.005',February='both populations already viewed, only frozen paired report; no routing/tuning',Ongoing='all probabilities and threshold unchanged',official_S_queue=None))
    for p in PANELS:
        ss=[s for s in tune if s['panel']==p];phase=np.array([(pd.Timestamp(s['origin']).hour*12+pd.Timestamp(s['origin']).minute//5)%3 for s in ss]);bp=[base.probability(s) for s in ss];ep=[anchor.probability(s) for s in ss];bv=metric(ss,bp,bt[0]);fallback=dict(alpha_new=0.,proxy=bv,phase_gains=[0.,0.,0.],eligible=True);candidates=[fallback]
        eligible=bool(gate[p] and min(int((phase==i).sum()) for i in range(3))>=4)
        if eligible:
            qp=[head.predict(features(s),num_threads=4).astype('float32') for s in ss]
            for a in [.5,1.]:
                ps=[concentrated_adjust(s,b,e,q,a) for s,b,e,q in zip(ss,bp,ep,qp)];gains=[]
                for ph in range(3):
                    ii=np.flatnonzero(phase==ph);gains.append(metric([ss[i] for i in ii],[ps[i] for i in ii],bt[0])-metric([ss[i] for i in ii],[bp[i] for i in ii],bt[0]))
                value=metric(ss,ps,bt[0]);candidates.append(dict(alpha_new=a,proxy=value,phase_gains=gains,eligible=bool(value>=bv+.004 and min(gains)>=-.015)))
        chosen=max((r for r in candidates if r['eligible']),key=lambda r:r['proxy']);routes[p]=chosen;reports.append(dict(panel=p,baseline_proxy=bv,early_and_phase_gate=eligible,candidates=candidates,selected=chosen));save('december-selection.json',reports);save('routes.json',[dict(panel=p,**r) for p,r in routes.items()]);progress('December_route_complete',panel=p,new_alpha=chosen['alpha_new'])
    def components(ss):
        bb=[];qq=[]
        for s in ss:
            b=base.probability(s);bb.append(b);a=routes[s['panel']]['alpha_new'] if s['condition']==0 else 0.
            qq.append(concentrated_adjust(s,b,anchor.probability(s),head.predict(features(s),num_threads=4).astype('float32'),a) if a else b)
        return bb,qq
    def mixed(ss,bb,qq,a):return [normalized_mix(s,b,q,a) if s['condition']==0 else b for s,b,q in zip(ss,bb,qq)]
    cal=[sample(e) for e in read(OUT/'task2-round11-phase-validation/calibrate-phase-index.json')['samples']];bb,qq=components(cal);neg=[s for s in role_samples('calibrate') if s['condition']==0 and not s['has_future_queue']];nb,nq=components(neg);nbase=negative(neg,nb);candidates=[]
    for a in [0.,.5,1.]:
        ns=negative(neg,mixed(neg,nb,nq,a));eligible=bool(a==0 or (ns['false_any_macro']<=nbase['false_any_macro']+.03 and ns['false_eligible_fraction_macro']<=nbase['false_eligible_fraction_macro']+.005));candidates.append(dict(alpha_paired=a,thresholds=bt,proxy=score(cal,mixed(cal,bb,qq,a),bt)['proxy_IoU'],negative_stats=ns,eligible=eligible))
    chosen=max((r for r in candidates if r['eligible']),key=lambda r:r['proxy']);hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['conditional-first-full.txt','timing_core.py','paired_runtime.py','gated_core.py']};frozen=dict(selected=chosen,candidates=candidates,hashes=hashes,route_sha256=hashlib.sha256((ROOT/'routes.json').read_bytes()).hexdigest(),baseline_frozen_sha256=hashlib.sha256((OUT/'task2-round38-concentration-policy/frozen-calibration.json').read_bytes()).hexdigest(),early_gate_sha256=training['early_gate_sha256'],condition_forecast_frozen=[chosen['alpha_paired']==0 or not any(r['alpha_new']>0 for r in routes.values()),True],all_thresholds_frozen=True,February_previously_used_report_only=True,Task1_frozen=True);save('frozen-calibration.json',frozen);progress('timing_frozen',alpha_paired=chosen['alpha_paired'])
    scores={};records=[]
    for population,ss in [('new_phase_population',[sample(e) for e in read(OUT/'task2-round11-phase-validation/audit-phase-index.json')['samples']]),('old410',role_samples('audit'))]:
        b,q=components(ss);final=mixed(ss,b,q,chosen['alpha_paired']);scores[population+'_round38']=score(ss,b,bt);scores[population+'_selected']=score(ss,final,bt)
        assert abs(scores[population+'_round38']['proxy_IoU']-read(OUT/'task2-round38-concentration-policy/proxy-results.json')['reports'][population+'_selected']['proxy_IoU'])<1e-12
        for s,bp,fp in zip(ss,b,final):
            assert np.array_equal(bp.max(0),fp.max(0))
            if s['condition']==1:assert np.array_equal(bp,fp)
            if s['has_future_queue']:records.append(dict(population=population,panel=s['panel'],condition=s['condition'],origin=s['origin'],date=s['origin'][:10],base_iou=window_iou(s['y'],bp>=bt[s['condition']],s['m']),selected_iou=window_iou(s['y'],fp>=bt[s['condition']],s['m'])))
    save('paired-window-metrics.json',records);save('proxy-results.json',dict(official_S_queue=None,reports=scores,selected=chosen,new_groups={'0':sum(r['alpha_new']>0 for r in routes.values()),'1':0},condition_forecast_frozen=frozen['condition_forecast_frozen'],all_six_step_peaks_identical=True,all_thresholds_identical=True,Task1_historical_S_state=.942828633,Task1_score_newly_measured=False,different_populations_not_directly_comparable=True))
    from paired_runtime import PairedPredictor
    runtime=PairedPredictor(ROOT);checks={}
    for split in ['validation','private']:
        frames=[]
        for p in PANELS:
            d=DATA/'task2'/p/split;index=pd.read_csv(d/'window_index.csv');template=pd.read_csv(d/'sample_submission_queue.csv');parts=[]
            for row in index.itertuples():
                s=released_sample(DATA,p,split,row.window_id);pred=runtime.predict(s);keys=template[template.window_id==row.window_id].copy();ii={l:i for i,l in enumerate(s['network']['link_ids'])};h=((pd.to_datetime(keys.timestamp,utc=True)-pd.Timestamp(s['origin'])).dt.total_seconds()/300).to_numpy(dtype=int)-1;keys['queue_pred']=pred[h,keys.link_id.map(ii).to_numpy()].astype(int);parts.append(keys)
            frame=pd.concat(parts,ignore_index=True);checks[split+'/'+p]=validate_queue(frame,template);frame['panel']=p;frames.append(frame)
        frame=pd.concat(frames,ignore_index=True);assert len(frame)==87000;frame.to_csv(ROOT/f'queue-{split}.csv',index=False)
    assert guard('development-after')==before;assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items());save('final-verification.json',dict(passed=True,rows=174000,checks=checks,condition_forecast_frozen=frozen['condition_forecast_frozen'],all_six_step_peaks_identical=True,Task1_package_unchanged=True,official_S_queue=None));progress('development_complete',new_phase_proxy=scores['new_phase_population_selected']['proxy_IoU'],old410_proxy=scores['old410_selected']['proxy_IoU'],seconds=time.time()-started)
if __name__=='__main__':
    try:main()
    except Exception:save('development-failure.json',dict(traceback=traceback.format_exc()));progress('failed');raise
