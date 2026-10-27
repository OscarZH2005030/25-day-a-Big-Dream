"""Shared-observation window weighting; original features/labels/cells stay fixed."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='4'
import sys,json,time,hashlib,traceback,gc,copy,itertools
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent;PROJECT=ROOT.parents[1]
sys.path[1:1]=[str(OUT/n) for n in ['task2-round23-path-context','task2-round21-link-category','task2-round19-interval-budget','task2-round18-interval-likelihood','task2-round17-onset-hazard','task2-round15-ongoing-history','task2-round14-history-repair','task2-round13-seed-stability','task2-round12-phase-soft','task2-round11-phase-validation','task2-round10-phase-coverage','task2-round9-hierarchical','task2-round8-soft-targets','task2-round7-structured','task2-round6-ramp-physics','task2-round5-distribution','task2-round2']]
from path_runtime import PathPredictor
from coverage_data import matrix as features
from weight_data import build
from run_round2 import sample,arrays,role_samples,score,DATA,PANELS
from task2q.data import make_sample,released_sample
from task2q.cache import atomic_json
from task2q.metrics import window_iou
from task2q.submission import validate_queue
SEED=20261027;WORK=PROJECT/'work/round25-overlap-weights';WORK.mkdir(parents=True,exist_ok=True)
TASK1=Path(r'C:\Users\18049\Documents\Codex\2026-10-24\trafficflowbench-186-global-road-adaptation\outputs\TrafficFlowBench-Best-20261026')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(n,v):atomic_json(ROOT/n,v)
def progress(phase,**kw):
    v=dict(phase=phase,pid=os.getpid(),utc=pd.Timestamp.now(tz='UTC').isoformat(),**kw);save('progress.json',v);print(json.dumps(v),flush=True)
def guard(stage):
    m=read(TASK1/'package-manifest.json');h={n:hashlib.sha256((TASK1/n).read_bytes()).hexdigest() for n in m['files']};assert all(h[n]==v['sha256'] for n,v in m['files'].items());save('task1-guard-'+stage+'.json',dict(passed=True,files=len(h),hashes=h,historical_S_state=.942828633,new_Task1_score_measured=False));return h
def fit(name,X,y,w,rounds,leaves=31):
    path=ROOT/(name+'.txt')
    if path.exists():return lgb.Booster(model_file=str(path))
    if pd.Timestamp.now(tz='UTC')>=pd.Timestamp('2026-10-27T15:00:00Z') or (PROJECT/'work/STOP_NIGHT').exists():raise RuntimeError('New-training cutoff or STOP_NIGHT reached')
    cp=ROOT/(name+'-checkpoint.txt');initial=lgb.Booster(model_file=str(cp)) if cp.exists() else None;done=initial.current_iteration() if initial else 0
    def callback(env):
        if (env.iteration+1)%100==0:env.model.save_model(str(cp));progress('training',model=name,step=env.iteration+1,total=rounds)
    model=lgb.train(dict(objective='binary',num_leaves=leaves,learning_rate=.035,min_data_in_leaf=80,feature_fraction=.85,lambda_l2=4,num_threads=4,verbosity=-1,seed=SEED,deterministic=True,force_col_wise=True),lgb.Dataset(X,label=y,weight=w/w.mean()),num_boost_round=rounds-done,init_model=initial,callbacks=[callback]) if done<rounds else initial;model.save_model(str(path));save(name+'-weight-audit.json',dict(features=model.num_feature(),unchanged_features=True,label_side_window_mass_only=True));return model

def metric(ss,ps,t):return float(np.mean([window_iou(s['y'],p>=t,s['m']) for s,p in zip(ss,ps)]))
def negative_stats(ss,ps,t):
    groups={}
    for s,p in zip(ss,ps):
        assert not s['has_future_queue'];fp=(p>=t)&s['m'];groups.setdefault(s['panel'],[]).append([float(fp.any()),float(fp.sum()/s['m'].sum())])
    if not groups:return None
    return dict(windows=len(ss),panels=len(groups),false_any_macro=float(np.mean([np.mean(v,axis=0)[0] for v in groups.values()])),false_eligible_fraction_macro=float(np.mean([np.mean(v,axis=0)[1] for v in groups.values()])))
def negative_ok(new,base):return bool(new is not None and base is not None and new['false_any_macro']<=base['false_any_macro']+.03 and new['false_eligible_fraction_macro']<=base['false_eligible_fraction_macro']+.005)
def main():
    started=time.time();before=guard('before');assert read(ROOT/'weight-verification.json')['passed'];overlap=read(OUT/'task2-round24-supervision-overlap/training-gates.json')['groups'];records=read(OUT/'task2-round24-supervision-overlap/window-groups.json');baseline=PathPredictor(OUT/'task2-round23-path-context');bt=baseline.rule['thresholds'];save('experiment.json',dict(seed=SEED,hypothesis='repeat views of identical positive physical observations over-weight a few supervision groups, especially Onset; rebalance window mass without deleting any positives',features='unchanged R10 555; no group ID, multiplicity, factor or future metadata as inference input',Onset='inverse shared-observation group size, equal total positive mass per group',Ongoing='inverse square root group size, softer correction retaining duration variation; only severe-overlap eligible groups',group_semantics='components of windows sharing identical eligible positive timestamp/link cells, within day/panel/condition; NOT independent physical episodes',sampling='all original R10 X/y/cells and within-window IPW retained; positive-window class mass .8 / negative .2 unchanged; all row weights positive',probe_scope='June-September factors and normalizers derived ONLY from that period; full factors use June-November',training_gate='R24 early windows/groups>=1.5, >=30 groups, factor std>=.05',max_models=2*sum(overlap.values()),max_tree_rounds=800*sum(overlap.values()),early_rounds=200,early_leaves=31,full_rounds=600,full_leaves='Onset31/Ongoing63 as R10',CPU_threads=4,temporal='October-November original windows vs identical-feature old R10 early200; median gain .1/.2/.35>=-.01 plus negative guard and Dec>=4 per phase',negative_guard='unchanged thresholds .5/.25; false-any increase<=.03 and eligible-cell FP increase<=.005, early holdout and January per-condition global calibration',December='fixed600 alpha.5/1, gain>=.004 and worstphase>=-.015',January='two global blends0/.5/1, fixed thresholds; no per-road tuning',baseline='R23 frozen January policy, retained despite mixed repeated-February reports; do not rollback using February',Task1='92files frozen, historical .942828633 not newly measured',official_S_queue=None))

    entries=read(OUT/'task2-round10-phase-coverage/expanded-fit.json')['samples'];tune=[sample(e) for e in read(OUT/'task2-round11-phase-validation/tune-phase-index.json')['samples']];models={};routes={};dev=[];hold_reports=[];audits=[];causal=[]
    for p,c in itertools.product(PANELS,[0,1]):
        e=next(e for e in entries if e['panel']==p and e['condition']==c);s=sample(e);v=copy.deepcopy(arrays(e['day_key']));sl=e['origin_slot']
        for k in ['speed','flow','occupancy','quality','origin_speed','origin_flow','origin_occupancy','origin_quality']:v[k][sl+1:]=999
        v['eligible'][sl+1:]=False;s2=make_sample(v,s['network'],sl,p);assert np.array_equal(features(s),features(s2));causal.append(dict(panel=p,condition=c,passed=True))
    save('feature-causality-check.json',dict(passed=True,groups=causal,group_labels_and_future_metadata_not_prediction_inputs=True))
    for p,c in itertools.product(PANELS,[0,1]):
        ss=[s for s in tune if s['panel']==p and s['condition']==c];phase=np.array([(pd.Timestamp(s['origin']).hour*12+pd.Timestamp(s['origin']).minute//5)%3 for s in ss]);phase_ok=min(int((phase==k).sum()) for k in [0,1,2])>=4;base=[baseline.probability(s) for s in ss];bv=metric(ss,base,bt[c]);fallback=dict(alpha_new=0.,proxy=bv,phase_gains=[0.,0.,0.],eligible=True)
        if not overlap[f'{p}/{c}']:routes[(p,c)]=fallback;dev.append(dict(panel=p,condition=c,selected=fallback,reason='early shared-observation heterogeneity gate failed'));save('december-selection.json',dev);save('routes.json',[dict(panel=pp,condition=cc,**r) for (pp,cc),r in routes.items()]);continue
        progress('building_overlap_weights',panel=p,condition=c,completed=len(routes),total=16);data,meta,audit=build(p,c,records,WORK,PROJECT);audits.append({k:v for k,v in audit.items() if k!='windows'});save('training-data-audit.json',audits);selected=data['early'].astype(bool);weights=np.asarray(data['w_early'][selected]);probe=fit(f'{p}-{c}-probe',np.asarray(data['X'][selected]),np.asarray(data['y'][selected]),weights,200);legacy=lgb.Booster(model_file=str(OUT/f'task2-round10-phase-coverage/{p}-{c}-probe-expanded.txt'));hold=[sample(e) for e in entries if e['panel']==p and e['condition']==c and e['source']=='original' and e['origin'][:10]>='2030-10-01'];bp=[];qp=[]
        for s in hold:
            xx=features(s);bp.append(legacy.predict(xx[:,:555],num_threads=4).astype('float32').reshape(6,-1));qp.append(probe.predict(xx,num_threads=4).astype('float32').reshape(6,-1))
        positive=[i for i,s in enumerate(hold) if s['has_future_queue']];negative=[i for i,s in enumerate(hold) if not s['has_future_queue']];hs=[hold[i] for i in positive];bpos=[bp[i] for i in positive];qpos=[qp[i] for i in positive];rr=[dict(threshold=t,control_IoU=metric(hs,bpos,t),weighted_IoU=metric(hs,qpos,t)) for t in [.1,.2,.35]];median=float(np.median([r['weighted_IoU']-r['control_IoU'] for r in rr]));nb=negative_stats([hold[i] for i in negative],[bp[i] for i in negative],bt[c]);nq=negative_stats([hold[i] for i in negative],[qp[i] for i in negative],bt[c]);ng=negative_ok(nq,nb);used=float(np.std([r['full_factor'] for r in audit['windows'] if r['positive']]));temporal=bool(median>=-.01 and ng and used>0);eligible=bool(temporal and phase_ok);hold_reports.append(dict(panel=p,condition=c,positive_windows=len(positive),negative_windows=len(negative),reports=rr,median_gain=median,negative_control=nb,negative_weighted=nq,negative_guard_passed=ng,positive_window_factor_std=used,temporal_guard_passed=temporal,December_phase_coverage_passed=phase_ok,full_training_guard_passed=eligible));save('temporal-holdout.json',hold_reports);del bp,qp,probe,legacy;gc.collect()
        if not eligible:
            routes[(p,c)]=fallback;dev.append(dict(panel=p,condition=c,selected=fallback,reason='time/negative/weight-change/phase gate failed'));save('december-selection.json',dev);save('routes.json',[dict(panel=pp,condition=cc,**r) for (pp,cc),r in routes.items()]);del data;gc.collect();continue
        models[(p,c)]=fit(f'{p}-{c}-weighted',data['X'],data['y'],data['w'],600,31 if c==0 else 63);del data;gc.collect();raw=[models[(p,c)].predict(features(s),num_threads=4).astype('float32').reshape(6,-1) for s in ss];candidates=[fallback]
        for alpha in [.5,1.]:
            ps=[alpha*a+(1-alpha)*b for a,b in zip(raw,base)];gains=[]
            for ph in [0,1,2]:
                ii=np.flatnonzero(phase==ph);gains.append(metric([ss[i] for i in ii],[ps[i] for i in ii],bt[c])-metric([ss[i] for i in ii],[base[i] for i in ii],bt[c]))
            value=metric(ss,ps,bt[c]);candidates.append(dict(alpha_new=alpha,proxy=value,phase_gains=gains,eligible=value>=bv+.004 and min(gains)>=-.015))
        chosen=max((r for r in candidates if r['eligible']),key=lambda r:r['proxy']);routes[(p,c)]=chosen;dev.append(dict(panel=p,condition=c,baseline_proxy=bv,selected=chosen,candidates=candidates));save('december-selection.json',dev);save('routes.json',[dict(panel=pp,condition=cc,**r) for (pp,cc),r in routes.items()])
    def components(ss):
        base=[];adapt=[]
        for s in ss:
            b=baseline.probability(s);base.append(b);key=(s['panel'],s['condition']);r=routes[key]
            if r['alpha_new']==0:adapt.append(b)
            else:
                q=models[key].predict(features(s),num_threads=4).astype('float32').reshape(6,-1);a=r['alpha_new'];adapt.append(a*q+(1-a)*b)
        return base,adapt
    def mixed(ss,base,adapt,alphas):
        return [alphas[s['condition']]*q+(1-alphas[s['condition']])*b for s,b,q in zip(ss,base,adapt)]
    progress('calibration');cal=[sample(e) for e in read(OUT/'task2-round11-phase-validation/calibrate-phase-index.json')['samples']];base,adapt=components(cal);negative_all=[s for s in role_samples('calibrate') if not s['has_future_queue']];negative={c:[s for s in negative_all if s['condition']==c] for c in [0,1]};ncomp={c:components(negative[c]) for c in [0,1]};nbase={c:negative_stats(negative[c],ncomp[c][0],bt[c]) for c in [0,1]};candidates=[]
    for a0,a1 in itertools.product([0.,.5,1.],repeat=2):
        alphas=[a0,a1];ns={c:negative_stats(negative[c],mixed(negative[c],*ncomp[c],alphas),bt[c]) for c in [0,1]};eligible=all(alphas[c]==0 or negative_ok(ns[c],nbase[c]) for c in [0,1]);candidates.append(dict(alphas_overlap=alphas,thresholds=bt,proxy=score(cal,mixed(cal,base,adapt,alphas),bt)['proxy_IoU'],negative_stats=ns,eligible=eligible))
    chosen=max((r for r in candidates if r['eligible']),key=lambda r:r['proxy']);hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.txt')};condition_frozen=[chosen['alphas_overlap'][c]==0 or not any(key[1]==c and r['alpha_new']>0 for key,r in routes.items()) for c in [0,1]];save('frozen-calibration.json',dict(selected=chosen,candidates=candidates,hashes=hashes,route_sha256=hashlib.sha256((ROOT/'routes.json').read_bytes()).hexdigest(),feature_source_sha256=hashlib.sha256((OUT/'task2-round10-phase-coverage/coverage_data.py').read_bytes()).hexdigest(),training_overlap_sha256=hashlib.sha256((OUT/'task2-round24-supervision-overlap/window-groups.json').read_bytes()).hexdigest(),baseline_frozen_sha256=hashlib.sha256((OUT/'task2-round23-path-context/frozen-calibration.json').read_bytes()).hexdigest(),condition_forecast_frozen=condition_frozen,all_thresholds_frozen=True,February_previously_used_report_only=True,Task1_frozen=True))
    progress('evaluating');reports={};records=[]
    for population,ss in [('new_phase_population',[sample(e) for e in read(OUT/'task2-round11-phase-validation/audit-phase-index.json')['samples']]),('old410',role_samples('audit'))]:
        base,adapt=components(ss);final=mixed(ss,base,adapt,chosen['alphas_overlap']);reports[population+'_round23']=score(ss,base,bt);reports[population+'_selected']=score(ss,final,bt)
        for s,b,q in zip(ss,base,final):
            if condition_frozen[s['condition']]:assert np.array_equal(b,q)
            if s['has_future_queue']:records.append(dict(population=population,panel=s['panel'],condition=s['condition'],origin=s['origin'],date=s['origin'][:10],base_iou=window_iou(s['y'],b>=bt[s['condition']],s['m']),selected_iou=window_iou(s['y'],q>=bt[s['condition']],s['m'])))
        assert abs(reports[population+'_round23']['proxy_IoU']-read(OUT/'task2-round23-path-context/proxy-results.json')['reports'][population+'_selected']['proxy_IoU'])<1e-12
    save('paired-window-metrics.json',records);save('proxy-results.json',dict(official_S_queue=None,reports=reports,selected=chosen,new_groups={str(c):sum(key[1]==c and r['alpha_new']>0 for key,r in routes.items()) for c in [0,1]},seconds=time.time()-started,condition_forecast_frozen=condition_frozen,all_thresholds_identical=True,different_populations_not_directly_comparable=True,Task1_historical_S_state=.942828633,Task1_score_newly_measured=False))
    progress('predicting_official');checks={}
    for split in ['validation','private']:
        frames=[]
        for p in PANELS:
            d=DATA/'task2'/p/split;index=pd.read_csv(d/'window_index.csv');template=pd.read_csv(d/'sample_submission_queue.csv');parts=[]
            for row in index.itertuples():
                s=released_sample(DATA,p,split,row.window_id);bb,aa=components([s]);q=mixed([s],bb,aa,chosen['alphas_overlap'])[0];keys=template[template.window_id==row.window_id].copy();ii={l:i for i,l in enumerate(s['network']['link_ids'])};h=((pd.to_datetime(keys.timestamp,utc=True)-pd.Timestamp(s['origin'])).dt.total_seconds()/300).to_numpy(dtype=int)-1;keys['queue_pred']=(q[h,keys.link_id.map(ii).to_numpy()]>=bt[s['condition']]).astype(int);parts.append(keys)
            frame=pd.concat(parts,ignore_index=True);checks[split+'/'+p]=validate_queue(frame,template);frame['panel']=p;frames.append(frame)
        frame=pd.concat(frames,ignore_index=True);assert len(frame)==87000;frame.to_csv(ROOT/f'queue-{split}.csv',index=False)
    assert guard('after')==before;assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items());save('final-verification.json',dict(passed=True,rows=174000,checks=checks,frozen_hashes_unchanged=True,condition_forecast_frozen=condition_frozen,all_thresholds_identical=True,Task1_package_unchanged=True,official_S_queue=None));progress('complete',new_phase_proxy=reports['new_phase_population_selected']['proxy_IoU'],old410_proxy=reports['old410_selected']['proxy_IoU'],seconds=time.time()-started)
if __name__=='__main__':
    import msvcrt
    lock=open(ROOT/'run.lock','a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise SystemExit('Already running')
    try:main()
    except Exception:save('failure.json',dict(traceback=traceback.format_exc()));progress('failed');raise
