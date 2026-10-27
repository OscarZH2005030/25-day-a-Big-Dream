"""Shared interval distribution: singleton-label control vs partial-label loss."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='4'
import sys,json,time,hashlib,traceback,gc,copy
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent;PROJECT=ROOT.parents[1]
sys.path[1:1]=[str(OUT/n) for n in ['task2-round18-interval-likelihood','task2-round17-onset-hazard','task2-round15-ongoing-history','task2-round14-history-repair','task2-round13-seed-stability','task2-round12-phase-soft','task2-round11-phase-validation','task2-round10-phase-coverage','task2-round9-hierarchical','task2-round8-soft-targets','task2-round7-structured','task2-round6-ramp-physics','task2-round5-distribution','task2-round2']]
from interval_runtime import IntervalPredictor
from coverage_data import matrix
from interval_data import features,build
from interval_core import PartialObjective,queue_probability,INTERVALS
from run_round2 import sample,arrays,role_samples,score,DATA,PANELS
from task2q.data import make_sample,released_sample
from task2q.cache import atomic_json
from task2q.metrics import window_iou
from task2q.submission import validate_queue
SEED=20261027;WORK=PROJECT/'work/round18-interval-likelihood';assert (WORK/'complete.json').exists(), 'Require completed read-only R18 cache'
TASK1=Path(r'C:\Users\18049\Documents\Codex\2026-10-24\trafficflowbench-186-global-road-adaptation\outputs\TrafficFlowBench-Best-20261026')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(n,v):atomic_json(ROOT/n,v)
def progress(phase,**kw):
    v=dict(phase=phase,pid=os.getpid(),utc=pd.Timestamp.now(tz='UTC').isoformat(),**kw);save('progress.json',v);print(json.dumps(v),flush=True)
def guard(stage):
    m=read(TASK1/'package-manifest.json');h={n:hashlib.sha256((TASK1/n).read_bytes()).hexdigest() for n in m['files']};assert all(h[n]==v['sha256'] for n,v in m['files'].items());save('task1-guard-'+stage+'.json',dict(passed=True,files=len(h),hashes=h,historical_S_state=.942828633,new_Task1_score_measured=False));return h
def fit(name,data,meta,early,kind,rounds):
    path=ROOT/(name+'.txt')
    if path.exists():return lgb.Booster(model_file=str(path))
    if pd.Timestamp.now(tz='UTC')>=pd.Timestamp('2026-10-27T15:00:00Z') or (PROJECT/'work/STOP_NIGHT').exists():raise RuntimeError('New-training cutoff or STOP_NIGHT reached')
    selected=np.ones(len(data['code']),bool)
    if early:selected&=data['early'].astype(bool)
    if kind=='control':selected&=(data['code']&(data['code']-1))==0
    w=np.asarray(data['w'][selected]).copy()
    if early:
        allcounts={b:sum(r['positive']==b for r in meta['windows']) for b in [False,True]};earlycounts={b:sum(r['positive']==b and r['early'] for r in meta['windows']) for b in [False,True]};w*=np.where(data['positive'][selected]>.5,allcounts[True]/earlycounts[True],allcounts[False]/earlycounts[False])
    X=np.asarray(data['X'][selected]);codes=np.asarray(data['code'][selected]);assert np.array_equal(codes,codes.astype('float32').astype('uint32'));obj=PartialObjective()
    def objective(raw,dataset):return obj(raw,dataset)
    cp=ROOT/(name+'-checkpoint.txt');parent=OUT/'task2-round18-interval-likelihood'/(name+'.txt') if name.endswith('-probe') else None;start=cp if cp.exists() else parent;initial=lgb.Booster(model_file=str(start)) if start is not None and start.exists() else None;done=initial.current_iteration() if initial else 0
    def callback(env):
        if (env.iteration+1)%20==0:env.model.save_model(str(cp));progress('training',model=name,boost_round=env.iteration+1,total_rounds=rounds,trees=env.model.num_trees(),rows=len(codes),weighted_partial_loss=obj.latest_loss)
    progress('starting_shared_model',model=name,rows=len(codes),features=X.shape[1],classes=22,rounds=rounds)
    model=lgb.train(dict(objective=objective,num_class=22,num_leaves=31,learning_rate=.05,min_data_in_leaf=80,feature_fraction=.85,lambda_l2=4,num_threads=4,verbosity=-1,seed=SEED,deterministic=True,force_col_wise=True,metric='None'),lgb.Dataset(X,label=codes.astype('float32'),weight=w/w.mean()),num_boost_round=rounds-done,init_model=initial,callbacks=[callback]) if done<rounds else initial;model.save_model(str(path));save(name+'-training-audit.json',dict(rows=len(codes),singleton_labels=int(((codes&(codes-1))==0).sum()),ambiguous_labels=int(((codes&(codes-1))!=0).sum()),classes=22,boost_rounds=model.current_iteration(),trees=model.num_trees(),last_training_loss=obj.latest_loss,custom_weights_applied_once=True));del model,X,codes,obj;gc.collect();return lgb.Booster(model_file=str(path))
def metric(ss,ps,t):return float(np.mean([window_iou(s['y'],p>=t,s['m']) for s,p in zip(ss,ps)]))
def main():
    started=time.time();before=guard('before');assert read(ROOT/'math-verification.json')['passed'];baseline=IntervalPredictor(OUT/'task2-round18-interval-likelihood');bt=baseline.rule['thresholds'];save('experiment.json',dict(seed=SEED,hypothesis='R18 learning curves were still improving strongly at60 rounds; a bounded200-round optimization budget test, preserving old failed gate',gate=read(ROOT/'budget-gate.json'),parent='R18 two60-round June-September probes continued to200; full models trained afresh if temporal guard passes',continuation_limit='exactly200, no larger-round or threshold search; warm continuation need not match uninterrupted stochastic feature sampling',features='same R18 557 causal features, exact same cached rows/codes/IPW and custom objective',control='singleton labels',partial='all informative compatible interval sets; no new labels or changed ambiguity definition',models_max=4,probe_total_rounds=200,full_rounds=200,classes=22,total_trees_in_models_max=17600,new_tree_training_max=14960,reused_parent_trees=2640,parameters='same31leaves,.05learning,.85feature_fraction,lambdaL2=4,minleaf80,seed20261027; positive diagonal curvature2p(1-p)',CPU_threads=4,temporal='same October-November original windows; median gain vs R10 direct at .1/.2/.35 >=-.01, compatibility>=.98',December='same fixed200/alpha .5or1, gain>=.004,worstphase>=-.015,min4perphase',January='only globalalpha0/.5/1, all thresholds frozen',Ongoing='R18 entire path frozen',February='both repeated populations report only after freeze',Task1='92files unchanged; historical .942828633 not newly measured',official_S_queue=None))
    assert read(ROOT/'budget-gate.json')['passed'];parent_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'task2-round18-interval-likelihood').glob('global-*-probe.txt')};save('warmstart-provenance.json',dict(parent_probe_sha256=parent_hashes,parent_frozen_sha256=hashlib.sha256((OUT/'task2-round18-interval-likelihood/frozen-calibration.json').read_bytes()).hexdigest(),cache_metadata_sha256=hashlib.sha256((WORK/'complete.json').read_bytes()).hexdigest(),source_code_sha256={n:hashlib.sha256((OUT/'task2-round18-interval-likelihood'/n).read_bytes()).hexdigest() for n in ['interval_core.py','interval_data.py']}))
    entries=read(OUT/'task2-round10-phase-coverage/expanded-fit.json')['samples'];causal=[]
    for p in PANELS:
        e=next(e for e in entries if e['panel']==p and e['condition']==0);s=sample(e);v=copy.deepcopy(arrays(e['day_key']));sl=e['origin_slot']
        for k in ['speed','flow','occupancy','quality','origin_speed','origin_flow','origin_occupancy','origin_quality']:v[k][sl+1:]=999
        v['eligible'][sl+1:]=False;s2=make_sample(v,s['network'],sl,p);assert np.array_equal(features(s),features(s2));s3=dict(s,y=1-s['y'],m=~s['m'],soft_y=np.full_like(s['soft_y'],999),speed_y=np.full_like(s['speed_y'],999),label_coverage=0.);assert np.array_equal(features(s),features(s3));causal.append(dict(panel=p,condition=0,passed=True))
    save('feature-causality-check.json',dict(passed=True,groups=causal,labels_and_future_metadata_not_inputs=True));data,meta=build(entries,WORK,progress);save('training-data-audit.json',{k:v for k,v in meta.items() if k!='windows'});panel_stats={a['panel']:a for a in meta['panels']};probes={k:fit('global-'+k+'-probe',data,meta,True,k,200) for k in ['control','partial']};hold_reports=[];guards={}
    for p in PANELS:
        hold=[sample(e) for e in entries if e['panel']==p and e['condition']==0 and e['source']=='original' and e['origin'][:10]>='2030-10-01' and e['has_future_queue']];legacy=lgb.Booster(model_file=str(OUT/f'task2-round10-phase-coverage/{p}-0-probe-expanded.txt'));base=[legacy.predict(matrix(s),num_threads=4).astype('float32').reshape(6,-1) for s in hold];xx=[features(s) for s in hold]
        for kind in ['control','partial']:
            q=[queue_probability(probes[kind],x) for x in xx];rr=[dict(threshold=t,direct_IoU=metric(hold,base,t),interval_IoU=metric(hold,q,t)) for t in [.1,.2,.35]];gain=float(np.median([r['interval_IoU']-r['direct_IoU'] for r in rr]));okay=bool(gain>=-.01 and panel_stats[p]['compatible_positive_fraction']>=.98);guards[(p,kind)]=okay;hold_reports.append(dict(panel=p,kind=kind,windows=len(hold),reports=rr,median_gain=gain,compatible_positive_fraction=panel_stats[p]['compatible_positive_fraction'],deployment_guard_passed=okay))
        save('temporal-holdout.json',hold_reports);del xx,base,legacy;gc.collect()
    del probes;gc.collect();models={kind:fit('global-'+kind,data,meta,False,kind,200) for kind in ['control','partial'] if any(guards[(p,kind)] for p in PANELS)};del data;gc.collect();tune=[sample(e) for e in read(OUT/'task2-round11-phase-validation/tune-phase-index.json')['samples']];routes={};dev=[]
    for p in PANELS:
        ss=[s for s in tune if s['panel']==p and s['condition']==0];phase=np.array([(pd.Timestamp(s['origin']).hour*12+pd.Timestamp(s['origin']).minute//5)%3 for s in ss]);base=[baseline.probability(s) for s in ss];bv=metric(ss,base,bt[0]);candidates=[dict(kind='baseline',alpha_new=0.,proxy=bv,phase_gains=[0.,0.,0.],eligible=True)];arm_scores={}
        for kind,model in models.items():
            raw=[queue_probability(model,features(s)) for s in ss];arm_scores[kind]=metric(ss,raw,bt[0])
            for alpha in [.5,1.]:
                ps=[alpha*a+(1-alpha)*b for a,b in zip(raw,base)];gains=[];counts=[]
                for ph in [0,1,2]:
                    ii=np.flatnonzero(phase==ph);counts.append(len(ii));gains.append(metric([ss[i] for i in ii],[ps[i] for i in ii],bt[0])-metric([ss[i] for i in ii],[base[i] for i in ii],bt[0]) if len(ii) else -1.)
                value=metric(ss,ps,bt[0]);candidates.append(dict(kind=kind,alpha_new=alpha,proxy=value,phase_gains=gains,eligible=guards[(p,kind)] and min(counts)>=4 and value>=bv+.004 and min(gains)>=-.015))
        chosen=max((r for r in candidates if r['eligible']),key=lambda r:r['proxy']);routes[p]=chosen;dev.append(dict(panel=p,condition=0,baseline_proxy=bv,raw_arm_proxy=arm_scores,selected=chosen,candidates=candidates));save('december-selection.json',dev);save('routes.json',[dict(panel=p,**r) for p,r in routes.items()])
    def components(ss):
        base=[];adapt=[]
        for s in ss:
            b=baseline.probability(s);base.append(b)
            if s['condition']==1 or routes[s['panel']]['alpha_new']==0:adapt.append(b)
            else:
                r=routes[s['panel']];q=queue_probability(models[r['kind']],features(s));adapt.append(r['alpha_new']*q+(1-r['alpha_new'])*b)
        return base,adapt
    def mixed(ss,base,adapt,a):return [b if s['condition']==1 else a*q+(1-a)*b for s,b,q in zip(ss,base,adapt)]
    progress('calibration');cal=[sample(e) for e in read(OUT/'task2-round11-phase-validation/calibrate-phase-index.json')['samples']];base,adapt=components(cal);candidates=[dict(alpha_interval=a,thresholds=bt,proxy=score(cal,mixed(cal,base,adapt,a),bt)['proxy_IoU']) for a in [0.,.5,1.]];chosen=max(candidates,key=lambda r:r['proxy']);hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.txt')};save('frozen-calibration.json',dict(selected=chosen,candidates=candidates,hashes=hashes,route_sha256=hashlib.sha256((ROOT/'routes.json').read_bytes()).hexdigest(),baseline_frozen_sha256=hashlib.sha256((OUT/'task2-round18-interval-likelihood/frozen-calibration.json').read_bytes()).hexdigest(),interval_source_sha256={n:hashlib.sha256((OUT/'task2-round18-interval-likelihood'/n).read_bytes()).hexdigest() for n in ['interval_core.py','interval_data.py']},ongoing_frozen=True,all_thresholds_frozen=True,February_previously_used_report_only=True,Task1_frozen=True))
    progress('evaluating');reports={};records=[]
    for population,ss in [('new_phase_population',[sample(e) for e in read(OUT/'task2-round11-phase-validation/audit-phase-index.json')['samples']]),('old410',role_samples('audit'))]:
        base,adapt=components(ss);final=mixed(ss,base,adapt,chosen['alpha_interval']);reports[population+'_round18']=score(ss,base,bt);reports[population+'_selected']=score(ss,final,bt)
        for s,b,q in zip(ss,base,final):
            if s['condition']==1:assert np.array_equal(b,q)
            if s['has_future_queue']:records.append(dict(population=population,panel=s['panel'],condition=s['condition'],origin=s['origin'],date=s['origin'][:10],base_iou=window_iou(s['y'],b>=bt[s['condition']],s['m']),selected_iou=window_iou(s['y'],q>=bt[s['condition']],s['m'])))
        assert abs(reports[population+'_round18']['proxy_IoU']-read(OUT/'task2-round18-interval-likelihood/proxy-results.json')['reports'][population+'_selected']['proxy_IoU'])<1e-12
    save('paired-window-metrics.json',records);save('proxy-results.json',dict(official_S_queue=None,reports=reports,selected=chosen,new_onset_roads=sum(r['alpha_new']>0 for r in routes.values()),selected_arms={kind:sum(r['alpha_new']>0 and r['kind']==kind for r in routes.values()) for kind in ['control','partial']},seconds=time.time()-started,ongoing_predictions_identical=True,all_thresholds_identical=True,different_populations_not_directly_comparable=True,Task1_historical_S_state=.942828633,Task1_score_newly_measured=False))
    progress('predicting_official');checks={}
    for split in ['validation','private']:
        frames=[]
        for p in PANELS:
            d=DATA/'task2'/p/split;index=pd.read_csv(d/'window_index.csv');template=pd.read_csv(d/'sample_submission_queue.csv');parts=[]
            for row in index.itertuples():
                s=released_sample(DATA,p,split,row.window_id);bb,aa=components([s]);q=mixed([s],bb,aa,chosen['alpha_interval'])[0];keys=template[template.window_id==row.window_id].copy();ii={l:i for i,l in enumerate(s['network']['link_ids'])};h=((pd.to_datetime(keys.timestamp,utc=True)-pd.Timestamp(s['origin'])).dt.total_seconds()/300).to_numpy(dtype=int)-1;keys['queue_pred']=(q[h,keys.link_id.map(ii).to_numpy()]>=bt[s['condition']]).astype(int);parts.append(keys)
            frame=pd.concat(parts,ignore_index=True);checks[split+'/'+p]=validate_queue(frame,template);frame['panel']=p;frames.append(frame)
        frame=pd.concat(frames,ignore_index=True);assert len(frame)==87000;frame.to_csv(ROOT/f'queue-{split}.csv',index=False)
    assert guard('after')==before;assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items());assert all(hashlib.sha256((OUT/'task2-round18-interval-likelihood'/n).read_bytes()).hexdigest()==h for n,h in parent_hashes.items());assert hashlib.sha256((WORK/'complete.json').read_bytes()).hexdigest()==read(ROOT/'warmstart-provenance.json')['cache_metadata_sha256'];save('parent-protection.json',dict(passed=True,read_only_parent_probes_unchanged=True,read_only_cache_metadata_unchanged=True));save('final-verification.json',dict(passed=True,rows=174000,checks=checks,frozen_hashes_unchanged=True,ongoing_predictions_identical=True,all_thresholds_identical=True,Task1_package_unchanged=True,official_S_queue=None));progress('complete',new_phase_proxy=reports['new_phase_population_selected']['proxy_IoU'],old410_proxy=reports['old410_selected']['proxy_IoU'],seconds=time.time()-started)
if __name__=='__main__':
    import msvcrt
    lock=open(ROOT/'run.lock','a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise SystemExit('Already running')
    try:main()
    except Exception:save('failure.json',dict(traceback=traceback.format_exc()));progress('failed');raise
