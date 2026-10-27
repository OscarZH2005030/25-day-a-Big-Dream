"""Static public link category, with exact-row and negative-window controls."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='4'
import sys,json,time,hashlib,traceback,gc,copy
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent;PROJECT=ROOT.parents[1]
sys.path[1:1]=[str(OUT/n) for n in ['task2-round19-interval-budget','task2-round18-interval-likelihood','task2-round17-onset-hazard','task2-round15-ongoing-history','task2-round14-history-repair','task2-round13-seed-stability','task2-round12-phase-soft','task2-round11-phase-validation','task2-round10-phase-coverage','task2-round9-hierarchical','task2-round8-soft-targets','task2-round7-structured','task2-round6-ramp-physics','task2-round5-distribution','task2-round2']]
from budget_runtime import BudgetPredictor
from category_data import features,build,original_matrix
from run_round2 import sample,arrays,role_samples,score,DATA,PANELS
from task2q.data import make_sample,released_sample
from task2q.cache import atomic_json
from task2q.metrics import window_iou
from task2q.submission import validate_queue
SEED=20261027;WORK=PROJECT/'work/round21-link-category';WORK.mkdir(parents=True,exist_ok=True)
TASK1=Path(r'C:\Users\18049\Documents\Codex\2026-10-24\trafficflowbench-186-global-road-adaptation\outputs\TrafficFlowBench-Best-20261026')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(n,v):atomic_json(ROOT/n,v)
def progress(phase,**kw):
    v=dict(phase=phase,pid=os.getpid(),utc=pd.Timestamp.now(tz='UTC').isoformat(),**kw);save('progress.json',v);print(json.dumps(v),flush=True)
def guard(stage):
    m=read(TASK1/'package-manifest.json');h={n:hashlib.sha256((TASK1/n).read_bytes()).hexdigest() for n in m['files']};assert all(h[n]==v['sha256'] for n,v in m['files'].items());save('task1-guard-'+stage+'.json',dict(passed=True,files=len(h),hashes=h,historical_S_state=.942828633,new_Task1_score_measured=False));return h
def fit(name,X,y,w,rounds):
    path=ROOT/(name+'.txt')
    if path.exists():return lgb.Booster(model_file=str(path))
    if pd.Timestamp.now(tz='UTC')>=pd.Timestamp('2026-10-27T15:00:00Z') or (PROJECT/'work/STOP_NIGHT').exists():raise RuntimeError('New-training cutoff or STOP_NIGHT reached')
    cp=ROOT/(name+'-checkpoint.txt');initial=lgb.Booster(model_file=str(cp)) if cp.exists() else None;done=initial.current_iteration() if initial else 0
    def callback(env):
        if (env.iteration+1)%100==0:env.model.save_model(str(cp));progress('training',model=name,step=env.iteration+1,total=rounds)
    model=lgb.train(dict(objective='binary',num_leaves=31,learning_rate=.035,min_data_in_leaf=80,feature_fraction=.85,lambda_l2=4,num_threads=4,verbosity=-1,seed=SEED,deterministic=True,force_col_wise=True,cat_smooth=20,cat_l2=20,min_data_per_group=50,max_cat_to_onehot=4),lgb.Dataset(X,label=y,weight=w/w.mean(),categorical_feature=[555]),num_boost_round=rounds-done,init_model=initial,callbacks=[callback]) if done<rounds else initial;model.save_model(str(path));info=model.dump_model()['feature_infos']['Column_555'];save(name+'-category-audit.json',dict(features=model.num_feature(),category_splits=int(model.feature_importance()[555]),learned_category_values=info.get('values',[]),known_static_ID_only=True));return model
def metric(ss,ps,t):return float(np.mean([window_iou(s['y'],p>=t,s['m']) for s,p in zip(ss,ps)]))
def negative_stats(ss,ps,t):
    groups={}
    for s,p in zip(ss,ps):
        assert not s['has_future_queue'] and s['condition']==0;fp=(p>=t)&s['m'];groups.setdefault(s['panel'],[]).append([float(fp.any()),float(fp.sum()/s['m'].sum())])
    if not groups:return None
    return dict(windows=len(ss),panels=len(groups),false_any_macro=float(np.mean([np.mean(v,axis=0)[0] for v in groups.values()])),false_eligible_fraction_macro=float(np.mean([np.mean(v,axis=0)[1] for v in groups.values()])))
def negative_ok(new,base):return bool(new is not None and base is not None and new['false_any_macro']<=base['false_any_macro']+.03 and new['false_eligible_fraction_macro']<=base['false_eligible_fraction_macro']+.005)
def main():
    started=time.time();before=guard('before');recurrence=read(OUT/'task2-round20-link-recurrence/training-gates.json')['roads'];baseline=BudgetPredictor(OUT/'task2-round19-interval-budget');bt=baseline.rule['thresholds'];save('experiment.json',dict(seed=SEED,hypothesis='R20 time-separated link recurrence is strong; known static link category may expose risk interaction missed by numeric metadata',inputs='same555 causal features +1 categorical public link index; no learned target prior, no future quality/coverage or Task1 offline input',targets_sampling='exact R10 expanded rows, all observed queue positives/IPW negatives, unchanged y/w',max_models=16,max_tree_rounds=6400,early_rounds=200,full_rounds=600,num_leaves=31,category=dict(index=555,cat_smooth=20,cat_l2=20,min_data_per_group=50,max_cat_to_onehot=4),fit='June-November; probe June-September',temporal='October-November original positive windows median gain vs R10 direct at .1/.2/.35 >=-.01',negative_guard='at unchanged threshold.5, macro false-any increase<=.03 and eligible-cell false-positive fraction increase<=.005; early time-holdout and January global blend selection',full_model_gate='R20 recurrence, temporal guard, negative guard, actual category split used, >=4 December windows/phase',December='fixed600, alpha .5/1, mean gain>=.004 and worstphase>=-.015',January='globalalpha0/.5/1 only, original thresholds frozen; same negative guard applied globally',Ongoing='entire R19 path and threshold frozen',Task1='92files frozen, historical .942828633 not newly measured',CPU_threads=4,official_S_queue=None))
    entries=read(OUT/'task2-round10-phase-coverage/expanded-fit.json')['samples'];tune=[sample(e) for e in read(OUT/'task2-round11-phase-validation/tune-phase-index.json')['samples']];vocabulary={p:[str(l) for l in arrays(next(e['network_key'] for e in entries if e['panel']==p))['link_ids']] for p in PANELS};save('link-vocabulary.json',vocabulary);models={};routes={};dev=[];hold_reports=[];audits=[];causal=[]
    for p in PANELS:
        e=next(e for e in entries if e['panel']==p and e['condition']==0);s=sample(e);v=copy.deepcopy(arrays(e['day_key']));sl=e['origin_slot']
        for k in ['speed','flow','occupancy','quality','origin_speed','origin_flow','origin_occupancy','origin_quality']:v[k][sl+1:]=999
        v['eligible'][sl+1:]=False;s2=make_sample(v,s['network'],sl,p);assert np.array_equal(features(s),features(s2));assert np.array_equal(features(s)[:,:555],original_matrix(s));assert np.array_equal(features(s)[:,555],np.tile(np.arange(s['x'].shape[1]),6));causal.append(dict(panel=p,condition=0,passed=True))
    save('feature-causality-check.json',dict(passed=True,groups=causal,static_public_ID_not_target_encoding=True))
    for p in PANELS:
        ss=[s for s in tune if s['panel']==p and s['condition']==0];phase=np.array([(pd.Timestamp(s['origin']).hour*12+pd.Timestamp(s['origin']).minute//5)%3 for s in ss]);phase_ok=min(int((phase==k).sum()) for k in [0,1,2])>=4;base=[baseline.probability(s) for s in ss];bv=metric(ss,base,bt[0]);fallback=dict(alpha_new=0.,proxy=bv,phase_gains=[0.,0.,0.],eligible=True)
        if not recurrence[p]:routes[p]=fallback;dev.append(dict(panel=p,selected=fallback,reason='recurrence audit gate failed'));save('december-selection.json',dev);save('routes.json',[dict(panel=p,**r) for p,r in routes.items()]);continue
        progress('building_static_category',panel=p,completed=len(routes),total=8);data,meta,audit=build(entries,p,WORK,PROJECT,progress);audits.append(audit);save('training-data-audit.json',audits);selected=data['early'].astype(bool);windows=[r for r in meta['windows'] if r['early']];counts={b:sum(r['positive']==b for r in windows) for b in [False,True]};allcounts={b:sum(r['positive']==b for r in meta['windows']) for b in [False,True]};weights=np.asarray(data['w'][selected])*np.where(data['window_positive'][selected]>.5,allcounts[True]/counts[True],allcounts[False]/counts[False]);probe=fit(p+'-probe',np.asarray(data['X'][selected]),np.asarray(data['y'][selected]),weights,200);legacy=lgb.Booster(model_file=str(OUT/f'task2-round10-phase-coverage/{p}-0-probe-expanded.txt'));hold=[sample(e) for e in entries if e['panel']==p and e['condition']==0 and e['source']=='original' and e['origin'][:10]>='2030-10-01'];bp=[];qp=[]
        for s in hold:
            xx=features(s);bp.append(legacy.predict(xx[:,:555],num_threads=4).astype('float32').reshape(6,-1));qp.append(probe.predict(xx,num_threads=4).astype('float32').reshape(6,-1))
        positive=[i for i,s in enumerate(hold) if s['has_future_queue']];negative=[i for i,s in enumerate(hold) if not s['has_future_queue']];hs=[hold[i] for i in positive];bpos=[bp[i] for i in positive];qpos=[qp[i] for i in positive];rr=[dict(threshold=t,control_IoU=metric(hs,bpos,t),category_IoU=metric(hs,qpos,t)) for t in [.1,.2,.35]];median=float(np.median([r['category_IoU']-r['control_IoU'] for r in rr]));nb=negative_stats([hold[i] for i in negative],[bp[i] for i in negative],bt[0]);nq=negative_stats([hold[i] for i in negative],[qp[i] for i in negative],bt[0]);ng=negative_ok(nq,nb);used=int(probe.feature_importance()[555]);temporal=bool(median>=-.01 and ng and used>0);eligible=bool(temporal and phase_ok);hold_reports.append(dict(panel=p,positive_windows=len(positive),negative_windows=len(negative),reports=rr,median_gain=median,negative_control=nb,negative_category=nq,negative_guard_passed=ng,category_splits=used,temporal_guard_passed=temporal,December_phase_coverage_passed=phase_ok,full_training_guard_passed=eligible));save('temporal-holdout.json',hold_reports);del bp,qp,probe,legacy;gc.collect()
        if not eligible:
            routes[p]=fallback;dev.append(dict(panel=p,selected=fallback,reason='time/negative/category-use/phase gate failed'));save('december-selection.json',dev);save('routes.json',[dict(panel=p,**r) for p,r in routes.items()]);del data;gc.collect();continue
        models[p]=fit(p+'-category',data['X'],data['y'],data['w'],600);del data;gc.collect();raw=[models[p].predict(features(s),num_threads=4).astype('float32').reshape(6,-1) for s in ss];candidates=[fallback]
        for alpha in [.5,1.]:
            ps=[alpha*a+(1-alpha)*b for a,b in zip(raw,base)];gains=[]
            for ph in [0,1,2]:
                ii=np.flatnonzero(phase==ph);gains.append(metric([ss[i] for i in ii],[ps[i] for i in ii],bt[0])-metric([ss[i] for i in ii],[base[i] for i in ii],bt[0]))
            value=metric(ss,ps,bt[0]);candidates.append(dict(alpha_new=alpha,proxy=value,phase_gains=gains,eligible=value>=bv+.004 and min(gains)>=-.015 and int(models[p].feature_importance()[555])>0))
        chosen=max((r for r in candidates if r['eligible']),key=lambda r:r['proxy']);routes[p]=chosen;dev.append(dict(panel=p,baseline_proxy=bv,selected=chosen,candidates=candidates));save('december-selection.json',dev);save('routes.json',[dict(panel=p,**r) for p,r in routes.items()])
    def components(ss):
        base=[];adapt=[]
        for s in ss:
            b=baseline.probability(s);base.append(b)
            if s['condition']==1 or routes[s['panel']]['alpha_new']==0:adapt.append(b)
            else:
                assert [str(l) for l in s['network']['link_ids']]==vocabulary[s['panel']];q=models[s['panel']].predict(features(s),num_threads=4).astype('float32').reshape(6,-1);a=routes[s['panel']]['alpha_new'];adapt.append(a*q+(1-a)*b)
        return base,adapt
    def mixed(ss,base,adapt,a):return [b if s['condition']==1 else a*q+(1-a)*b for s,b,q in zip(ss,base,adapt)]
    progress('calibration');cal=[sample(e) for e in read(OUT/'task2-round11-phase-validation/calibrate-phase-index.json')['samples']];base,adapt=components(cal);negative=[s for s in role_samples('calibrate') if s['condition']==0 and not s['has_future_queue']];nb,na=components(negative);nbase=negative_stats(negative,nb,bt[0]);candidates=[]
    for alpha in [0.,.5,1.]:
        ns=negative_stats(negative,mixed(negative,nb,na,alpha),bt[0]);candidates.append(dict(alpha_category=alpha,thresholds=bt,proxy=score(cal,mixed(cal,base,adapt,alpha),bt)['proxy_IoU'],negative_stats=ns,eligible=alpha==0 or negative_ok(ns,nbase)))
    chosen=max((r for r in candidates if r['eligible']),key=lambda r:r['proxy']);hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.txt')};save('frozen-calibration.json',dict(selected=chosen,candidates=candidates,hashes=hashes,route_sha256=hashlib.sha256((ROOT/'routes.json').read_bytes()).hexdigest(),vocabulary_sha256=hashlib.sha256((ROOT/'link-vocabulary.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256((ROOT/'category_data.py').read_bytes()).hexdigest(),baseline_frozen_sha256=hashlib.sha256((OUT/'task2-round19-interval-budget/frozen-calibration.json').read_bytes()).hexdigest(),ongoing_frozen=True,all_thresholds_frozen=True,February_previously_used_report_only=True,Task1_frozen=True))
    progress('evaluating');reports={};records=[]
    for population,ss in [('new_phase_population',[sample(e) for e in read(OUT/'task2-round11-phase-validation/audit-phase-index.json')['samples']]),('old410',role_samples('audit'))]:
        base,adapt=components(ss);final=mixed(ss,base,adapt,chosen['alpha_category']);reports[population+'_round19']=score(ss,base,bt);reports[population+'_selected']=score(ss,final,bt)
        for s,b,q in zip(ss,base,final):
            if s['condition']==1:assert np.array_equal(b,q)
            if s['has_future_queue']:records.append(dict(population=population,panel=s['panel'],condition=s['condition'],origin=s['origin'],date=s['origin'][:10],base_iou=window_iou(s['y'],b>=bt[s['condition']],s['m']),selected_iou=window_iou(s['y'],q>=bt[s['condition']],s['m'])))
        assert abs(reports[population+'_round19']['proxy_IoU']-read(OUT/'task2-round19-interval-budget/proxy-results.json')['reports'][population+'_selected']['proxy_IoU'])<1e-12
    save('paired-window-metrics.json',records);save('proxy-results.json',dict(official_S_queue=None,reports=reports,selected=chosen,new_onset_roads=sum(r['alpha_new']>0 for r in routes.values()),seconds=time.time()-started,ongoing_predictions_identical=True,all_thresholds_identical=True,different_populations_not_directly_comparable=True,Task1_historical_S_state=.942828633,Task1_score_newly_measured=False))
    progress('predicting_official');checks={}
    for split in ['validation','private']:
        frames=[]
        for p in PANELS:
            d=DATA/'task2'/p/split;index=pd.read_csv(d/'window_index.csv');template=pd.read_csv(d/'sample_submission_queue.csv');parts=[]
            for row in index.itertuples():
                s=released_sample(DATA,p,split,row.window_id);bb,aa=components([s]);q=mixed([s],bb,aa,chosen['alpha_category'])[0];keys=template[template.window_id==row.window_id].copy();ii={l:i for i,l in enumerate(s['network']['link_ids'])};h=((pd.to_datetime(keys.timestamp,utc=True)-pd.Timestamp(s['origin'])).dt.total_seconds()/300).to_numpy(dtype=int)-1;keys['queue_pred']=(q[h,keys.link_id.map(ii).to_numpy()]>=bt[s['condition']]).astype(int);parts.append(keys)
            frame=pd.concat(parts,ignore_index=True);checks[split+'/'+p]=validate_queue(frame,template);frame['panel']=p;frames.append(frame)
        frame=pd.concat(frames,ignore_index=True);assert len(frame)==87000;frame.to_csv(ROOT/f'queue-{split}.csv',index=False)
    assert guard('after')==before;assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items());save('final-verification.json',dict(passed=True,rows=174000,checks=checks,frozen_hashes_unchanged=True,ongoing_predictions_identical=True,all_thresholds_identical=True,Task1_package_unchanged=True,official_S_queue=None));progress('complete',new_phase_proxy=reports['new_phase_population_selected']['proxy_IoU'],old410_proxy=reports['old410_selected']['proxy_IoU'],seconds=time.time()-started)
if __name__=='__main__':
    import msvcrt
    lock=open(ROOT/'run.lock','a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise SystemExit('Already running')
    try:main()
    except Exception:save('failure.json',dict(traceback=traceback.format_exc()));progress('failed');raise
