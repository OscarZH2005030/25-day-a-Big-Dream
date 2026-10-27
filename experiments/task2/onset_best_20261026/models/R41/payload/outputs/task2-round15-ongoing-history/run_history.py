"""Condition-specific reconstruction features: Ongoing only, Onset frozen."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import sys,json,time,hashlib,traceback,gc,copy
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent;PROJECT=ROOT.parents[1]
sys.path[1:1]=[str(OUT/n) for n in ['task2-round14-history-repair','task2-round13-seed-stability','task2-round12-phase-soft','task2-round11-phase-validation','task2-round10-phase-coverage','task2-round9-hierarchical','task2-round8-soft-targets','task2-round7-structured','task2-round6-ramp-physics','task2-round5-distribution','task2-round2']]
from seed_runtime import SeedPredictor
from coverage_data import matrix as old_matrix
from history_repair import extras as repair_extras
from special_features import latest_state
from run_round2 import sample,arrays,role_samples,score,DATA,PANELS
from task2q.data import make_sample,released_sample
from task2q.cache import atomic_json
from task2q.metrics import window_iou
from task2q.submission import validate_queue
SEED=20261027;WORK=PROJECT/'work/round15-ongoing-history';WORK.mkdir(parents=True,exist_ok=True)
TASK1=Path(r'C:\Users\18049\Documents\Codex\2026-10-24\trafficflowbench-186-global-road-adaptation\outputs\TrafficFlowBench-Best-20261026')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(n,v):atomic_json(ROOT/n,v)
def progress(phase,**kw):
    v=dict(phase=phase,pid=os.getpid(),utc=pd.Timestamp.now(tz='UTC').isoformat(),**kw);save('progress.json',v);print(json.dumps(v),flush=True)
def guard(stage):
    m=read(TASK1/'package-manifest.json');h={n:hashlib.sha256((TASK1/n).read_bytes()).hexdigest() for n in m['files']};assert all(h[n]==v['sha256'] for n,v in m['files'].items());save('task1-guard-'+stage+'.json',dict(passed=True,files=len(h),hashes=h,historical_S_state=.942828633,new_Task1_score_measured=False));return h
def matrix(s):return np.concatenate([old_matrix(s),repair_extras(s)],axis=1)
def allowed_new_training():
    return pd.Timestamp.now(tz='UTC')<pd.Timestamp('2026-10-27T15:00:00Z') and not (PROJECT/'work/STOP_NIGHT').exists()
def fit(name,X,y,w,rounds,leaves=63):
    path=ROOT/(name+'.txt')
    if path.exists():return lgb.Booster(model_file=str(path))
    checkpoint=ROOT/(name+'-checkpoint.txt');initial=lgb.Booster(model_file=str(checkpoint)) if checkpoint.exists() else None;done=initial.current_iteration() if initial else 0
    if not allowed_new_training():raise RuntimeError('New-training cutoff or STOP_NIGHT reached')
    def callback(env):
        if (env.iteration+1)%100==0:env.model.save_model(str(checkpoint));progress('training',model=name,step=env.iteration+1,total=rounds)
    model=lgb.train(dict(objective='binary',num_leaves=leaves,learning_rate=.035,min_data_in_leaf=80,feature_fraction=.85,lambda_l2=4,num_threads=4,verbosity=-1,seed=SEED,deterministic=True,force_col_wise=True),lgb.Dataset(X,label=y,weight=w/w.mean()),num_boost_round=rounds-done,init_model=initial,callbacks=[callback]) if done<rounds else initial
    model.save_model(str(path));return model
def group_data(entries,p):
    folder=WORK/p;folder.mkdir(exist_ok=True);base=PROJECT/f'work/round10-phase/{p}-1';data={k:np.load(base/(k+'.npy'),mmap_mode='r') for k in ['X','y','w','early','window_positive']};meta=read(base/'complete.json');done=folder/'complete.json'
    if done.exists():return data,np.load(folder/'X.npy',mmap_mode='r'),meta,read(done)
    dest=folder/'building.npy';X=np.lib.format.open_memmap(dest,mode='w+',dtype='float32',shape=(len(data['y']),691));lookup={(e['origin'],e['source']):e for e in entries if e['panel']==p and e['condition']==1};rngs={source:np.random.default_rng(SEED+offset+PANELS.index(p)*2+1) for source,offset in [('original',0),('added_phase',2000)]};nrows=0
    for i,win in enumerate(meta['windows']):
        e=lookup[(win['origin'],win['source'])];s=sample(e);m=s['m'].ravel();y=s['y'].ravel();last,queued=latest_state(s);flip=y.astype(bool)!=np.tile(queued,6);mandatory=(m&(y>0))|(m&flip);keep=np.flatnonzero(mandatory);negative=np.flatnonzero(m&~mandatory);risk=np.tile((last<.9)|(s['x'][-2,:,1]>.7)|(s['x'][-2,:,2]>.2),6);a=negative[risk[negative]];b=negative[~risk[negative]];ka=min(96,len(a));kb=min(192-ka,len(b));ka=min(192-kb,len(a));rng=rngs[win['source']];aa=rng.choice(a,ka,replace=False) if ka else np.array([],int);bb=rng.choice(b,kb,replace=False) if kb else np.array([],int);ids=np.concatenate([keep,aa,bb]);start,end=win['start'],win['end'];assert start==nrows and len(ids)==end-start;assert np.array_equal(y[ids],data['y'][start:end]);original=old_matrix(s)[ids];assert np.array_equal(original,data['X'][start:end]);X[start:end,:555]=data['X'][start:end];X[start:end,555:]=repair_extras(s)[ids];nrows=end
        if (i+1)%128==0:progress('replaying_exact_rows',panel=p,windows=i+1,total=len(meta['windows']),rows=nrows)
    assert nrows==len(data['y']) and np.isfinite(X).all();X.flush();del X;os.replace(dest,folder/'X.npy');audit=dict(panel=p,condition=1,windows=len(meta['windows']),rows=nrows,features=691,original555_features_labels_IPW_unchanged=True,all_row_features_exact_match=True,source_y_sha256=hashlib.sha256((base/'y.npy').read_bytes()).hexdigest(),source_w_sha256=hashlib.sha256((base/'w.npy').read_bytes()).hexdigest(),source_metadata_sha256=hashlib.sha256((base/'complete.json').read_bytes()).hexdigest());atomic_json(done,audit);return data,np.load(folder/'X.npy',mmap_mode='r'),meta,audit
def metric(ss,ps,t):return float(np.mean([window_iou(s['y'],p>=t,s['m']) for s,p in zip(ss,ps)]))
def main():
    started=time.time();before=guard('before');diagnostic=read(OUT/'task2-round14-history-repair/reconstruction-diagnostic.json');ongoing=[r for r in diagnostic['comparisons'] if r['condition']==1];gate=bool(np.mean([r['relative_improvement'] for r in ongoing])>=.02 and max(r['latest3_speed_ratio'] for r in ongoing)<=1.02 and max(r['latest3_flow_ratio'] for r in ongoing)<=1.02);assert gate
    save('experiment.json',dict(seed=SEED,hypothesis='Condition-specific bounded history reconstruction; all-condition R14 gate failed, only Ongoing satisfies the original reconstruction limits',reconstruction_gate_passed_ongoing=gate,onset='entire R13 probability and threshold frozen',fit_windows='same R10 expanded Ongoing origins and exact selected cells/IPW',features='original555 +136 reconstructed-history summaries; original missing flags remain',target='same hard future queue labels',models=8,full_rounds=600,temporal_probes=8,probe_rounds=200,total_tree_rounds=6400,control='reuse R10 June-September expanded probes200; fixed rows/weights/parameters except appended features',temporal_guard='median IoU gain at .1/.2/.35 >=-.01 on October-November original positive windows',December_guard='fixed600, alpha .5/1; >=4 per phase, gain>=.004 and worst phase>=-.015',January='global blend 0/.5/1 only; both thresholds frozen',CPU_threads=4,Task1='92file freeze; history reconstruction is not a new Task1 score',official_S_queue=None))
    entries=read(OUT/'task2-round10-phase-coverage/expanded-fit.json')['samples'];baseline=SeedPredictor(OUT/'task2-round13-seed-stability');bt=baseline.rule['thresholds'];tune=[sample(e) for e in read(OUT/'task2-round11-phase-validation/tune-phase-index.json')['samples']];models={};routes={};dev=[];hold_reports=[];audits=[];causal=[]
    for p in PANELS:
        e=next(e for e in entries if e['panel']==p and e['condition']==1);s=sample(e);v=copy.deepcopy(arrays(e['day_key']));slot=e['origin_slot']
        for k in ['speed','flow','occupancy','quality','origin_speed','origin_flow','origin_occupancy','origin_quality']:v[k][slot+1:]=999
        v['eligible'][slot+1:]=False;s2=make_sample(v,s['network'],slot,p);assert np.array_equal(matrix(s),matrix(s2));label_perturb=dict(s,y=1-s['y'],m=~s['m'],speed_y=np.full_like(s['speed_y'],999),soft_y=np.full_like(s['soft_y'],999),label_coverage=0.);assert np.array_equal(repair_extras(s),repair_extras(label_perturb));causal.append(dict(panel=p,condition=1,passed=True))
    save('feature-causality-check.json',dict(passed=True,groups=causal,forecast_future_and_label_metadata_do_not_change_inputs=True,history_boundary='T-60..T public observations only; no offline Task1 outputs'))
    for p in PANELS:
        progress('building_history_rows',panel=p,completed=len(routes),total=8);data,X,meta,audit=group_data(entries,p);audits.append(audit);save('training-data-audit.json',audits);selected=data['early'].astype(bool);windows=[r for r in meta['windows'] if r['early']];counts={b:sum(r['positive']==b for r in windows) for b in [False,True]};allcounts={b:sum(r['positive']==b for r in meta['windows']) for b in [False,True]};weights=np.asarray(data['w'][selected])*np.where(data['window_positive'][selected]>.5,allcounts[True]/counts[True],allcounts[False]/counts[False]);probe=fit(p+'-probe',np.asarray(X[selected]),np.asarray(data['y'][selected]),weights,200,31);control=lgb.Booster(model_file=str(OUT/f'task2-round10-phase-coverage/{p}-1-probe-expanded.txt'));hold=[sample(e) for e in entries if e['panel']==p and e['condition']==1 and e['source']=='original' and e['origin'][:10]>='2030-10-01' and e['has_future_queue']];bp=[];qp=[]
        for s in hold:
            xx=matrix(s);bp.append(control.predict(xx[:,:555],num_threads=4).astype('float32').reshape(6,-1));qp.append(probe.predict(xx,num_threads=4).astype('float32').reshape(6,-1))
        comparisons=[dict(threshold=t,control_IoU=metric(hold,bp,t),history_IoU=metric(hold,qp,t)) for t in [.1,.2,.35]];gain=float(np.median([r['history_IoU']-r['control_IoU'] for r in comparisons]));temporal=bool(gain>=-.01);hold_reports.append(dict(panel=p,condition=1,windows=len(hold),reports=comparisons,median_gain=gain,deployment_guard_passed=temporal));save('temporal-holdout.json',hold_reports);del bp,qp,probe,control;gc.collect();models[p]=fit(p+'-history',X,data['y'],data['w'],600);del data,X;gc.collect()
        ss=[s for s in tune if s['panel']==p and s['condition']==1];phase=np.array([(pd.Timestamp(s['origin']).hour*12+pd.Timestamp(s['origin']).minute//5)%3 for s in ss]);base=[baseline.probability(s) for s in ss];raw=[models[p].predict(matrix(s),num_threads=4).astype('float32').reshape(6,-1) for s in ss];bv=metric(ss,base,bt[1]);candidates=[dict(alpha_new=0.,proxy=bv,phase_gains=[0.,0.,0.],eligible=True)]
        for alpha in [.5,1.]:
            ps=[alpha*a+(1-alpha)*b for a,b in zip(raw,base)];gains=[];counts=[]
            for ph in [0,1,2]:
                ii=np.flatnonzero(phase==ph);counts.append(len(ii));gains.append(metric([ss[i] for i in ii],[ps[i] for i in ii],bt[1])-metric([ss[i] for i in ii],[base[i] for i in ii],bt[1]) if len(ii) else -1.)
            val=metric(ss,ps,bt[1]);candidates.append(dict(alpha_new=alpha,proxy=val,phase_gains=gains,eligible=temporal and min(counts)>=4 and val>=bv+.004 and min(gains)>=-.015))
        chosen=max((r for r in candidates if r['eligible']),key=lambda r:r['proxy']);routes[p]=chosen;dev.append(dict(panel=p,condition=1,baseline_proxy=bv,temporal_guard=temporal,selected=chosen,candidates=candidates));save('december-selection.json',dev);save('routes.json',[dict(panel=p,**r) for p,r in routes.items()])
    def components(ss):
        base=[];adapt=[]
        for s in ss:
            b=baseline.probability(s);base.append(b)
            if s['condition']==0 or routes[s['panel']]['alpha_new']==0:adapt.append(b)
            else:
                q=models[s['panel']].predict(matrix(s),num_threads=4).astype('float32').reshape(6,-1);a=routes[s['panel']]['alpha_new'];adapt.append(a*q+(1-a)*b)
        return base,adapt
    def mixed(ss,base,adapt,a):return [b if s['condition']==0 else a*q+(1-a)*b for s,b,q in zip(ss,base,adapt)]
    progress('calibration');cal=[sample(e) for e in read(OUT/'task2-round11-phase-validation/calibrate-phase-index.json')['samples']];base,adapt=components(cal);candidates=[dict(alpha_history=a,thresholds=bt,proxy=score(cal,mixed(cal,base,adapt,a),bt)['proxy_IoU']) for a in [0.,.5,1.]];chosen=max(candidates,key=lambda r:r['proxy']);hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.txt')};save('frozen-calibration.json',dict(selected=chosen,candidates=candidates,hashes=hashes,route_sha256=hashlib.sha256((ROOT/'routes.json').read_bytes()).hexdigest(),baseline_frozen_sha256=hashlib.sha256((OUT/'task2-round13-seed-stability/frozen-calibration.json').read_bytes()).hexdigest(),repair_source_sha256=hashlib.sha256((OUT/'task2-round14-history-repair/history_repair.py').read_bytes()).hexdigest(),onset_frozen=True,all_thresholds_frozen=True,February_previously_used_report_only=True,Task1_frozen=True))
    progress('evaluating');reports={};records=[]
    for population,ss in [('new_phase_population',[sample(e) for e in read(OUT/'task2-round11-phase-validation/audit-phase-index.json')['samples']]),('old410',role_samples('audit'))]:
        base,adapt=components(ss);final=mixed(ss,base,adapt,chosen['alpha_history']);reports[population+'_round13']=score(ss,base,bt);reports[population+'_selected']=score(ss,final,bt)
        for s,b,q in zip(ss,base,final):
            if s['condition']==0:assert np.array_equal(b,q)
            if s['has_future_queue']:records.append(dict(population=population,panel=s['panel'],condition=s['condition'],origin=s['origin'],date=s['origin'][:10],base_iou=window_iou(s['y'],b>=bt[s['condition']],s['m']),selected_iou=window_iou(s['y'],q>=bt[s['condition']],s['m'])))
        assert abs(reports[population+'_round13']['proxy_IoU']-read(OUT/'task2-round13-seed-stability/proxy-results.json')['reports'][population+'_selected']['proxy_IoU'])<1e-12
    save('paired-window-metrics.json',records);save('proxy-results.json',dict(official_S_queue=None,reports=reports,selected=chosen,new_ongoing_roads=sum(r['alpha_new']>0 for r in routes.values()),seconds=time.time()-started,onset_predictions_identical=True,all_thresholds_identical=True,different_populations_not_directly_comparable=True,Task1_historical_S_state=.942828633,Task1_score_newly_measured=False))
    progress('predicting_official');checks={}
    for split in ['validation','private']:
        frames=[]
        for p in PANELS:
            d=DATA/'task2'/p/split;index=pd.read_csv(d/'window_index.csv');template=pd.read_csv(d/'sample_submission_queue.csv');parts=[]
            for row in index.itertuples():
                s=released_sample(DATA,p,split,row.window_id);bb,aa=components([s]);q=mixed([s],bb,aa,chosen['alpha_history'])[0];keys=template[template.window_id==row.window_id].copy();ii={l:i for i,l in enumerate(s['network']['link_ids'])};h=((pd.to_datetime(keys.timestamp,utc=True)-pd.Timestamp(s['origin'])).dt.total_seconds()/300).to_numpy(dtype=int)-1;keys['queue_pred']=(q[h,keys.link_id.map(ii).to_numpy()]>=bt[s['condition']]).astype(int);parts.append(keys)
            frame=pd.concat(parts,ignore_index=True);checks[split+'/'+p]=validate_queue(frame,template);frame['panel']=p;frames.append(frame)
        frame=pd.concat(frames,ignore_index=True);assert len(frame)==87000;frame.to_csv(ROOT/f'queue-{split}.csv',index=False)
    assert guard('after')==before;assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items());save('final-verification.json',dict(passed=True,rows=174000,checks=checks,frozen_hashes_unchanged=True,onset_predictions_identical=True,all_thresholds_identical=True,Task1_package_unchanged=True,official_S_queue=None));progress('complete',new_phase_proxy=reports['new_phase_population_selected']['proxy_IoU'],old410_proxy=reports['old410_selected']['proxy_IoU'],seconds=time.time()-started)
if __name__=='__main__':
    import msvcrt
    lock=open(ROOT/'run.lock','a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise SystemExit('Already running')
    try:main()
    except Exception:save('failure.json',dict(traceback=traceback.format_exc()));progress('failed');raise
