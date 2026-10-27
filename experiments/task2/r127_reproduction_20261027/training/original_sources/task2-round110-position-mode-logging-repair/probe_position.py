
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import sys,json,time,hashlib,traceback,copy,gc
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent;PROJECT=ROOT.parents[1]
sys.path.insert(1,str(OUT/'task2-round41-paired-policy'))
from paired_runtime import PairedPredictor
from timing_core import features,unseen_queue
from run_round2 import sample,arrays,PANELS,DATA
from task2q.data import make_sample,released_sample
from task2q.cache import atomic_json
from task2q.metrics import window_iou
from offset_core import baseline_logit,residual_derivatives,corrected_peak
from coverage_data import matrix
from position_mode_core import make_basis,extras,feature_matrix
from special_features import latest_state
from timing_core import unseen_queue
from gated_core import gated_adjust
TASK1=Path(r'C:\Users\18049\Documents\Codex\2026-10-24\trafficflowbench-186-global-road-adaptation\outputs\TrafficFlowBench-Best-20261026')
PARENT=OUT/'task2-round41-paired-policy';EARLY=OUT/'task2-round98-stagewise-event-offset'
WORK=PROJECT/'work/round99-stagewise-event-full';PANEL='D12_I5_S'




def read(p):return json.loads(Path(p).read_text(encoding='utf8'))

def save(n,v):atomic_json(ROOT/n,v)

def guard(stage):
    m=read(TASK1/'package-manifest.json');h={n:hashlib.sha256((TASK1/n).read_bytes()).hexdigest() for n in m['files']};assert all(h[n]==v['sha256'] for n,v in m['files'].items());save('task1-guard-'+stage+'.json',dict(passed=True,files=len(h),hashes=h));return h

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def entries():return read(OUT/'task2-round10-phase-coverage/expanded-fit.json')['samples']

def metric(ss,ps):return float(np.mean([window_iou(s['y'],p>=.5,s['m']) for s,p in zip(ss,ps)]))

def negative(ss,ps):return dict(false_any=float(np.mean([((p>=.5)&s['m']).any() for s,p in zip(ss,ps)])),eligible_FP=float(np.mean([((p>=.5)&s['m']).sum()/s['m'].sum() for s,p in zip(ss,ps)])))

def binary_report(y,w,old,new):
    y=np.asarray(y);w=np.asarray(w);w/=w.sum()
    def ce(q):
        q=np.clip(q,1e-8,1-1e-8);return float(np.dot(w,-y*np.log(q)-(1-y)*np.log1p(-q)))
    a=ce(old);b=ce(new);ap0=float(average_precision_score(y,old,sample_weight=w));ap1=float(average_precision_score(y,new,sample_weight=w));return dict(views=len(y),positive_views=int(y.sum()),old_CE=a,new_CE=b,CE_gain=a-b,old_AP=ap0,new_AP=ap1,AP_gain=ap1-ap0)


AUDIT=OUT/'task2-round108-position-mode-preflight';WORK=PROJECT/'work/round110-position-mode-early';SEED=20261027

def progress(phase,**kw):
    d=dict(phase=phase,pid=os.getpid(),utc=pd.Timestamp.now(tz='UTC').isoformat(),**kw);save('progress.json',d);atomic_json(OUT/'overnight-status.json',dict(d,training=phase not in ['position_mode_probe_complete','failed'],script=str(Path(__file__))));print(json.dumps(d),flush=True)

def check_start():
    if pd.Timestamp.now(tz='UTC')>=pd.Timestamp('2026-10-27T15:00:00Z') or (PROJECT/'work/STOP_NIGHT').exists():raise RuntimeError('No new fitting after deadline or humanpause')

def build(panel,basis):
    check_start();folder=WORK/panel;folder.mkdir(parents=True,exist_ok=False);base=PROJECT/f'work/round10-phase/{panel}-0';meta=read(base/'complete.json');old={k:np.load(base/f'{k}.npy',mmap_mode='r') for k in ['X','y','w','early','window_positive']};mask=np.asarray(old['early'],dtype=bool);source_ids=np.flatnonzero(mask);n=len(source_ids);E={name:np.lib.format.open_memmap(folder/f'{name}.npy',mode='w+',dtype='float32',shape=(n,359)) for name in ['position','random']};lookup={(e['origin'],e['source']):e for e in entries() if e['panel']==panel and e['condition']==0};rngs={name:np.random.default_rng(SEED+off+PANELS.index(panel)*2) for name,off in [('original',0),('added_phase',2000)]};cursor=0;windows=0
    for j,win in enumerate(meta['windows']):
        e=lookup[(win['origin'],win['source'])];s=sample(e);m=s['m'].ravel();y=s['y'].ravel();last,queued=latest_state(s);keep=np.flatnonzero(m&(y>0));negative=np.flatnonzero(m&~(y>0));risk=np.tile((last<.9)|(s['x'][-2,:,1]>.7)|(s['x'][-2,:,2]>.2),6);a=negative[risk[negative]];b=negative[~risk[negative]];ka=min(96,len(a));kb=min(192-ka,len(b));ka=min(192-kb,len(a));rng=rngs[win['source']];aa=rng.choice(a,ka,replace=False) if ka else np.array([],int);bb=rng.choice(b,kb,replace=False) if kb else np.array([],int);ids=np.concatenate([keep,aa,bb]);start,end=win['start'],win['end'];assert len(ids)==end-start and np.array_equal(y[ids],old['y'][start:end])
        if bool(win['early']):
            assert e['origin'][:10]<'2030-10-01';xx=matrix(s);assert np.array_equal(xx[ids],old['X'][start:end]);a0,z0=map(int,np.searchsorted(source_ids,[start,end]));assert a0==cursor and z0-a0==len(ids);nn=s['x'].shape[1]
            for mode in E:E[mode][a0:z0]=extras(s,basis,mode)[ids%nn]
            cursor=z0;windows+=1
        if (j+1)%200==0:
            for v in E.values():v.flush()
            progress('position_mode_replay',panel=panel,windows=j+1,total=len(meta['windows']),early_rows=cursor)
    assert cursor==n;counts={v:sum(bool(e['positive'])==v and e['early'] for e in meta['windows']) for v in [False,True]};flags=np.asarray(old['window_positive'][mask],dtype=bool);w=np.asarray(old['w'][mask],dtype=float)*np.where(flags,meta['class_counts']['True']/counts[True],meta['class_counts']['False']/counts[False]);assert abs(w[flags].sum()-.8)<1e-6 and abs(w[~flags].sum()-.2)<1e-6
    for v in E.values():v.flush()
    np.save(folder/'y.npy',old['y'][mask]);np.save(folder/'w.npy',w);np.save(folder/'source_ids.npy',source_ids);audit=dict(passed=True,panel=panel,rows=n,early_windows=windows,source_features=555,newfeatures=359,total_features=914,all_original_selected_cells_and_labels_replayed=True,weights_only_original_early_prefix_normalizer=True,positive_window_mass=float(w[flags].sum()),negative_window_mass=float(w[~flags].sum()),source_hashes={str(base/f'{k}.npy'):sha(base/f'{k}.npy') for k in ['X','y','w','early','window_positive']},new_data_hashes={k:sha(folder/f'{k}.npy') for k in ['position','random','y','w','source_ids']});atomic_json(folder/'complete.json',audit);save(panel+'-data-audit.json',audit);return folder,old,source_ids,audit

def fit(panel,basis):
    folder,old,ids,audit=build(panel,basis);y=np.load(folder/'y.npy');w=np.load(folder/'w.npy');w/=w.mean();models={};reports=[]
    for mode in ['position','random']:
        check_start();X=np.empty((len(y),914),dtype='float32');X[:,:555]=old['X'][ids];X[:,555:]=np.load(folder/f'{mode}.npy',mmap_mode='r');path=ROOT/f'{panel}-{mode}.txt';cp=ROOT/f'{panel}-{mode}-checkpoint.txt'
        def checkpoint(env):
            if (env.iteration+1)%100==0:env.model.save_model(str(cp));progress('position_mode_training',panel=panel,model=mode,step=env.iteration+1,total=200)
        params=dict(objective='binary',num_leaves=31,learning_rate=.035,min_data_in_leaf=80,feature_fraction=.85,lambda_l2=4,num_threads=4,verbosity=-1,seed=SEED,deterministic=True,force_col_wise=True);ds=lgb.Dataset(X,label=y,weight=w);model=lgb.train(params,ds,num_boost_round=200,callbacks=[checkpoint]);model.save_model(str(path));assert model.num_feature()==914 and model.num_trees()==200;models[mode]=model;reports.append(dict(mode=mode,features=914,rounds=200,rows=len(y),sha256=sha(path),matched_original555_y_weights_and_capacity=True));del X,ds;gc.collect()
    assert all(sha(p)==h for p,h in audit['source_hashes'].items()) and all(sha(folder/f'{k}.npy')==h for k,h in audit['new_data_hashes'].items());save(panel+'-model-audit.json',reports);return models

def adapt(s,b,raw,basis):
    if s['condition']!=0:return b.copy()
    peak=b.max(0);use=unseen_queue(s)&(peak>1e-6)&basis['supported'];v=raw[:,use];rp=v.max(0);shape=v/np.maximum(rp,1e-20);out=b.copy();out[:,use]=shape*peak[use];out[v.argmax(0),np.flatnonzero(use)]=peak[use];assert np.array_equal(out.max(0),peak) and np.array_equal(out[:,~use],b[:,~use]) and np.isfinite(out).all();return out

def verify(models,bases,head):
    checks=[]
    for p in PANELS:
        source=lgb.Booster(model_file=str(OUT/f'task2-round10-phase-coverage/{p}-0-probe-expanded.txt'))
        def outputs(s):
            b=gated_adjust(s,source.predict(matrix(s),num_threads=4).astype('float32').reshape(6,-1),head.predict(features(s),num_threads=4).astype('float32'),1.)
            if p not in models:return [b]
            return [b]+[adapt(s,b,models[p][mode].predict(feature_matrix(s,bases[p],mode),num_threads=4).astype('float32').reshape(6,-1),bases[p]) for mode in ['position','random']]
        for c in [0,1]:
            e=next(e for e in entries() if e['panel']==p and e['condition']==c and e['origin'][:10]>='2030-10-01');s=sample(e);a=copy.deepcopy(arrays(e['day_key']));slot=e['origin_slot']
            for k in ['speed','flow','occupancy','quality','origin_speed','origin_flow','origin_occupancy','origin_quality']:a[k][slot+1:]=999
            a['eligible'][slot+1:]=False;s2=make_sample(a,s['network'],slot,p);b=outputs(s);q=outputs(s2);assert all(np.array_equal(x,y) for x,y in zip(b,q))
            for z in b[1:]:
                assert np.array_equal(z.max(0),b[0].max(0)) and np.array_equal(z[:,~unseen_queue(s)],b[0][:,~unseen_queue(s)])
                if c==1:assert np.array_equal(z,b[0])
            import ramp_features as rf
            day=rf.day;date=s['origin'][:10]
            def perturb(panel,split,dt):
                z=day(panel,split,dt).copy()
                if dt>=date:z[slot if dt==date else 0:]=999999
                return z
            try:rf.day=perturb;assert all(np.array_equal(x,y) for x,y in zip(b,outputs(s)))
            finally:rf.day=day
            checks.append(dict(panel=p,condition=c,passed=True,actual_parent_both_model_mainfuture_rampTfuture_invariant=True,Ongoing_knownqueue_all6peak_frozen=True))
    return checks

def main():
    start=time.time();before=guard('before');assert read(AUDIT/'preflight-verification.json')['passed'];pre=read(AUDIT/'progress.json');assert pre['phase']=='position_mode_preflight_complete';roads=pre['qualified_roads'];assert roads
    save('experiment.json',dict(hypothesis='Location-sensitive public corridor-state field provides information absent from globalmoments while sparsegraph preflight prohibits inventing connections. Use fixed publicposition polynomialmodes vs matched randomcompressions.',source='R108 positionbasis preflight passed original555/causalfeatures, no heldscores. R107graphstructuralallfailed, no oldmodel to tweak. R106 dynamicanalogues failed, no rescue.',data='EXACT originalR10 selectedcells555/y/earlyIPW replays; add359<=T observed-only spatialprojectionfeatures, 914total. Original allpositive/192sampledneg cells with .8/.2 windowmass and earlyprefix-only classnormalizers unchanged.',models='Each structurallyqualified road fresh nativeBCE200 position914 vs random914, same31/.035/minleaf80/.85/L2=4/seed20261027/4threads. No teacher or Task1 input. Checkpoint100, source/newdata SHA.',policy='Solealpha1 normalized newrawsixcurve to CURRENT R10JuneSep200+R40JuneSep200 parentpeak, no neweventgate/classificationpeak. ownunseen/position-supported/sourcepeak>1e-6 only; knownqueue/Ongoing/unsupported same. No coefficient/threshold search or repeated CDF.',control='Same914dimension, constantmode, orthonormalrank8, data/params/capacity. Randomgeometry only differs from providedorder polynomialgeometry; control never adopted.',gates='UNCHANGED positivewindowIoU parent+.004/control+.002; eachOctNov>=-.01both; negativefalseany<=both+.03/eligibleFP<=both+.005; eventCE/AP same bitwisepeaks. Allroads decided solelystructuraltrainingpreflight.',later='Onlyallgatespassed full600 withsamefeatures/rules thenACTUALR41Dec>=4phase parent+.004/control+.002/worstboth-.015; Janonlyglobal0/.5/1; freeze beforebothpreviouslyviewedFeb reports/fullacceptance/independentpreserve. No bestupdate here.',budget=f'{len(roads)}roads*2*200={len(roads)*400}newtrees CPU4serial cutoffUTC15:00/userSTOP',Task1='frozen92 historicalSstate notremeasured',Ongoing='frozen',official_S_queue=None))
    hashes={str(AUDIT/'preflight-verification.json'):sha(AUDIT/'preflight-verification.json')};bases={}
    for p in roads:
        path=AUDIT/(p+'-basis.npz');meta=read(AUDIT/(p+'-basis.json'));assert sha(path)==meta['basis_sha256'];hashes[str(path)]=sha(path)
        with np.load(path) as f:bases[p]={k:f[k] for k in ['position','random','supported']}
    headpath=OUT/'task2-round40-paired-timing/conditional-first-probe.txt';hashes[str(headpath)]=sha(headpath);head=lgb.Booster(model_file=str(headpath));models={};reports=[]
    for p in roads:
        models[p]=fit(p,bases[p]);path=OUT/f'task2-round10-phase-coverage/{p}-0-probe-expanded.txt';hashes[str(path)]=sha(path);source=lgb.Booster(model_file=str(path));ss=[sample(e) for e in sorted(entries(),key=lambda e:e['origin']) if e['panel']==p and e['condition']==0 and e['origin'][:10]>='2030-10-01'];curves={n:[] for n in ['parent','position','random']};peak_same=True
        for s in ss:
            b=gated_adjust(s,source.predict(matrix(s),num_threads=4).astype('float32').reshape(6,-1),head.predict(features(s),num_threads=4).astype('float32'),1.);curves['parent'].append(b)
            for mode in ['position','random']:
                raw=models[p][mode].predict(feature_matrix(s,bases[p],mode),num_threads=4).astype('float32').reshape(6,-1);z=adapt(s,b,raw,bases[p]);curves[mode].append(z);assert np.array_equal(z.max(0),b.max(0))
        pos=[i for i,s in enumerate(ss) if s['has_future_queue']];neg=[i for i,s in enumerate(ss) if not s['has_future_queue']];scores={}
        for mode,ps in curves.items():
            months={}
            for month in ['2030-10','2030-11']:
                ids=[i for i in pos if ss[i]['origin'].startswith(month)];months[month]=metric([ss[i] for i in ids],[ps[i] for i in ids])
            scores[mode]=dict(positive_window_IoU=metric([ss[i] for i in pos],[ps[i] for i in pos]),monthly_IoU=months,equal_eval_month_mean_IoU=float(np.mean(list(months.values()))),negative=negative([ss[i] for i in neg],[ps[i] for i in neg]))
        new=scores['position'];base=scores['parent'];control=scores['random'];gains={m:dict(vs_parent=new['monthly_IoU'][m]-base['monthly_IoU'][m],vs_control=new['monthly_IoU'][m]-control['monthly_IoU'][m]) for m in ['2030-10','2030-11']};passed=bool(new['positive_window_IoU']-base['positive_window_IoU']>=.004 and new['positive_window_IoU']-control['positive_window_IoU']>=.002 and all(v>=-.01 for g in gains.values() for v in g.values()) and all(new['negative']['false_any']<=r['negative']['false_any']+.03 and new['negative']['eligible_FP']<=r['negative']['eligible_FP']+.005 for r in [base,control]));reports.append(dict(panel=p,classification=dict(event_peak_identical=True,CE_gain=0.,AP_gain=0.),scores=scores,monthly_gains=gains,passed=passed));save('temporal-holdout.json',reports);progress('position_mode_held_road',panel=p,passed=passed,completed=len(reports),total=len(roads));gc.collect()
    checks=verify(models,bases,head);assert guard('after')==before and all(sha(p)==h for p,h in hashes.items());save('source-hashes.json',hashes);save('training-gates.json',dict(roads={p:any(r['panel']==p and r['passed'] for r in reports) for p in PANELS},passed_roads=sum(r['passed'] for r in reports),full_started=False));save('early-verification.json',dict(passed=True,causality_groups=checks,Task1_92files_unchanged=True,knownqueue_Ongoing_all6peak_frozen=True,source_hashes_unchanged=True,no_December_January_February_scores=True,no_best_update=True,official_S_queue=None));progress('position_mode_probe_complete',passed_roads=sum(r['passed'] for r in reports),seconds=time.time()-start)

if __name__=='__main__':
    try:main()
    except Exception:save('failure.json',dict(traceback=traceback.format_exc()));progress('failed');raise
