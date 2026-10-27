
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import sys,json,time,hashlib,traceback,copy,gc
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent;PROJECT=ROOT.parents[1]
sys.path.insert(1,str(OUT/'task2-round116-position-policy-preflight-repair'))
from position_runtime import PositionPredictor
from timing_core import features,unseen_queue
from gated_core import gated_adjust,concentrated_adjust
from position_mode_core import extras,feature_matrix
from run_round2 import sample,arrays,PANELS,DATA
from coverage_data import matrix
from task2q.data import make_sample
from task2q.cache import atomic_json
from task2q.metrics import window_iou
TASK1=Path(r'C:\Users\18049\Documents\Codex\2026-10-24\trafficflowbench-186-global-road-adaptation\outputs\TrafficFlowBench-Best-20261026')
PARENT=OUT/'task2-round116-position-policy-preflight-repair';WORK=PROJECT/'work/round124-conditional-position';BASE=PROJECT/'work/round40-paired-timing';AUDIT=OUT/'task2-round39-paired-phase-calendar'

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))

def save(n,v):atomic_json(ROOT/n,v)

def guard(stage):
    m=read(TASK1/'package-manifest.json');h={n:hashlib.sha256((TASK1/n).read_bytes()).hexdigest() for n in m['files']};assert all(h[n]==v['sha256'] for n,v in m['files'].items());save('task1-guard-'+stage+'.json',dict(passed=True,files=len(h),hashes=h));return h

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def entries():return read(OUT/'task2-round10-phase-coverage/expanded-fit.json')['samples']

def negative(ss,ps):return dict(false_any=float(np.mean([((p>=.5)&s['m']).any() for s,p in zip(ss,ps)])),eligible_FP=float(np.mean([((p>=.5)&s['m']).sum()/s['m'].sum() for s,p in zip(ss,ps)])))

def class_report(y,w,old,new):
    w=np.asarray(w,dtype=float);w/=w.sum();idx=np.arange(len(y));a=float(np.dot(w,-np.log(np.clip(old[idx,y],1e-8,1))));b=float(np.dot(w,-np.log(np.clip(new[idx,y],1e-8,1))));aa=float(np.dot(w,old.argmax(1)==y));bb=float(np.dot(w,new.argmax(1)==y));return dict(views=len(y),old_logloss=a,new_logloss=b,logloss_gain=a-b,old_accuracy=aa,new_accuracy=bb,accuracy_gain=bb-aa)


def progress(phase,**kw):
    d=dict(phase=phase,pid=os.getpid(),utc=pd.Timestamp.now(tz='UTC').isoformat(),**kw);save('progress.json',d);atomic_json(OUT/'overnight-status.json',dict(d,training=phase not in ['conditional_position_complete','failed'],script=str(Path(__file__)),current_accepted_parent='R116'));print(json.dumps(d),flush=True)

def metric(ss,ps,t=.5):return float(np.mean([window_iou(s['y'],p>=t,s['m']) for s,p in zip(ss,ps)]))

def head_features(s,bases,mode):
    x=features(s);z=extras(s,bases[s['panel']],mode);out=np.column_stack([x,z]).astype('float32');assert out.shape==(len(x),916) and np.array_equal(out[:,:557],x);return out

def extend(mode,bases):
    folder=WORK/mode;folder.mkdir(parents=True,exist_ok=False);source=BASE/mode;old={k:np.load(source/f'{k}.npy',mmap_mode='r') for k in ['X','y','w','panel']};paired=read(AUDIT/'paired-targets.json')
    rows=([r for r in read(AUDIT/'original-conditional-targets.json') if r['period']=='early']+[r for r in paired if r['period']=='early']) if mode=='early' else [r for r in paired if r['period']=='held']
    assert np.array_equal(old['y'],np.array([r['first_step'] for r in rows])) and np.array_equal(old['panel'],np.array([r['panel'] for r in rows]));groups={}
    for i,r in enumerate(rows):groups.setdefault((r['panel'],r['origin']),[]).append((i,r))
    X={m:np.lib.format.open_memmap(folder/f'{m}.npy',mode='w+',dtype='float32',shape=(len(rows),916)) for m in ['position','random']}
    for i,items in enumerate(groups.values()):
        if (PROJECT/'work/STOP_NIGHT').exists():raise RuntimeError('Paused by user')
        s=sample(items[0][1]);rr=np.array([j for j,r in items]);ids=np.array([r['target_link_index'] for j,r in items]);assert np.array_equal(features(s)[ids],old['X'][rr]) and unseen_queue(s)[ids].all() and s['m'][:,ids].all() and s['y'][:,ids].any(0).all()
        for m in X:X[m][rr]=head_features(s,bases,m)[ids]
        if (i+1)%400==0:progress('conditional_position_data',mode=mode,windows=i+1,total=len(groups))
    for v in X.values():v.flush()
    data={k:np.array(old[k],copy=True) for k in ['y','w','panel']}
    for k,v in data.items():np.save(folder/f'{k}.npy',v)
    meta=dict(passed=True,mode=mode,rows=len(rows),features=916,original557_X_y_w_panel_exact=True,source_hashes={str(source/f'{k}.npy'):sha(source/f'{k}.npy') for k in old},data_hashes={m:sha(folder/f'{m}.npy') for m in ['position','random','y','w','panel']});atomic_json(folder/'complete.json',meta);save(mode+'-data-audit.json',meta);return {**data,**{m:np.load(folder/f'{m}.npy',mmap_mode='r') for m in ['position','random']}},meta

def fit(data):
    models={};audits=[]
    for m in ['position','random']:
        if pd.Timestamp.now(tz='UTC')>=pd.Timestamp('2026-10-27T15:00:00Z') or (PROJECT/'work/STOP_NIGHT').exists():raise RuntimeError('No new fit after cutoff/pause')
        path=ROOT/(m+'-first-probe.txt');cp=ROOT/(m+'-first-checkpoint.txt')
        def checkpoint(env):
            if (env.iteration+1)%100==0:env.model.save_model(str(cp));progress('conditional_position_training',model=m,step=env.iteration+1,total=200)
        params=dict(objective='multiclass',num_class=6,num_leaves=31,learning_rate=.035,min_data_in_leaf=80,feature_fraction=.85,lambda_l2=4,num_threads=4,verbosity=-1,seed=20261027,deterministic=True,force_col_wise=True);ds=lgb.Dataset(data[m],label=data['y'],weight=data['w']/data['w'].mean());model=lgb.train(params,ds,num_boost_round=200,callbacks=[checkpoint]);model.save_model(str(path));assert model.num_feature()==916 and model.current_iteration()==200 and model.num_trees()==1200;models[m]=model;audits.append(dict(mode=m,rows=len(data['y']),features=916,rounds=200,trees=1200,sha256=sha(path)));del ds;gc.collect()
    save('model-audit.json',audits);return models

def compose(s,parent,anchor,old_prob,new_prob):
    q=concentrated_adjust(s,parent,anchor,new_prob,1.);keep=(old_prob.max(1)<.5)|(~unseen_queue(s));q[:,keep]=parent[:,keep];assert np.array_equal(q.max(0),parent.max(0)) and np.array_equal(q[:,keep],parent[:,keep]);return q

def verify(models,bases,head,reference):
    checks=[]
    for p in PANELS:
        source=lgb.Booster(model_file=str(OUT/f'task2-round10-phase-coverage/{p}-0-probe-expanded.txt'))
        def outputs(s):
            anchor=source.predict(matrix(s),num_threads=4).astype('float32').reshape(6,-1);old=head.predict(features(s),num_threads=4).astype('float32');b=gated_adjust(s,anchor,old,1.);raw=reference[p].predict(feature_matrix(s,bases[p],'position'),num_threads=4).astype('float32').reshape(6,-1);peak=b.max(0);use=unseen_queue(s)&(peak>1e-6);parent=b.copy();parent[:,use]=raw[:,use]/np.maximum(raw[:,use].max(0),1e-20)*peak[use];parent[raw[:,use].argmax(0),np.flatnonzero(use)]=peak[use]
            if s['condition']==1:parent=b
            return [parent]+[compose(s,parent,anchor,old,models[m].predict(head_features(s,bases,m),num_threads=4).astype('float32')) for m in ['position','random']]
        for c in [0,1]:
            e=next(e for e in entries() if e['panel']==p and e['condition']==c and e['origin'][:10]>='2030-10-01');s=sample(e);a=copy.deepcopy(arrays(e['day_key']));slot=e['origin_slot']
            for k in ['speed','flow','occupancy','quality','origin_speed','origin_flow','origin_occupancy','origin_quality']:a[k][slot+1:]=999
            a['eligible'][slot+1:]=False;s2=make_sample(a,s['network'],slot,p);b=outputs(s);q=outputs(s2);assert all(np.array_equal(x,y) for x,y in zip(b,q))
            if c==1:assert all(np.array_equal(b[0],z) for z in b[1:])
            import ramp_features as rf
            day=rf.day;date=s['origin'][:10]
            def perturb(panel,split,dt):
                z=day(panel,split,dt).copy()
                if dt>=date:z[slot if dt==date else 0:]=999999
                return z
            try:rf.day=perturb;assert all(np.array_equal(x,y) for x,y in zip(b,outputs(s)))
            finally:rf.day=day
            checks.append(dict(panel=p,condition=c,passed=True,actual_early_parent_bothheads_future_ramp_invariant=True,all6peak_knownqueue_Ongoing_frozen=True,no_full_parent_on_OctNov=True))
    return checks

def main():
    start=time.time();before=guard('before');assert read(OUT/'task2-round123-regional-development/development-verification.json')['passed'];assert read(OUT/'task2-round123-regional-development/progress.json')['selected_roads']==0
    save('experiment.json',dict(hypothesis='R123 regional cellBCE timingfailedDecember while acceptedR116 showspositioninformationuseful. Learn first-arrival directly CONDITIONAL on an event from pairedpositive examples, rather than calibrating unconditionalcellprobabilities dominated by negatives. Representation/target factorization changes, no rejectedregional/coefficientparameter rescue.',distinct='R53 addedgraph-neighbor snapshots to conditional head; NEW uses validated globalPUBLICposition field fromR116, sameold paired557/y/w exactly, matchedsame916 randomprojection control. Not changingLegendredegree, bins, alpha, loss orK.',data='R40early9970 paired/original conditionalpositive rows, original557/y/w/panel byteexact. Append359causalpositionfeatures=916; no label/group/mask/futurequality inputs. Complete labels used for historical target selection only.',models='Shared6class200position916 vsrandom916, same31/.035/minleaf80/.85/L2=4/seed/CPU4serial/checkpoint100; 2400newtrees. Freshfull600onlyifallgatespass.',classification='RetainR53 gates vsR40: original>=20/paired>=30 targetviews; both logloss/accuracy gains>=-.02, at leastone population BOTH LL+.02 andaccuracy+.03. ADD noninferior to matchedrandomcontrol LL>=0 andaccuracy>=0 on BOTH; no relaxinggates.',timing='Solealpha1, require originalR40 mode>=.5 ANDnew mode>=.5 asR53; use singleCDF from rawR10anchor, never re-CDF alreadycorrectedparent. Reference parentALL8R110JuneSeppositioncurve (noDecroute selection). Keep sixpeak/Ongoing/knownqueue. MeanIoU parentR110+.004 ANDbaseR10R40+.004 ANDmatchedrandomhead+.002; eachOctNov all3>=-.01; negativefalseany<=all3+.03/eligibleFP<=all3+.005.',later='Fullheads trainJuneNov R40samepaireddata, actualCURRENT R116 parent & originalR31anchor (parent.base.anchor) onceCDF, R40fullprob controls oldconcentration; originalDec/Jan/frozenFeb+independentpreserve required. No fullR116 orR41 OctNov predictions.',budget='2*200*6=2400earlytrees; cutUTC15:00/userSTOP before eachnewfit',Task1='frozen92 historicalSstate notremeasured',Ongoing='frozen',official_S_queue=None));bases={};references={};hashes={str(PARENT/'frozen-calibration.json'):sha(PARENT/'frozen-calibration.json')}
    for p in PANELS:
        path=OUT/'task2-round108-position-mode-preflight'/(p+'-basis.npz');hashes[str(path)]=sha(path)
        with np.load(path) as f:bases[p]={k:f[k] for k in ['position','random','supported']}
        path=OUT/f'task2-round110-position-mode-logging-repair/{p}-position.txt';hashes[str(path)]=sha(path);references[p]=lgb.Booster(model_file=str(path))
    train,tm=extend('early',bases);models=fit(train);del train;gc.collect();pairs,pm=extend('paired-held',bases);oldpath=OUT/'task2-round40-paired-timing/conditional-first-probe.txt';hashes[str(oldpath)]=sha(oldpath);head=lgb.Booster(model_file=str(oldpath));pair_reports={}
    for p in PANELS:
        use=pairs['panel']==p;y=pairs['y'][use].astype(int);w=pairs['w'][use];x=pairs['position'][use];pred=models['position'].predict(x,num_threads=4);control=models['random'].predict(pairs['random'][use],num_threads=4);pair_reports[p]=dict(source=class_report(y,w,head.predict(x[:,:557],num_threads=4),pred),control=class_report(y,w,control,pred))
    del pairs;gc.collect();reports=[]
    for p in PANELS:
        source=lgb.Booster(model_file=str(OUT/f'task2-round10-phase-coverage/{p}-0-probe-expanded.txt'));ss=[sample(e) for e in sorted(entries(),key=lambda e:e['origin']) if e['panel']==p and e['condition']==0 and e['origin'][:10]>='2030-10-01'];curves={m:[] for m in ['base','parent','position','random']};Y=[];W=[];oldp=[];npred=[];cpred=[]
        for s in ss:
            anchor=source.predict(matrix(s),num_threads=4).astype('float32').reshape(6,-1);old=head.predict(features(s),num_threads=4).astype('float32');base=gated_adjust(s,anchor,old,1.);raw=references[p].predict(feature_matrix(s,bases[p],'position'),num_threads=4).astype('float32').reshape(6,-1);peak=base.max(0);use=unseen_queue(s)&(peak>1e-6);parent=base.copy();parent[:,use]=raw[:,use]/np.maximum(raw[:,use].max(0),1e-20)*peak[use];parent[raw[:,use].argmax(0),np.flatnonzero(use)]=peak[use];curves['base'].append(base);curves['parent'].append(parent);probs={}
            for m in ['position','random']:probs[m]=models[m].predict(head_features(s,bases,m),num_threads=4).astype('float32');curves[m].append(compose(s,parent,anchor,old,probs[m]))
            known=unseen_queue(s)&s['m'].all(0)&s['y'].any(0)
            if known.any():Y.extend(np.argmax(s['y'][:,known],axis=0).tolist());W.extend([1/int(known.sum())]*int(known.sum()));oldp.extend(old[known]);npred.extend(probs['position'][known]);cpred.extend(probs['random'][known])
        y=np.asarray(Y,dtype=int);original=dict(source=class_report(y,W,np.asarray(oldp),np.asarray(npred)),control=class_report(y,W,np.asarray(cpred),np.asarray(npred)));paired=pair_reports[p];pop=[original['source'],paired['source']];controls=[original['control'],paired['control']];classok=bool(pop[0]['views']>=20 and pop[1]['views']>=30 and all(r['logloss_gain']>=-.02 and r['accuracy_gain']>=-.02 for r in pop) and any(r['logloss_gain']>=.02 and r['accuracy_gain']>=.03 for r in pop) and all(r['logloss_gain']>=0 and r['accuracy_gain']>=0 for r in controls));pos=[i for i,s in enumerate(ss) if s['has_future_queue']];neg=[i for i,s in enumerate(ss) if not s['has_future_queue']];scores={}
        for m,ps in curves.items():
            monthly={}
            for month in ['2030-10','2030-11']:
                ids=[i for i in pos if ss[i]['origin'].startswith(month)];monthly[month]=metric([ss[i] for i in ids],[ps[i] for i in ids])
            scores[m]=dict(IoU=metric([ss[i] for i in pos],[ps[i] for i in pos]),monthly=monthly,negative=negative([ss[i] for i in neg],[ps[i] for i in neg]))
        n=scores['position'];refs=[scores[k] for k in ['base','parent','random']];taskok=bool(n['IoU']-scores['base']['IoU']>=.004 and n['IoU']-scores['parent']['IoU']>=.004 and n['IoU']-scores['random']['IoU']>=.002 and all(n['monthly'][m]-r['monthly'][m]>=-.01 for m in ['2030-10','2030-11'] for r in refs) and all(n['negative']['false_any']<=r['negative']['false_any']+.03 and n['negative']['eligible_FP']<=r['negative']['eligible_FP']+.005 for r in refs));reports.append(dict(panel=p,original=original,paired=paired,classification_gate=classok,scores=scores,task_gate=taskok,passed=classok and taskok));save('temporal-holdout.json',reports);progress('conditional_position_held_road',panel=p,passed=classok and taskok)
    checks=verify(models,bases,head,references);assert guard('after')==before and all(sha(p)==h for p,h in hashes.items());assert all(sha(p)==h for meta in [tm,pm] for p,h in meta['source_hashes'].items());save('source-hashes.json',hashes);save('training-gates.json',dict(roads={r['panel']:r['passed'] for r in reports},passed_roads=sum(r['passed'] for r in reports),full_started=False));save('early-verification.json',dict(passed=True,causality_groups=checks,originalR40_557_y_w_preserved=True,Task1_92files_unchanged=True,current_R116_unmodified=True,no_fullparent_on_OctNov=True,no_December_January_February_scores=True,no_bestupdate=True,official_S_queue=None));progress('conditional_position_complete',passed_roads=sum(r['passed'] for r in reports),seconds=time.time()-start)

if __name__=='__main__':
    try:main()
    except Exception:save('failure.json',dict(traceback=traceback.format_exc()));progress('failed');raise
