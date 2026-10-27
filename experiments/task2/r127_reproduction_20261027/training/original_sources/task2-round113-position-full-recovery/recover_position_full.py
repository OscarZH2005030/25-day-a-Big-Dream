
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import sys,json,time,hashlib,traceback,copy,gc,shutil
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

def check_start():
    if pd.Timestamp.now(tz='UTC')>=pd.Timestamp('2026-10-27T15:00:00Z') or (PROJECT/'work/STOP_NIGHT').exists():raise RuntimeError('No new fitting after deadline or humanpause')

def build(panel,basis):
    check_start();folder=WORK/panel;folder.mkdir(parents=True,exist_ok=False);base=PROJECT/f'work/round10-phase/{panel}-0';meta=read(base/'complete.json');old={k:np.load(base/f'{k}.npy',mmap_mode='r') for k in ['X','y','w','early','window_positive']};mask=np.ones(len(old['y']),dtype=bool);source_ids=np.flatnonzero(mask);n=len(source_ids);E={name:np.lib.format.open_memmap(folder/f'{name}.npy',mode='w+',dtype='float32',shape=(n,359)) for name in ['position','random']};lookup={(e['origin'],e['source']):e for e in entries() if e['panel']==panel and e['condition']==0};rngs={name:np.random.default_rng(SEED+off+PANELS.index(panel)*2) for name,off in [('original',0),('added_phase',2000)]};cursor=0;windows=0
    for j,win in enumerate(meta['windows']):
        e=lookup[(win['origin'],win['source'])];s=sample(e);m=s['m'].ravel();y=s['y'].ravel();last,queued=latest_state(s);keep=np.flatnonzero(m&(y>0));negative=np.flatnonzero(m&~(y>0));risk=np.tile((last<.9)|(s['x'][-2,:,1]>.7)|(s['x'][-2,:,2]>.2),6);a=negative[risk[negative]];b=negative[~risk[negative]];ka=min(96,len(a));kb=min(192-ka,len(b));ka=min(192-kb,len(a));rng=rngs[win['source']];aa=rng.choice(a,ka,replace=False) if ka else np.array([],int);bb=rng.choice(b,kb,replace=False) if kb else np.array([],int);ids=np.concatenate([keep,aa,bb]);start,end=win['start'],win['end'];assert len(ids)==end-start and np.array_equal(y[ids],old['y'][start:end])
        if True:
            assert '2030-06-01'<=e['origin'][:10]<'2030-12-01';xx=matrix(s);assert np.array_equal(xx[ids],old['X'][start:end]);a0,z0=map(int,np.searchsorted(source_ids,[start,end]));assert a0==cursor and z0-a0==len(ids);nn=s['x'].shape[1]
            for mode in E:E[mode][a0:z0]=extras(s,basis,mode)[ids%nn]
            cursor=z0;windows+=1
        if (j+1)%200==0:
            for v in E.values():v.flush()
            progress('position_mode_full_replay',panel=panel,windows=j+1,total=len(meta['windows']),full_rows=cursor)
    assert cursor==n;counts={v:sum(bool(e['positive'])==v for e in meta['windows']) for v in [False,True]};flags=np.asarray(old['window_positive'][mask],dtype=bool);w=np.asarray(old['w'][mask],dtype=float)*np.where(flags,meta['class_counts']['True']/counts[True],meta['class_counts']['False']/counts[False]);assert abs(w[flags].sum()-.8)<1e-6 and abs(w[~flags].sum()-.2)<1e-6
    for v in E.values():v.flush()
    np.save(folder/'y.npy',old['y'][mask]);np.save(folder/'w.npy',w);np.save(folder/'source_ids.npy',source_ids);audit=dict(passed=True,panel=panel,rows=n,full_windows=windows,source_features=555,newfeatures=359,total_features=914,all_original_selected_cells_and_labels_replayed=True,weights_same_original_full_prefix_normalizer=True,positive_window_mass=float(w[flags].sum()),negative_window_mass=float(w[~flags].sum()),source_hashes={str(base/f'{k}.npy'):sha(base/f'{k}.npy') for k in ['X','y','w','early','window_positive']},new_data_hashes={k:sha(folder/f'{k}.npy') for k in ['position','random','y','w','source_ids']});atomic_json(folder/'complete.json',audit);save(panel+'-data-audit.json',audit);return folder,old,source_ids,audit


REUSE=OUT/'task2-round111-position-mode-full';OLDWORK=PROJECT/'work/round111-position-mode-full';REUSED_HASHES={}
_build_new=build

def build(panel,basis):
    folder=OLDWORK/panel;path=folder/'complete.json'
    if not path.exists():return _build_new(panel,basis)
    audit=read(path);assert audit['passed'];REUSED_HASHES[str(path)]=sha(path)
    for p,h in audit['source_hashes'].items():assert sha(p)==h;REUSED_HASHES[p]=h
    for k,h in audit['new_data_hashes'].items():
        p=folder/f'{k}.npy';assert sha(p)==h;REUSED_HASHES[str(p)]=h
    base=PROJECT/f'work/round10-phase/{panel}-0';old={k:np.load(base/f'{k}.npy',mmap_mode='r') for k in ['X','y','w','early','window_positive']};ids=np.load(folder/'source_ids.npy');save(panel+'-data-audit.json',dict(audit,reused_readonly_cache=str(folder),reused_cache_metadata_sha256=sha(path)));return folder,old,ids,audit

def fit(panel,basis):
    folder,old,ids,audit=build(panel,basis);y=np.load(folder/'y.npy');w=np.load(folder/'w.npy');w/=w.mean();models={};reports=[]
    for mode in ['position','random']:
        path=ROOT/f'{panel}-{mode}.txt';cp=ROOT/f'{panel}-{mode}-checkpoint.txt';original=REUSE/f'{panel}-{mode}.txt';sourcecp=REUSE/f'{panel}-{mode}-checkpoint.txt';record=dict(mode=mode,features=914,rounds=600,rows=len(y),matched_original555_y_weights_and_capacity=True)
        if original.exists():
            sourceaudit=read(REUSE/(panel+'-model-audit.json'));expected=next(a['sha256'] for a in sourceaudit if a['mode']==mode);assert sha(original)==expected;shutil.copyfile(original,path);REUSED_HASHES[str(original)]=expected;model=lgb.Booster(model_file=str(path));record.update(reused_completed_model=True,source_sha256=expected,new_trees=0)
        else:
            check_start();X=np.empty((len(y),914),dtype='float32');X[:,:555]=old['X'][ids];X[:,555:]=np.load(folder/f'{mode}.npy',mmap_mode='r');done=0;initial=None
            if sourcecp.exists():
                REUSED_HASHES[str(sourcecp)]=sha(sourcecp);shutil.copyfile(sourcecp,cp);initial=str(cp);done=lgb.Booster(model_file=initial).current_iteration();assert panel=='D7_I210_W' and mode=='position' and done==300;record.update(resumed_checkpoint=True,source_checkpoint_sha256=sha(sourcecp),checkpoint_rounds=done)
            def checkpoint(env):
                if (env.iteration+1)%100==0:env.model.save_model(str(cp));progress('position_mode_full_training',panel=panel,model=mode,step=env.iteration+1,total=600)
            params=dict(objective='binary',num_leaves=31,learning_rate=.035,min_data_in_leaf=80,feature_fraction=.85,lambda_l2=4,num_threads=4,verbosity=-1,seed=SEED,deterministic=True,force_col_wise=True)
            # Match the interrupted candidate's 300+300 native continuation schedule
            # in its paired random control. No December scores have been inspected.
            stages=([300,300] if panel=='D7_I210_W' and mode=='random' and done==0 else [600-done]);first_done=done
            for rounds in stages:
                check_start();ds=lgb.Dataset(X,label=y,weight=w);model=lgb.train(params,ds,num_boost_round=rounds,init_model=initial,callbacks=[checkpoint]);model.save_model(str(cp));done=model.current_iteration();initial=str(cp);del ds;gc.collect()
            model.save_model(str(path));record.update(new_trees=600-first_done,continuation_schedule_matched=panel=='D7_I210_W',not_claimed_bitwise_equivalent_to_uninterrupted_random_state=panel=='D7_I210_W');del X;gc.collect()
        assert model.num_feature()==914 and model.num_trees()==600 and model.current_iteration()==600;models[mode]=model;record['sha256']=sha(path);reports.append(record);save(panel+'-model-audit.json',reports)
    assert all(sha(p)==h for p,h in audit['source_hashes'].items()) and all(sha(folder/f'{k}.npy')==h for k,h in audit['new_data_hashes'].items());return models

def adapt(s,b,raw,basis):
    if s['condition']!=0:return b.copy()
    peak=b.max(0);use=unseen_queue(s)&(peak>1e-6)&basis['supported'];v=raw[:,use];rp=v.max(0);shape=v/np.maximum(rp,1e-20);out=b.copy();out[:,use]=shape*peak[use];out[v.argmax(0),np.flatnonzero(use)]=peak[use];assert np.array_equal(out.max(0),peak) and np.array_equal(out[:,~use],b[:,~use]) and np.isfinite(out).all();return out

def parent_preflight(parent):
    expected=pd.read_csv(PARENT/'queue-validation.csv');checks=[]
    for p in PANELS:
        d=DATA/'task2'/p/'validation';row=pd.read_csv(d/'window_index.csv').iloc[0];s=released_sample(DATA,p,'validation',row.window_id);pred=parent.predict(s);keys=pd.read_csv(d/'sample_submission_queue.csv');keys=keys[keys.window_id==row.window_id].copy();match=keys.merge(expected[expected.panel==p][['window_id','timestamp','link_id','queue_pred']],on=['window_id','timestamp','link_id'],how='left',suffixes=('_template','_expected'));assert len(match)==len(keys) and match.queue_pred_expected.notna().all();ii={l:i for i,l in enumerate(s['network']['link_ids'])};h=((pd.to_datetime(match.timestamp,utc=True)-pd.Timestamp(s['origin'])).dt.total_seconds()/300).to_numpy(dtype=int)-1;actual=pred[h,match.link_id.map(ii).to_numpy()];assert np.array_equal(actual,match.queue_pred_expected.to_numpy());checks.append(dict(panel=p,rows=len(keys),passed=True))
    save('R41-public-input-reproduction.json',dict(passed=True,split='validation current masked public inputs only, no future labels or February evaluation',checks=checks));return checks


AUDIT=OUT/'task2-round108-position-mode-preflight';EARLY=OUT/'task2-round110-position-mode-logging-repair';WORK=PROJECT/'work/round113-position-mode-full-recovery';SEED=20261027

def progress(phase,**kw):
    d=dict(phase=phase,pid=os.getpid(),utc=pd.Timestamp.now(tz='UTC').isoformat(),**kw);save('progress.json',d);atomic_json(OUT/'overnight-status.json',dict(d,training=phase not in ['position_mode_full_complete','failed'],script=str(Path(__file__))));print(json.dumps(d),flush=True)

def actual_future(models,bases,parent):
    checks=[];es=read(OUT/'task2-round11-phase-validation/tune-phase-index.json')['samples']
    def outputs(s):
        b=parent.probability(s);p=s['panel']
        if p not in models:return [b]
        return [b]+[adapt(s,b,models[p][mode].predict(feature_matrix(s,bases[p],mode),num_threads=4).astype('float32').reshape(6,-1),bases[p]) for mode in ['position','random']]
    for p in PANELS:
        for c in [0,1]:
            e=next(e for e in es if e['panel']==p and e['condition']==c);s=sample(e);assert s['origin'][:7]=='2030-12';a=copy.deepcopy(arrays(e['day_key']));slot=e['origin_slot']
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
            checks.append(dict(panel=p,condition=c,passed=True,actual_R41_and_full_candidate_control_mainfuture_rampTfuture_invariant=True,all6peak_knownqueue_Ongoing_frozen=True))
    return checks

def main():
    start=time.time();before=guard('before');check_start();assert read(EARLY/'early-verification.json')['passed'] and read(EARLY/'progress.json')['phase']=='position_mode_probe_complete';gate=read(EARLY/'training-gates.json');roads=[p for p in PANELS if gate['roads'][p]];assert len(roads)==5
    save('experiment.json',dict(stage='RECOVER interrupted full training: reuse4completed600, resumeI210W300 with matched300+300control, finishremaining5roadfull capacity6000',qualified_roads=roads,source='ImmutableR108 providedposition Legendrebasis; no parameter/feature/memory/route search. Rejected3roads not fitted.',data='R10 FULL555/y/IPW cells exact replay for all1032selectedwindows eachroad; full .8/.2 normalization from originalweights. Same359observed-only profileprojection features andsameposition/randombases.',models='Reuse4completed600 models byte-exact; I210W position300+300 continuation and randomcontrol matched300+300 schedule; remainingI5N/S fresh600. SameR110params/data. Continuation not claimed bitwise identical to uninterrupted RNG; CPU4serial/checkpoint100.',policy='Same solealpha1 normalizedrawsixcurve to actualR41 currentpeak; ownunseen/publicpeak>1e-6/explicitposition-supported only. Allunmasked6peak/knownqueue/Ongoing unchanged.',December_predeclared='No coefficientsearch. Eligible5roads+>=4positivewindows eachphase, meanIoU>=R41+.004 ANDmatchedcontrol+.002, everyphasevsboth>=-.015. No oldFebselection.',later='NextNEWdevelopment actualR41Dec route only, ifpassedJanuaryonlyglobal0/.5/1+negativeguards atfixed.5/.25, thenfreeze beforebothpreviouslyusedFebruary reports/full174000/independent87000/sourcecacheIOblocked/integrity/independentsave beforebestadoption.',acceptance='Thisstage8publicmaskedR41inputreproduction+16actualR41fullmodelmainfuture/rampTfuture/allpeak/knownqueue/Ongoing+original92/sourcehash. NoDecember scoring orbestupdate here.',budget='3300remaining newtrees;2700trees reused from R111 including300checkpoint; totalcapacity6000. No newfit afterUTC15:00/userSTOP',Task1='frozen92 historicalSstate notremeasured',Ongoing='frozen',official_S_queue=None))
    hashes={str(EARLY/'training-gates.json'):sha(EARLY/'training-gates.json'),str(EARLY/'early-verification.json'):sha(EARLY/'early-verification.json')};bases={}
    for p in roads:
        path=AUDIT/(p+'-basis.npz');assert sha(path)==read(AUDIT/(p+'-basis.json'))['basis_sha256'];hashes[str(path)]=sha(path)
        with np.load(path) as f:bases[p]={k:f[k] for k in ['position','random','supported']}
    for n in ['frozen-calibration.json','routes.json','queue-validation.csv','queue-private.csv','paired_runtime.py','conditional-first-full.txt']:hashes[str(PARENT/n)]=sha(PARENT/n)
    parent=PairedPredictor(PARENT);parent_preflight(parent);models={}
    for p in roads:models[p]=fit(p,bases[p]);progress('position_mode_full_road_complete',panel=p,completed=len(models),total=len(roads))
    hashes.update(REUSED_HASHES);checks=actual_future(models,bases,parent);assert guard('after')==before and all(sha(p)==h for p,h in hashes.items());save('source-hashes.json',hashes);save('full-verification.json',dict(passed=True,qualified_roads=roads,model_hashes={p+'/'+mode:sha(ROOT/f'{p}-{mode}.txt') for p in roads for mode in ['position','random']},basis_hashes={p:sha(AUDIT/(p+'-basis.npz')) for p in roads},causality_groups=checks,actual_R41_public_input_reproduced=True,all6peak_knownqueue_Ongoing_frozen=True,Task1_92files_unchanged=True,no_held_evaluation_scores_generated=True,no_best_update=True,official_S_queue=None));progress('position_mode_full_complete',qualified_roads=roads,seconds=time.time()-start)

if __name__=='__main__':
    try:main()
    except Exception:save('failure.json',dict(traceback=traceback.format_exc()));progress('failed');raise

