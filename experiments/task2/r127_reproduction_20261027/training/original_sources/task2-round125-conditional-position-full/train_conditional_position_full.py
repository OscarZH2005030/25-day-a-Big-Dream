
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
from task2q.data import make_sample,released_sample
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

def head_features(s,bases,mode):
    x=features(s);z=extras(s,bases[s['panel']],mode);out=np.column_stack([x,z]).astype('float32');assert out.shape==(len(x),916) and np.array_equal(out[:,:557],x);return out

def extend(mode,bases):
    folder=WORK/mode;folder.mkdir(parents=True,exist_ok=False);source=BASE/mode;old={k:np.load(source/f'{k}.npy',mmap_mode='r') for k in ['X','y','w','panel']};paired=read(AUDIT/'paired-targets.json')
    assert mode=='full';rows=read(AUDIT/'original-conditional-targets.json')+[r for r in paired if r['period']=='early']+[r for r in paired if r['period']=='held']
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
        path=ROOT/(m+'-first-full.txt');cp=ROOT/(m+'-first-checkpoint.txt')
        def checkpoint(env):
            if (env.iteration+1)%100==0:env.model.save_model(str(cp));progress('conditional_position_training',model=m,step=env.iteration+1,total=600)
        params=dict(objective='multiclass',num_class=6,num_leaves=31,learning_rate=.035,min_data_in_leaf=80,feature_fraction=.85,lambda_l2=4,num_threads=4,verbosity=-1,seed=20261027,deterministic=True,force_col_wise=True);ds=lgb.Dataset(data[m],label=data['y'],weight=data['w']/data['w'].mean());model=lgb.train(params,ds,num_boost_round=600,callbacks=[checkpoint]);model.save_model(str(path));assert model.num_feature()==916 and model.current_iteration()==600 and model.num_trees()==3600;models[m]=model;audits.append(dict(mode=m,rows=len(data['y']),features=916,rounds=600,trees=3600,sha256=sha(path)));del ds;gc.collect()
    save('model-audit.json',audits);return models

def compose(s,parent,anchor,old_prob,new_prob):
    q=concentrated_adjust(s,parent,anchor,new_prob,1.);keep=(old_prob.max(1)<.5)|(~unseen_queue(s));q[:,keep]=parent[:,keep];assert np.array_equal(q.max(0),parent.max(0)) and np.array_equal(q[:,keep],parent[:,keep]);return q

def parent_preflight(parent):
    expected=pd.read_csv(PARENT/'queue-validation.csv');checks=[]
    for p in PANELS:
        d=DATA/'task2'/p/'validation';row=pd.read_csv(d/'window_index.csv').iloc[0];s=released_sample(DATA,p,'validation',row.window_id);pred=parent.predict(s);keys=pd.read_csv(d/'sample_submission_queue.csv');keys=keys[keys.window_id==row.window_id].copy();match=keys.merge(expected[expected.panel==p][['window_id','timestamp','link_id','queue_pred']],on=['window_id','timestamp','link_id'],how='left',suffixes=('_template','_expected'));assert len(match)==len(keys) and match.queue_pred_expected.notna().all();ii={l:i for i,l in enumerate(s['network']['link_ids'])};h=((pd.to_datetime(match.timestamp,utc=True)-pd.Timestamp(s['origin'])).dt.total_seconds()/300).to_numpy(dtype=int)-1;actual=pred[h,match.link_id.map(ii).to_numpy()];assert np.array_equal(actual,match.queue_pred_expected.to_numpy());checks.append(dict(panel=p,rows=len(keys),passed=True))
    save('R116-public-input-reproduction.json',dict(passed=True,split='validation current masked public inputs only, no future labels or February evaluation',checks=checks));return checks


EARLY=OUT/'task2-round124-conditional-position-probe';WORK=PROJECT/'work/round125-conditional-position-full'

def progress(phase,**kw):
    d=dict(phase=phase,pid=os.getpid(),utc=pd.Timestamp.now(tz='UTC').isoformat(),**kw);save('progress.json',d);atomic_json(OUT/'overnight-status.json',dict(d,training=phase not in ['conditional_position_full_complete','failed'],script=str(Path(__file__)),current_accepted_parent='R116'));print(json.dumps(d),flush=True)

def actual_future(models,bases,parent,oldhead,roads):
    es=read(OUT/'task2-round11-phase-validation/tune-phase-index.json')['samples'];checks=[]
    def outputs(s):
        b=parent.probability(s)
        if s['panel'] not in roads or s['condition']==1:return [b,b,b]
        anchor=parent.base.anchor.probability(s);assert np.array_equal(b.max(0),anchor.max(0));old=oldhead.predict(features(s),num_threads=4).astype('float32');return [b]+[compose(s,b,anchor,old,models[m].predict(head_features(s,bases,m),num_threads=4).astype('float32')) for m in ['position','random']]
    for p in PANELS:
        for c in [0,1]:
            e=next(e for e in es if e['panel']==p and e['condition']==c);s=sample(e);assert s['origin'][:7]=='2030-12';a=copy.deepcopy(arrays(e['day_key']));slot=e['origin_slot']
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
            checks.append(dict(panel=p,condition=c,passed=True,ACTUALR116_parent_rawR31anchor_bothheads_mainfuture_rampTfuture_invariant=True,knownqueue_Ongoing_peak_frozen=True,oneCDF_from_originalanchor_only=True))
    return checks

def main():
    start=time.time();before=guard('before');assert read(EARLY/'early-verification.json')['passed'];gate=read(EARLY/'training-gates.json');roads=[p for p in PANELS if gate['roads'][p]];assert roads==['D7_I210_E'];assert read(OUT/'paired-timing-candidate-index.json')['root']==str(PARENT)
    save('experiment.json',dict(stage='FULL shared conditionalhead600 position916 andmatchedrandom916, originalR40full14681positivepairedrows, onlyI210E allowed for eventualroute bycompleteearlyallgates.',data='ALL originalR40 FULL X557/y/w/panel retained byteexact; order=ALLR39original +pairedearly +pairedheld. No filteringsharedtrainingdata byDecemberorroadacceptance. Appendfixed359position/randomfeatures, no otherchange.',models='Freshmulticlass6class600rounds/3600trees each, total7200newtrees CPU4serial/checkpoint100. Same31/.035/minleaf80/.85/L2=4/seed, no557 warmstart.',parent='ActualCURRENT R116 PositionPredictor, anchor parent.base.anchor=originalR31. OnlyapplyoneCDF from rawanchor, not currentalreadycorrectedcurve. BotholdR40fullconditionalmode andnewmode >=.5; oldlowconcentration/knownqueue/Ongoing frozen; peak exactlysame.',December='Solealpha1, onlyearlypassed+>=4positivewindows eachphase; meancurrentR116+.004/matchedcontrol+.002/worstboth-.015. No rescueorrepeatedcoefsearch.',later='Janonlyglobal0/.5/1/negativeguard thenfreeze beforebothpreviouslyusedFebreports/fulltemplate174000/independent87000sourcecacheIOblocked/preservation beforeparentupdate.',budget='7200newtrees max, no newfit afterUTC15:00 orSTOP',Task1='92frozen historicalSstate notremeasured',Ongoing='frozen',official_S_queue=None));hashes={str(EARLY/'training-gates.json'):sha(EARLY/'training-gates.json')};bases={}
    for p in PANELS:
        path=OUT/'task2-round108-position-mode-preflight'/(p+'-basis.npz');hashes[str(path)]=sha(path)
        with np.load(path) as f:bases[p]={k:f[k] for k in ['position','random','supported']}
    for n in ['frozen-calibration.json','routes.json','queue-validation.csv','queue-private.csv','position_runtime.py','position_mode_core.py','timing_core.py','gated_core.py']:hashes[str(PARENT/n)]=sha(PARENT/n)
    oldpath=OUT/'task2-round40-paired-timing/conditional-first-full.txt';hashes[str(oldpath)]=sha(oldpath);oldhead=lgb.Booster(model_file=str(oldpath));parent=PositionPredictor(PARENT);parent_preflight(parent);data,meta=extend('full',bases);models=fit(data);del data;gc.collect();checks=actual_future(models,bases,parent,oldhead,roads);assert guard('after')==before and all(sha(p)==h for p,h in hashes.items()) and all(sha(p)==h for p,h in meta['source_hashes'].items());save('source-hashes.json',hashes);save('full-verification.json',dict(passed=True,qualified_roads=roads,model_hashes={m:sha(ROOT/(m+'-first-full.txt')) for m in ['position','random']},basis_hashes={p:sha(OUT/'task2-round108-position-mode-preflight'/(p+'-basis.npz')) for p in PANELS},causality_groups=checks,parent_root=str(PARENT),parent_frozen_sha256=sha(PARENT/'frozen-calibration.json'),actual_R116_publicinputs_reproduced=True,Task1_92files_unchanged=True,no_December_scores_generated=True,no_bestupdate=True,official_S_queue=None));progress('conditional_position_full_complete',qualified_roads=roads,seconds=time.time()-start)

if __name__=='__main__':
    try:main()
    except Exception:save('failure.json',dict(traceback=traceback.format_exc()));progress('failed');raise
