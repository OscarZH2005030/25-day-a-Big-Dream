
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

def adapt(s,b,raw,basis):
    if s['condition']!=0:return b.copy()
    peak=b.max(0);use=unseen_queue(s)&(peak>1e-6)&basis['supported'];v=raw[:,use];rp=v.max(0);shape=v/np.maximum(rp,1e-20);out=b.copy();out[:,use]=shape*peak[use];out[v.argmax(0),np.flatnonzero(use)]=peak[use];assert np.array_equal(out.max(0),peak) and np.array_equal(out[:,~use],b[:,~use]) and np.isfinite(out).all();return out

def parent_preflight(parent):
    expected=pd.read_csv(PARENT/'queue-validation.csv');checks=[]
    for p in PANELS:
        d=DATA/'task2'/p/'validation';row=pd.read_csv(d/'window_index.csv').iloc[0];s=released_sample(DATA,p,'validation',row.window_id);pred=parent.predict(s);keys=pd.read_csv(d/'sample_submission_queue.csv');keys=keys[keys.window_id==row.window_id].copy();match=keys.merge(expected[expected.panel==p][['window_id','timestamp','link_id','queue_pred']],on=['window_id','timestamp','link_id'],how='left',suffixes=('_template','_expected'));assert len(match)==len(keys) and match.queue_pred_expected.notna().all();ii={l:i for i,l in enumerate(s['network']['link_ids'])};h=((pd.to_datetime(match.timestamp,utc=True)-pd.Timestamp(s['origin'])).dt.total_seconds()/300).to_numpy(dtype=int)-1;actual=pred[h,match.link_id.map(ii).to_numpy()];assert np.array_equal(actual,match.queue_pred_expected.to_numpy());checks.append(dict(panel=p,rows=len(keys),passed=True))
    save('R41-public-input-reproduction.json',dict(passed=True,split='validation current masked public inputs only, no future labels or February evaluation',checks=checks));return checks

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


FULL=OUT/'task2-round113-position-full-recovery';AUDIT=OUT/'task2-round108-position-mode-preflight'

def progress(phase,**kw):
    d=dict(phase=phase,pid=os.getpid(),utc=pd.Timestamp.now(tz='UTC').isoformat(),**kw);save('progress.json',d);atomic_json(OUT/'overnight-status.json',dict(d,training=False,script=str(Path(__file__))));print(json.dumps(d),flush=True)

def metric(ss,ps):return float(np.mean([window_iou(s['y'],p>=.5,s['m']) for s,p in zip(ss,ps)]))

def main():
    start=time.time();before=guard('before');fv=read(FULL/'full-verification.json');assert fv['passed'] and read(FULL/'progress.json')['phase']=='position_mode_full_complete';roads=fv['qualified_roads'];assert len(roads)==5
    save('development-protocol.json',dict(parent='Immutable actualR41 policy, publicmaskedvalidationinput reproduction BEFOREDecember scoring',method='Only5R110allgatespassed roads, R113 recovered position914600 vs matchedrandom914600;4models copied byte-exact, I210W matched300+300 checkpointcontinuation documented before any December scores. R108bases byte-identical, allsameoriginal555/y/IPW/fullprefixnormalizers.',policy='Solealpha1 normalizednative6curve to currentR41 exact6peak; ownunseen/sourcepeak>1e-6/knownpositions only, knownqueue/Ongoing/unsupported unchanged. No decoder/coefficient/threshold search.',December='>=4positivewindows eachphase; candidate meanIoU parent+.004 ANDcontrol+.002; everyphase gainsvsboth>=-.015. Fallbacksourceonly otherwise. No training orotherroad/tuning.',later='Ifanypassed nextNEWJanuaryonlyglobal0/.5/1 plusnegativefalseany+.03/eligibleFP+.005 atfrozen.5/.25; freeze BEFOREbothpreviouslyusedFebreports, no perroadFebselection/rollback. Full174000/independent87000/sourcecacheIOprohibited/integrity/independentsave beforebestadoption.',Task1='original92frozen historicalSstate notremeasured',Ongoing='frozen',official_S_queue=None))
    hashes={str(FULL/'full-verification.json'):sha(FULL/'full-verification.json')};models={};bases={}
    for p in roads:
        basispath=AUDIT/(p+'-basis.npz');assert sha(basispath)==fv['basis_hashes'][p];hashes[str(basispath)]=sha(basispath)
        with np.load(basispath) as f:bases[p]={k:f[k] for k in ['position','random','supported']}
        models[p]={}
        for mode in ['position','random']:
            path=FULL/f'{p}-{mode}.txt';assert sha(path)==fv['model_hashes'][p+'/'+mode];hashes[str(path)]=sha(path);models[p][mode]=lgb.Booster(model_file=str(path));assert models[p][mode].num_feature()==914 and models[p][mode].num_trees()==600
    for n in ['frozen-calibration.json','routes.json','queue-validation.csv','queue-private.csv','paired_runtime.py']:hashes[str(PARENT/n)]=sha(PARENT/n)
    parent=PairedPredictor(PARENT);assert parent.rule['thresholds']==[.5,.25];parent_preflight(parent);es=read(OUT/'task2-round11-phase-validation/tune-phase-index.json')['samples'];reports=[];routes=[]
    for p in PANELS:
        ss=[sample(e) for e in es if e['panel']==p and e['condition']==0];assert all(s['origin'][:7]=='2030-12' and s['has_future_queue'] for s in ss);phase=np.array([(pd.Timestamp(s['origin']).hour*12+pd.Timestamp(s['origin']).minute//5)%3 for s in ss]);counts=[int((phase==k).sum()) for k in range(3)];bb=[parent.probability(s) for s in ss];bv=metric(ss,bb);eligible=p in models and min(counts)>=4;row=dict(panel=p,windows=len(ss),phase_counts=counts,full_and_phase_eligible=eligible,parent_IoU=bv,use_new=False)
        if eligible:
            pp={mode:[adapt(s,b,models[p][mode].predict(feature_matrix(s,bases[p],mode),num_threads=4).astype('float32').reshape(6,-1),bases[p]) for s,b in zip(ss,bb)] for mode in ['position','random']};nv=metric(ss,pp['position']);cv=metric(ss,pp['random']);gains=[]
            for k in range(3):
                ids=np.flatnonzero(phase==k);ms=[ss[i] for i in ids];v=metric(ms,[pp['position'][i] for i in ids]);gains.append(dict(vs_parent=v-metric(ms,[bb[i] for i in ids]),vs_control=v-metric(ms,[pp['random'][i] for i in ids])))
            passed=bool(nv-bv>=.004 and nv-cv>=.002 and all(v>=-.015 for g in gains for v in g.values()));row.update(new_IoU=nv,matched_IoU=cv,gain_parent=nv-bv,gain_matched=nv-cv,phase_gains=gains,use_new=passed);assert all(np.array_equal(b.max(0),q.max(0)) for b,q in zip(bb,pp['position']))
        reports.append(row);routes.append(dict(panel=p,use_new=bool(row['use_new']),alpha_new=1. if row['use_new'] else 0.));save('december-selection.json',reports);save('routes.json',routes);progress('position_mode_December_road',panel=p,use_new=bool(row['use_new']))
    checks=actual_future(models,bases,parent);assert guard('after')==before and all(sha(p)==h for p,h in hashes.items());save('source-hashes.json',hashes);save('development-verification.json',dict(passed=True,causality_groups=checks,Task1_92files_unchanged=True,R41_public_inputs_reproduced=True,all6peak_knownqueue_Ongoing_frozen=True,January_February_not_evaluated=True,no_best_update=True,official_S_queue=None));n=sum(r['use_new'] for r in routes);progress('position_mode_December_complete_pending_January' if n else 'position_mode_December_gate_failed_no_calibration',selected_roads=n,seconds=time.time()-start)

if __name__=='__main__':
    try:main()
    except Exception:save('failure.json',dict(traceback=traceback.format_exc()));progress('failed');raise
