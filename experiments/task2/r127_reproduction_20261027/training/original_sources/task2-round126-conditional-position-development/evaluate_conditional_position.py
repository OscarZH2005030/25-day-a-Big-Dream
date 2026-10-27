
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

def compose(s,parent,anchor,old_prob,new_prob):
    q=concentrated_adjust(s,parent,anchor,new_prob,1.);keep=(old_prob.max(1)<.5)|(~unseen_queue(s));q[:,keep]=parent[:,keep];assert np.array_equal(q.max(0),parent.max(0)) and np.array_equal(q[:,keep],parent[:,keep]);return q

def parent_preflight(parent):
    expected=pd.read_csv(PARENT/'queue-validation.csv');checks=[]
    for p in PANELS:
        d=DATA/'task2'/p/'validation';row=pd.read_csv(d/'window_index.csv').iloc[0];s=released_sample(DATA,p,'validation',row.window_id);pred=parent.predict(s);keys=pd.read_csv(d/'sample_submission_queue.csv');keys=keys[keys.window_id==row.window_id].copy();match=keys.merge(expected[expected.panel==p][['window_id','timestamp','link_id','queue_pred']],on=['window_id','timestamp','link_id'],how='left',suffixes=('_template','_expected'));assert len(match)==len(keys) and match.queue_pred_expected.notna().all();ii={l:i for i,l in enumerate(s['network']['link_ids'])};h=((pd.to_datetime(match.timestamp,utc=True)-pd.Timestamp(s['origin'])).dt.total_seconds()/300).to_numpy(dtype=int)-1;actual=pred[h,match.link_id.map(ii).to_numpy()];assert np.array_equal(actual,match.queue_pred_expected.to_numpy());checks.append(dict(panel=p,rows=len(keys),passed=True))
    save('R116-public-input-reproduction.json',dict(passed=True,split='validation current masked public inputs only, no future labels or February evaluation',checks=checks));return checks

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


FULL=OUT/'task2-round125-conditional-position-full'

def progress(phase,**kw):
    d=dict(phase=phase,pid=os.getpid(),utc=pd.Timestamp.now(tz='UTC').isoformat(),**kw);save('progress.json',d);atomic_json(OUT/'overnight-status.json',dict(d,training=False,script=str(Path(__file__)),current_accepted_parent='R116'));print(json.dumps(d),flush=True)

def metric(ss,ps):return float(np.mean([window_iou(s['y'],p>=.5,s['m']) for s,p in zip(ss,ps)]))

def main():
    start=time.time();before=guard('before');fv=read(FULL/'full-verification.json');assert fv['passed'] and read(FULL/'progress.json')['phase']=='conditional_position_full_complete';roads=fv['qualified_roads'];assert roads==['D7_I210_E'];assert fv['parent_root']==str(PARENT) and fv['parent_frozen_sha256']==sha(PARENT/'frozen-calibration.json') and read(OUT/'paired-timing-candidate-index.json')['root']==str(PARENT)
    save('development-protocol.json',dict(stage='DecemberONLY actualCURRENT R116 parent, conditional firsttime916position vs916random sharedheads600. OnlyI210E all earlygatespassed.',policy='Solealpha1 ONCECDF from originalR31 anchor=parent.base.anchor, never correctedcurrentcurve. OldR40fullmode>=.5 andnewmode>=.5 gate, sourceknownqueue/Ongoing fixed, all6peakunchanged. No concentration/coefficient/threshold search.',December='>=4positivewindows EACHphase; meanIoU currentR116+.004/matchedrandomcontrol+.002; eachphasevsboth>=-.015. Fixedsourcefallbackotherwise. No otherroad/controlchoice.',later='Onlyifpass newJan-global0/.5/1 atfixed.5/.25+negativefalseany+.03/eligibleFP+.005; freeze BEFOREbothalready-usedFeb reports/full174000/independent87000sourcecacheIOblocked/preservation. NeverFebrouting/tuning/rollback, no blindscoreclaim.',Task1='92frozen historicalSstate notremeasured',Ongoing='frozen',official_S_queue=None))
    hashes={str(FULL/'full-verification.json'):sha(FULL/'full-verification.json')};models={};bases={};data=read(FULL/'full-data-audit.json');cache=PROJECT/'work/round125-conditional-position-full/full'
    for k,h in data['data_hashes'].items():assert sha(cache/f'{k}.npy')==h;hashes[str(cache/f'{k}.npy')]=h
    for m in ['position','random']:
        path=FULL/(m+'-first-full.txt');assert sha(path)==fv['model_hashes'][m];hashes[str(path)]=sha(path);models[m]=lgb.Booster(model_file=str(path));assert models[m].num_feature()==916 and models[m].num_trees()==3600
    for p in PANELS:
        path=OUT/'task2-round108-position-mode-preflight'/(p+'-basis.npz');assert sha(path)==fv['basis_hashes'][p];hashes[str(path)]=sha(path)
        with np.load(path) as f:bases[p]={k:f[k] for k in ['position','random','supported']}
    oldpath=OUT/'task2-round40-paired-timing/conditional-first-full.txt';hashes[str(oldpath)]=sha(oldpath);oldhead=lgb.Booster(model_file=str(oldpath))
    for n in ['frozen-calibration.json','routes.json','queue-validation.csv','queue-private.csv','position_runtime.py','position_mode_core.py','timing_core.py','gated_core.py']:hashes[str(PARENT/n)]=sha(PARENT/n)
    parent=PositionPredictor(PARENT);parent_preflight(parent);es=read(OUT/'task2-round11-phase-validation/tune-phase-index.json')['samples'];reports=[];routes=[]
    for p in PANELS:
        ss=[sample(e) for e in es if e['panel']==p and e['condition']==0];assert all(s['origin'][:7]=='2030-12' and s['has_future_queue'] for s in ss);phase=np.array([(pd.Timestamp(s['origin']).hour*12+pd.Timestamp(s['origin']).minute//5)%3 for s in ss]);counts=[int((phase==k).sum()) for k in range(3)];bb=[parent.probability(s) for s in ss];bv=metric(ss,bb);eligible=p in roads and min(counts)>=4;row=dict(panel=p,windows=len(ss),phase_counts=counts,full_and_phase_eligible=eligible,parent_IoU=bv,use_new=False)
        if eligible:
            pp={m:[] for m in ['position','random']}
            for s,b in zip(ss,bb):
                anchor=parent.base.anchor.probability(s);assert np.array_equal(b.max(0),anchor.max(0));old=oldhead.predict(features(s),num_threads=4).astype('float32')
                for m in pp:pp[m].append(compose(s,b,anchor,old,models[m].predict(head_features(s,bases,m),num_threads=4).astype('float32')))
            nv=metric(ss,pp['position']);cv=metric(ss,pp['random']);gains=[]
            for k in range(3):
                ids=np.flatnonzero(phase==k);ms=[ss[i] for i in ids];v=metric(ms,[pp['position'][i] for i in ids]);gains.append(dict(vs_parent=v-metric(ms,[bb[i] for i in ids]),vs_control=v-metric(ms,[pp['random'][i] for i in ids])))
            passed=bool(nv-bv>=.004 and nv-cv>=.002 and all(v>=-.015 for g in gains for v in g.values()));row.update(new_IoU=nv,matched_IoU=cv,gain_parent=nv-bv,gain_matched=nv-cv,phase_gains=gains,use_new=passed)
        reports.append(row);routes.append(dict(panel=p,use_new=bool(row['use_new']),alpha_new=1. if row['use_new'] else 0.));save('december-selection.json',reports);save('routes.json',routes);progress('conditional_position_December_road',panel=p,use_new=bool(row['use_new']))
    checks=actual_future(models,bases,parent,oldhead,roads);assert guard('after')==before and all(sha(p)==h for p,h in hashes.items());save('source-hashes.json',hashes);save('development-verification.json',dict(passed=True,parent_root=str(PARENT),parent_frozen_sha256=sha(PARENT/'frozen-calibration.json'),causality_groups=checks,R116_publicinputs_reproduced=True,Task1_92files_unchanged=True,knownqueue_Ongoing_peak_frozen=True,originalR31anchor_onceCDF_proved=True,January_February_not_evaluated=True,no_bestupdate=True,official_S_queue=None));n=sum(r['use_new'] for r in routes);progress('conditional_position_December_complete_pending_January' if n else 'conditional_position_December_gate_failed_no_calibration',selected_roads=n,seconds=time.time()-start)

if __name__=='__main__':
    try:main()
    except Exception:save('failure.json',dict(traceback=traceback.format_exc()));progress('failed');raise
