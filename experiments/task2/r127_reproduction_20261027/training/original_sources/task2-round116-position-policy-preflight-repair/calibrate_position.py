
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
from coverage_data import matrix
from position_mode_core import make_basis,extras,feature_matrix
from position_runtime import PositionPredictor,reshape_to_parent
from paired_runtime import normalized_mix
from run_round2 import role_samples,score
from task2q.data import released_sample
from task2q.submission import validate_queue
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

def parent_preflight(parent):
    expected=pd.read_csv(PARENT/'queue-validation.csv');checks=[]
    for p in PANELS:
        d=DATA/'task2'/p/'validation';row=pd.read_csv(d/'window_index.csv').iloc[0];s=released_sample(DATA,p,'validation',row.window_id);pred=parent.predict(s);keys=pd.read_csv(d/'sample_submission_queue.csv');keys=keys[keys.window_id==row.window_id].copy();match=keys.merge(expected[expected.panel==p][['window_id','timestamp','link_id','queue_pred']],on=['window_id','timestamp','link_id'],how='left',suffixes=('_template','_expected'));assert len(match)==len(keys) and match.queue_pred_expected.notna().all();ii={l:i for i,l in enumerate(s['network']['link_ids'])};h=((pd.to_datetime(match.timestamp,utc=True)-pd.Timestamp(s['origin'])).dt.total_seconds()/300).to_numpy(dtype=int)-1;actual=pred[h,match.link_id.map(ii).to_numpy()];assert np.array_equal(actual,match.queue_pred_expected.to_numpy());checks.append(dict(panel=p,rows=len(keys),passed=True))
    save('R41-public-input-reproduction.json',dict(passed=True,split='validation current masked public inputs only, no future labels or February evaluation',checks=checks));return checks


FULL=OUT/'task2-round113-position-full-recovery';DEV=OUT/'task2-round114-position-recovered-development';AUDIT=OUT/'task2-round108-position-mode-preflight'

def progress(phase,**kw):
    d=dict(phase=phase,pid=os.getpid(),utc=pd.Timestamp.now(tz='UTC').isoformat(),**kw);save('progress.json',d);atomic_json(OUT/'overnight-status.json',dict(d,training=False,script=str(Path(__file__))));print(json.dumps(d),flush=True)

def negative(ss,ps):
    groups={}
    for s,p in zip(ss,ps):
        fp=(p>=.5)&s['m'];groups.setdefault(s['panel'],[]).append([float(fp.any()),float(fp.sum()/s['m'].sum())])
    return dict(windows=len(ss),false_any_macro=float(np.mean([np.mean(v,axis=0)[0] for v in groups.values()])),false_eligible_fraction_macro=float(np.mean([np.mean(v,axis=0)[1] for v in groups.values()])))

def actual_future(runtime,parent):
    es=read(OUT/'task2-round11-phase-validation/tune-phase-index.json')['samples'];checks=[]
    for p in PANELS:
        for c in [0,1]:
            e=next(e for e in es if e['panel']==p and e['condition']==c);s=sample(e);a=copy.deepcopy(arrays(e['day_key']));slot=e['origin_slot']
            for k in ['speed','flow','occupancy','quality','origin_speed','origin_flow','origin_occupancy','origin_quality']:a[k][slot+1:]=999
            a['eligible'][slot+1:]=False;s2=make_sample(a,s['network'],slot,p);b=parent.probability(s);q=runtime.probability(s);assert np.array_equal(q,runtime.probability(s2)) and np.array_equal(b.max(0),q.max(0)) and np.array_equal(q[:,~unseen_queue(s)],b[:,~unseen_queue(s)])
            if c==1:assert np.array_equal(q,b)
            import ramp_features as rf
            day=rf.day;date=s['origin'][:10]
            def perturb(panel,split,dt):
                z=day(panel,split,dt).copy()
                if dt>=date:z[slot if dt==date else 0:]=999999
                return z
            try:rf.day=perturb;assert np.array_equal(q,runtime.probability(s))
            finally:rf.day=day
            checks.append(dict(panel=p,condition=c,passed=True,actual_frozen_runtime_mainfuture_rampTfuture_invariant=True,allpeak_knownqueue_Ongoing_frozen=True))
    return checks

def main():
    start=time.time();before=guard('before');assert read(DEV/'development-verification.json')['passed'] and read(DEV/'progress.json')['phase']=='position_mode_December_complete_pending_January';routes=read(DEV/'routes.json');roads=[r['panel'] for r in routes if r['use_new']];assert roads==['D7_I10_E','D12_I5_N'];save('routes.json',routes)
    save('development-protocol.json',dict(stage='January ONLY globalalpha0/.5/1, fixed thresholds.5/.25, no perroad tuning; Dec accepted I10E/I5N routes immutable.',negativeguard='Januaryphase0 negatives falseanymacro<=parent+.03, eligibleFPmacro<=parent+.005, sameR41 calibration views. alpha0eligible; maxJanuaryproxy amongeligible, stabletieparent.',freeze='Save selectedalpha/model/basis/runtime/hash/routes BEFORE accessing two previouslyusedFebruary populations. No February selection/rollback.',later='Ifselectednonzero, descriptive pairedreport bothFebpopulations +actualfullruntimefuture/92/sourcehash +174000templates; onlyafter independent87000source/cacheIO-blocked andportablepreservation may parent/bestregistryupdate. No bestupdate here.',Task1='92frozen historicalSstate notremeasured',Ongoing='all probabilities/thresholds frozen',official_S_queue=None))
    fv=read(FULL/'full-verification.json');hashes={str(DEV/'routes.json'):sha(DEV/'routes.json'),str(DEV/'development-verification.json'):sha(DEV/'development-verification.json'),str(FULL/'full-verification.json'):sha(FULL/'full-verification.json')};files=['position_runtime.py','position_mode_core.py','gated_core.py','timing_core.py']
    for p in roads:
        for name,folder in [(p+'-position.txt',FULL),(p+'-basis.npz',AUDIT),(p+'-basis.json',AUDIT)]:
            src=folder/name;hashes[str(src)]=sha(src);shutil.copyfile(src,ROOT/name);assert sha(ROOT/name)==hashes[str(src)];files.append(name)
        assert sha(ROOT/(p+'-position.txt'))==fv['model_hashes'][p+'/position'] and sha(ROOT/(p+'-basis.npz'))==fv['basis_hashes'][p]
    for n in ['frozen-calibration.json','routes.json','queue-validation.csv','queue-private.csv']:hashes[str(PARENT/n)]=sha(PARENT/n)
    parent=PairedPredictor(PARENT);parent_preflight(parent);models={p:lgb.Booster(model_file=str(ROOT/(p+'-position.txt'))) for p in roads};bases={}
    for p in roads:
        with np.load(ROOT/(p+'-basis.npz')) as f:bases[p]={k:f[k] for k in ['position','random','supported']}
    def components(ss):
        bb=[];qq=[]
        for s in ss:
            b=parent.probability(s);bb.append(b);p=s['panel']
            if s['condition']==0 and p in roads:
                raw=models[p].predict(feature_matrix(s,bases[p],'position'),num_threads=4).astype('float32').reshape(6,-1);qq.append(reshape_to_parent(s,b,raw,bases[p]))
            else:qq.append(b)
        return bb,qq
    def mixed(ss,bb,qq,a):return [normalized_mix(s,b,q,a) if s['condition']==0 else b for s,b,q in zip(ss,bb,qq)]
    cal=[sample(e) for e in read(OUT/'task2-round11-phase-validation/calibrate-phase-index.json')['samples']];assert all(s['origin'].startswith('2031-01') for s in cal);bb,qq=components(cal);baseproxy=score(cal,bb,[.5,.25])['proxy_IoU'];assert abs(baseproxy-read(PARENT/'frozen-calibration.json')['selected']['proxy'])<1e-12
    neg=[s for s in role_samples('calibrate') if s['condition']==0 and not s['has_future_queue']];assert all(s['origin'].startswith('2031-01') for s in neg);nb,nq=components(neg);nbase=negative(neg,nb);candidates=[]
    for a in [0.,.5,1.]:
        stats=negative(neg,mixed(neg,nb,nq,a));eligible=bool(a==0 or (stats['false_any_macro']<=nbase['false_any_macro']+.03 and stats['false_eligible_fraction_macro']<=nbase['false_eligible_fraction_macro']+.005));candidates.append(dict(alpha_position=a,thresholds=[.5,.25],proxy=score(cal,mixed(cal,bb,qq,a),[.5,.25])['proxy_IoU'],negative_stats=stats,eligible=eligible))
    chosen=max((c for c in candidates if c['eligible']),key=lambda c:c['proxy']);frozen=dict(selected=chosen,candidates=candidates,hashes={n:sha(ROOT/n) for n in files},route_sha256=sha(ROOT/'routes.json'),parent_frozen_sha256=sha(PARENT/'frozen-calibration.json'),December_route_source_sha256=sha(DEV/'routes.json'),February_previously_used_report_only=True,condition_forecast_frozen=[chosen['alpha_position']==0,True],Task1_frozen=True,all_thresholds_frozen=True,official_S_queue=None);save('frozen-calibration.json',frozen);progress('position_policy_frozen',alpha_position=chosen['alpha_position'],January_proxy=chosen['proxy']);runtime=PositionPredictor(ROOT)
    for s,b,q in zip(cal,bb,mixed(cal,bb,qq,chosen['alpha_position'])):assert np.array_equal(runtime.probability(s),q)
    checks=actual_future(runtime,parent);save('causality-check.json',dict(passed=True,checks=checks))
    if chosen['alpha_position']==0:
        assert guard('after')==before and all(sha(p)==h for p,h in hashes.items());save('source-hashes.json',hashes);save('final-verification.json',dict(passed=True,selected_parent=True,no_February_scores=True,no_best_update=True,Task1_92files_unchanged=True,official_S_queue=None));progress('position_calibration_selected_parent_no_adoption',seconds=time.time()-start);return
    reports={};records=[];reference=read(PARENT/'proxy-results.json')['reports']
    for population,ss in [('new_phase_population',[sample(e) for e in read(OUT/'task2-round11-phase-validation/audit-phase-index.json')['samples']]),('old410',role_samples('audit'))]:
        assert all(s['origin'].startswith('2031-02') for s in ss);base=[parent.probability(s) for s in ss];final=[runtime.probability(s) for s in ss];reports[population+'_parent']=score(ss,base,[.5,.25]);reports[population+'_selected']=score(ss,final,[.5,.25]);assert abs(reports[population+'_parent']['proxy_IoU']-reference[population+'_selected']['proxy_IoU'])<1e-12
        for s,b,q in zip(ss,base,final):
            assert np.array_equal(b.max(0),q.max(0))
            if s['condition']==1:assert np.array_equal(b,q)
            if s['has_future_queue']:records.append(dict(population=population,panel=s['panel'],condition=s['condition'],origin=s['origin'],parent_iou=window_iou(s['y'],b>=[.5,.25][s['condition']],s['m']),selected_iou=window_iou(s['y'],q>=[.5,.25][s['condition']],s['m'])))
        progress('position_February_repeat_report',population=population,proxy=reports[population+'_selected']['proxy_IoU'])
    save('proxy-results.json',dict(official_S_queue=None,reports=reports,selected=chosen,new_groups={'0':len(roads),'1':0},Task1_historical_S_state=.942828633,Task1_score_newly_measured=False,February_repeated_development_not_blind=True,no_February_routing_tuning_or_rollback=True,different_populations_not_comparable=True));save('paired-window-metrics.json',records)
    template_checks={}
    for split in ['validation','private']:
        frames=[]
        for p in PANELS:
            directory=DATA/'task2'/p/split;index=pd.read_csv(directory/'window_index.csv');template=pd.read_csv(directory/'sample_submission_queue.csv');parts=[]
            for row in index.itertuples():
                s=released_sample(DATA,p,split,row.window_id);pred=runtime.predict(s);keys=template[template.window_id==row.window_id].copy();ii={l:i for i,l in enumerate(s['network']['link_ids'])};h=((pd.to_datetime(keys.timestamp,utc=True)-pd.Timestamp(s['origin'])).dt.total_seconds()/300).to_numpy(dtype=int)-1;keys['queue_pred']=pred[h,keys.link_id.map(ii).to_numpy()].astype(int);parts.append(keys)
            frame=pd.concat(parts,ignore_index=True);template_checks[split+'/'+p]=validate_queue(frame,template);frame['panel']=p;frames.append(frame)
        frame=pd.concat(frames,ignore_index=True);assert len(frame)==87000;frame.to_csv(ROOT/f'queue-{split}.csv',index=False);progress('position_template_complete',split=split,rows=len(frame))
    assert guard('after')==before and all(sha(p)==h for p,h in hashes.items()) and all(sha(ROOT/n)==h for n,h in frozen['hashes'].items());save('source-hashes.json',hashes);save('final-verification.json',dict(passed=True,rows=174000,checks=template_checks,Task1_92files_unchanged=True,Ongoing_all6peak_frozen=True,January_frozen_before_February=True,February_report_only_no_rollback=True,portable_independent_87000_and_preservation_pending=True,no_best_update=True,official_S_queue=None));progress('position_policy_complete_pending_independent_preservation',seconds=time.time()-start)

if __name__=='__main__':
    try:main()
    except Exception:save('failure.json',dict(traceback=traceback.format_exc()));progress('failed');raise
