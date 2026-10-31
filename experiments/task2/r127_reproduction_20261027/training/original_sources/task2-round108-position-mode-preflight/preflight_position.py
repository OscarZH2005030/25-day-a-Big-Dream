
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
TASK1=Path(r'C:\Users\18049\Documents\Codex\2026-10-24\trafficflowbench-186-global-road-adaptation\outputs\TrafficFlowBench-Best-20261026')
PARENT=OUT/'task2-round41-paired-policy';EARLY=OUT/'task2-round98-stagewise-event-offset'
WORK=PROJECT/'work/round99-stagewise-event-full';PANEL='D12_I5_S'



def read(p):return json.loads(Path(p).read_text(encoding='utf8'))

def save(n,v):atomic_json(ROOT/n,v)

def guard(stage):
    m=read(TASK1/'package-manifest.json');h={n:hashlib.sha256((TASK1/n).read_bytes()).hexdigest() for n in m['files']};assert all(h[n]==v['sha256'] for n,v in m['files'].items());save('task1-guard-'+stage+'.json',dict(passed=True,files=len(h),hashes=h));return h

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def entries():return read(OUT/'task2-round10-phase-coverage/expanded-fit.json')['samples']

def progress(phase,**kw):
    d=dict(phase=phase,pid=os.getpid(),utc=pd.Timestamp.now(tz='UTC').isoformat(),**kw);save('progress.json',d);atomic_json(OUT/'overnight-status.json',dict(d,training=False,script=str(Path(__file__)),**{}));print(json.dumps(d),flush=True)


def main():
    start=time.time();before=guard('before');prior=read(OUT/'task2-round107-graph-mode-preflight/progress.json');assert prior['phase']=='graph_mode_preflight_complete' and not prior['qualified_roads']
    save('experiment.json',dict(hypothesis='R107 public witnessed-path graph largestcomponent only0.4%-12.8%, all structuralgates failed before any model fit. Do NOT fabricate connections or lower thatgate. Test a different information representation: global spatialposition field using explicit public LWR order_index, which exists for eachobserved mainline link.',distinct='Coordinate projection, NOT a graph, road adjacency, inferred travel distance or changed R107 threshold. Fixed Legendre degree0..7 on provided order coordinates. OriginalR6alreadyhas ownposition; NEW feature is allcorridor past field located in this coordinate, not a reused ownposition scalar.',basis='N>=16,>=.8N explicit matched publicpositions,>=16unique positions, condition<=1e8. Orthonormal position8 vsfixedseed random8 sharing sameconstant, same data/solver359features andoriginal555 preserved=914.',features='13pastframes*3 public channels, observedHQ-only ridge1e-3 projection; pernode modecontributions312 +39pastobservedfractions +8staticbasis. No futurequality/coverage/label inputs; original masks unchanged.',planned_R109='Fresh nativeBCE200 candidate914/control914, same originalR10 JuneSep555/y/w/sampling, perroad. Onlysameinputdimension projectiongeometry differs. Fixed normalize6rawcurve to CURRENTcausalparentpeak; unsupportedpositions/knownqueue/Ongoing fixed. No coef/threshold search. Full600 only earlyallgatespassed then actualR41Dec guards/Jan global/frozenFeb/acceptance.',gates='UNCHANGED parentIoU+.004/control+.002, eachOctNov>=-.01both, negativefalseany+.03/FP+.005, peaks/eventCE/AP retained. No R107model exists or heldscore to rescue.',stage='Zero fitted forecastmodels/scores. Positionbasis+inputcausality preflight only; actualmodel proof pendingR109.',Task1='92files frozen',Ongoing='frozen',official_S_queue=None));summaries=[];checks=[];hashes={}
    for p in PANELS:
        e=next(e for e in entries() if e['panel']==p);s=sample(e);ids=list(s['network']['link_ids']);path=DATA/'corridors'/p/'network/lwr_mainline_topology.csv';hashes[str(path)]=sha(path);df=pd.read_csv(path);assert not df.mainline_link_id.astype(str).duplicated().any();lookup=dict(zip(df.mainline_link_id.astype(str),pd.to_numeric(df.order_index)));known=np.array([i in lookup and np.isfinite(lookup[i]) for i in ids]);order=np.array([lookup.get(i,0.) for i in ids]);basis,meta=make_basis(order,known,20261027+PANELS.index(p));np.savez_compressed(ROOT/(p+'-basis.npz'),**basis,public_order=order);meta.update(panel=p,link_ids=ids,source_order_sha256=hashes[str(path)],basis_sha256=sha(ROOT/(p+'-basis.npz')));save(p+'-basis.json',meta);summaries.append(meta)
        for c in [0,1]:
            e=next(e for e in entries() if e['panel']==p and e['condition']==c and e['origin'][:10]>='2030-10-01');s=sample(e);a=copy.deepcopy(arrays(e['day_key']));slot=e['origin_slot']
            for k in ['speed','flow','occupancy','quality','origin_speed','origin_flow','origin_occupancy','origin_quality']:a[k][slot+1:]=999
            a['eligible'][slot+1:]=False;s2=make_sample(a,s['network'],slot,p)
            for mode in ['position','random']:assert np.array_equal(feature_matrix(s,basis,mode),feature_matrix(s2,basis,mode))
            import ramp_features as rf
            day=rf.day;date=s['origin'][:10]
            def perturb(panel,split,dt):
                z=day(panel,split,dt).copy()
                if dt>=date:z[slot if dt==date else 0:]=999999
                return z
            original=[feature_matrix(s,basis,m) for m in ['position','random']]
            try:rf.day=perturb;assert all(np.array_equal(x,feature_matrix(s,basis,m)) for x,m in zip(original,['position','random']))
            finally:rf.day=day
            checks.append(dict(panel=p,condition=c,passed=True,mainfuture_rampTfuture_input_invariance=True,actual_model_proof_pending=True))
        toy=copy.deepcopy(s);n=len(ids);rng=np.random.default_rng(20261027);toy['x']=toy['x'].copy();toy['x'][:,:,0:3]=rng.uniform(.1,1,(13,n,3));toy['x'][:,:,3]=1;toy['x'][:,:,4:7]=0;toy['x'][:,:,7]=1;swapped=copy.deepcopy(toy);swapped['x']=toy['x'][:,rng.permutation(n),:];assert np.allclose(np.sort(toy['x'],axis=1),np.sort(swapped['x'],axis=1))
        if meta['training_gate']:assert not np.allclose(extras(toy,basis,'position'),extras(swapped,basis,'position'))
        progress('position_mode_basis_complete',panel=p,training_gate=meta['training_gate'])
    assert guard('after')==before and all(sha(p)==h for p,h in hashes.items());save('source-hashes.json',hashes);save('basis-summary.json',summaries);save('preflight-verification.json',dict(passed=True,checks=checks,Task1_92files_unchanged=True,original555_preserved=True,toy_position_sensitive_beyond_global_marginals=True,all_public_coordinate_hashes_unchanged=True,no_fitted_forecast_models_or_scores=True,actual_model_future_proof_pending=True,official_S_queue=None));progress('position_mode_preflight_complete',qualified_roads=[r['panel'] for r in summaries if r['training_gate']],seconds=time.time()-start)

if __name__=='__main__':
    try:main()
    except Exception:save('failure.json',dict(traceback=traceback.format_exc()));progress('failed');raise
