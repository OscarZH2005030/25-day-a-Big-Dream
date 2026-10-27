"""Pre-training gate: observed-history masking, plus December observability audit."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import sys,json,time,traceback,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1];OUT=ROOT.parent
sys.path[1:1]=[str(OUT/n) for n in ['task2-round13-seed-stability','task2-round12-phase-soft','task2-round11-phase-validation','task2-round10-phase-coverage','task2-round9-hierarchical','task2-round8-soft-targets','task2-round7-structured','task2-round6-ramp-physics','task2-round5-distribution','task2-round2']]
from seed_runtime import SeedPredictor
from run_round2 import sample,PANELS
from task2q.cache import atomic_json
from task2q.metrics import window_iou
from history_repair import repair
read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
TASK1=Path(r'C:\Users\18049\Documents\Codex\2026-10-24\trafficflowbench-186-global-road-adaptation\outputs\TrafficFlowBench-Best-20261026')
def save(n,v):atomic_json(ROOT/n,v)
def progress(phase,**kw):
    v=dict(phase=phase,pid=os.getpid(),utc=pd.Timestamp.now(tz='UTC').isoformat(),**kw);save('progress.json',v);print(json.dumps(v),flush=True)
def guard(stage):
    manifest=read(TASK1/'package-manifest.json');h={n:hashlib.sha256((TASK1/n).read_bytes()).hexdigest() for n in manifest['files']};assert all(h[n]==v['sha256'] for n,v in manifest['files'].items());save('task1-guard-'+stage+'.json',dict(passed=True,files=len(h),hashes=h));return h
def main():
    start=time.time();before=guard('before');rng=np.random.default_rng(20261030)
    save('experiment.json',dict(hypothesis='missing history obscures local onset trajectories; bounded-window interpolation may add useful estimates',phase='pre-training diagnostic gate',CPU_threads=4,Task1='frozen historical .942828633; artificial reconstruction diagnostic is NOT Task1 score',training_gate='fit-history mean weighted normalized RMSE improves >=2%; latest3 speed and flow RMSE each not worse than carry by >2%',methods=['past-only carry with free-speed/zero-flow defaults','linear interpolation within released <=forecast-T history, endpoint nearest value'],mask_rates=[.2,.35,.5],no_training_on_December=True,official_S_queue=None))
    entries=read(OUT/'task2-round10-phase-coverage/expanded-fit.json')['samples'];chosen=[]
    for p in PANELS:
        for c in [0,1]:
            group=[e for e in entries if e['panel']==p and e['condition']==c and e['origin'][:10]<'2030-10-01'];ids=rng.choice(len(group),min(24,len(group)),replace=False);chosen.extend(group[i] for i in ids)
    save('reconstruction-index.json',dict(samples=chosen,selection='24 per panel-condition sampled only June-September; naturally balanced training windows, not natural population'))
    accum={};invariance=True
    for count,e in enumerate(chosen):
        s=sample(e);x=s['x'];free=s['network']['free_speed'];lanes=s['network']['static'][:,1]*8;capacity=s['network']['capacity'];n=x.shape[1]
        for rate in [.2,.35,.5]:
            dropped=rng.random((13,n))<rate
            for name,ch,missing,default,scale in [('speed',0,4,1.,free),('flow',1,5,0.,capacity/lanes)]:
                observed=x[:,:,missing]<.5;available=observed&~dropped;target=observed&dropped&(x[:,:,3]>=.75)
                values=x[:,:,ch];perturbed=values.copy();perturbed[~available]=12345
                for method,linear in [('carry',False),('linear',True)]:
                    q=repair(values,available,default,linear);assert np.array_equal(q,repair(perturbed,available,default,linear))
                    error=((q-values)*scale)**2
                    for region,mask in [('all',target),('latest3',target&(np.arange(13)[:,None]>=10))]:
                        key=(s['condition'],rate,name,method,region);a=accum.setdefault(key,[0.,0]);a[0]+=float(error[mask].sum());a[1]+=int(mask.sum())
        if (count+1)%48==0:progress('history_reconstruction',windows=count+1,total=len(chosen))
    records=[dict(condition=c,mask_rate=r,channel=ch,method=m,region=g,RMSE=(a[0]/a[1])**.5,cells=a[1]) for (c,r,ch,m,g),a in accum.items()];lookup={(r['condition'],r['mask_rate'],r['channel'],r['method'],r['region']):r['RMSE'] for r in records};comparisons=[]
    for c in [0,1]:
        for rate in [.2,.35,.5]:
            get=lambda ch,m,g:lookup[(c,rate,ch,m,g)]
            a=.54*get('speed','carry','all')/25+.46*get('flow','carry','all')/600;b=.54*get('speed','linear','all')/25+.46*get('flow','linear','all')/600
            comparisons.append(dict(condition=c,mask_rate=rate,carry_weighted_normalized_RMSE=a,linear_weighted_normalized_RMSE=b,relative_improvement=1-b/a,latest3_speed_ratio=get('speed','linear','latest3')/get('speed','carry','latest3'),latest3_flow_ratio=get('flow','linear','latest3')/get('flow','carry','latest3')))
    gate=bool(np.mean([r['relative_improvement'] for r in comparisons])>=.02 and max(r['latest3_speed_ratio'] for r in comparisons)<=1.02 and max(r['latest3_flow_ratio'] for r in comparisons)<=1.02)
    save('reconstruction-diagnostic.json',dict(passed=gate,records=records,comparisons=comparisons,artificial_hidden_values_invariant=invariance,boundary='only 13 released frames <=forecast T; interpolation may use later historical frames, never future forecast observations',scope='public noisy observed targets artificially hidden; NOT official Task1 mask or S_state'))
    progress('december_observability');model=SeedPredictor(OUT/'task2-round13-seed-stability');rows=[];tune=read(OUT/'task2-round11-phase-validation/tune-phase-index.json')['samples']
    for e in tune:
        s=sample(e);x=s['x'];p=model.probability(s);n=x.shape[1];rows.append(dict(panel=s['panel'],condition=s['condition'],origin=s['origin'],IoU=window_iou(s['y'],p>=model.rule['thresholds'][s['condition']],s['m']),history_speed_missing=float(x[:12,:,4].mean()),history_flow_missing=float(x[:12,:,5].mean()),history_low_quality=float((x[:12,:,3]<.75).mean()),origin_speed_missing=float(x[-1,:,4].mean()),isolated_fraction=float(s['network']['isolated'].mean()),eligible_future_cells=int(s['m'].sum())))
    summary=[]
    for c in [0,1]:
        group=[r for r in rows if r['condition']==c]
        for feature in ['history_speed_missing','history_flow_missing','history_low_quality','origin_speed_missing','isolated_fraction']:
            values=np.array([r[feature] for r in group]);y=np.array([r['IoU'] for r in group]);order=np.argsort(values);low=order[:len(order)//3];high=order[-len(order)//3:];corr=float(np.corrcoef(values,y)[0,1]) if values.std()>1e-8 else None
            summary.append(dict(condition=c,feature=feature,mean=float(values.mean()),min=float(values.min()),max=float(values.max()),raw_IoU_correlation=corr,low_third_IoU=float(y[low].mean()),high_third_IoU=float(y[high].mean()),confounded='descriptive across panels and traffic severity; not a causal effect'))
    save('december-observability.json',dict(summary=summary,windows=rows,tuning_uses_no_February=True));assert guard('after')==before;progress('diagnostic_complete',training_gate_passed=gate,seconds=time.time()-start)
if __name__=='__main__':
    try:main()
    except Exception:save('failure.json',dict(traceback=traceback.format_exc()));progress('failed');raise
