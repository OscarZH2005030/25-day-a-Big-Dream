"""冻结相位均衡开发/校准人口，检验已有扩展模型；没有新训练、没有按二月调参。"""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import sys,json,time,hashlib,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent;R10=ROOT.parent/'task2-round10-phase-coverage';R9=ROOT.parent/'task2-round9-hierarchical';R8=ROOT.parent/'task2-round8-soft-targets';R7=ROOT.parent/'task2-round7-structured';R6=ROOT.parent/'task2-round6-ramp-physics';R5=ROOT.parent/'task2-round5-distribution';R2=ROOT.parent/'task2-round2';sys.path[1:1]=[str(R10),str(R9),str(R8),str(R7),str(R6),str(R5),str(R2)]
from coverage_runtime import CoveragePredictor
from coverage_data import matrix
from phase_dataset import build
from run_round2 import sample,role_samples,score,DATA,PANELS
from task2q.cache import atomic_json
from task2q.metrics import window_iou
from task2q.data import released_sample
from task2q.submission import validate_queue
TASK1=Path(r'C:\Users\18049\Documents\Codex\2026-10-24\trafficflowbench-186-global-road-adaptation\outputs\TrafficFlowBench-Best-20261026')
def save(n,v):atomic_json(ROOT/n,v)
def progress(phase,**kw):
 v=dict(phase=phase,pid=os.getpid(),utc=pd.Timestamp.now(tz='UTC').isoformat(),**kw);save('progress.json',v);print(json.dumps(v,ensure_ascii=True),flush=True)
def guard(stage):
 m=json.loads((TASK1/'package-manifest.json').read_text());h={n:hashlib.sha256((TASK1/n).read_bytes()).hexdigest() for n in m['files']};assert all(h[n]==v['sha256'] for n,v in m['files'].items());save('task1-guard-'+stage+'.json',dict(passed=True,files=92,hashes=h,historical_S_state=.942828633,new_Task1_score_measured=False));return h
def metric(ss,ps,t):return float(np.mean([window_iou(s['y'],p>=t,s['m']) for s,p in zip(ss,ps)]))
def main():
 started=time.time();assert (R10/'inference-reproduction.json').exists();before=guard('before');baseline=CoveragePredictor(R10);bt=baseline.rule['thresholds'];models={(p,c):lgb.Booster(model_file=str(R10/f'{p}-{c}-expanded.txt')) for p in PANELS for c in [0,1]};hold={(r['panel'],r['condition']):r['deployment_guard_passed'] for r in json.loads((R10/'temporal-holdout.json').read_text())};routes={};development=[]
 save('experiment.json',dict(training='none; existing round10 full600 tree candidates only',new_hypothesis='old development/calibration population covers phase0 only; validate phase robustness before selecting policy',phase_population='8 per phase per panel-condition; three phases, all future-positive; >=360min between any selected origins within group',selection='December same policy across phases; gains>=.005 and worst phase>=-.02; retain temporal fit holdout guard',calibration='January global alpha in0/.5/1 and coarse threshold grids; baseline frozen decision retained',audit='new February proxy population built only after policy freeze; whole month already seen in prior experiments; old410 retained separately',cpu_threads=4,official_S_queue=None,Task1='frozen92file package .942828633 historical only'))
 tune_index=build('tune',progress);tune=[sample(e) for e in tune_index['samples']]
 for p in PANELS:
  for c in [0,1]:
   ss=[s for s in tune if s['panel']==p and s['condition']==c];phase=np.array([(pd.Timestamp(s['origin']).hour*12+pd.Timestamp(s['origin']).minute//5)%3 for s in ss]);base=[baseline.probability(s) for s in ss];raw=[models[(p,c)].predict(matrix(s),num_threads=4).astype('float32').reshape(6,-1) for s in ss];bv=metric(ss,base,bt[c]);candidates=[dict(alpha_new=0.,proxy=bv,phase_gains=[0.,0.,0.],eligible=True)];phase_counts=[int((phase==k).sum()) for k in [0,1,2]]
   for alpha in [.5,1.]:
    ps=[alpha*a+(1-alpha)*b for a,b in zip(raw,base)];gains=[]
    for k in [0,1,2]:
     ii=np.flatnonzero(phase==k);gains.append(metric([ss[i] for i in ii],[ps[i] for i in ii],bt[c])-metric([ss[i] for i in ii],[base[i] for i in ii],bt[c]) if len(ii) else -1.)
    value=metric(ss,ps,bt[c]);candidates.append(dict(alpha_new=alpha,proxy=value,phase_gains=gains,eligible=hold[(p,c)] and min(phase_counts)>=4 and value>=bv+.005 and min(gains)>=-.02))
   chosen=max((a for a in candidates if a['eligible']),key=lambda a:a['proxy']);routes[(p,c)]=chosen;development.append(dict(panel=p,condition=c,temporal_guard=hold[(p,c)],phase_counts=phase_counts,baseline_proxy=bv,selected=chosen,candidates=candidates));save('december-phase-selection.json',development);save('routes.json',[dict(panel=p,condition=c,**r) for (p,c),r in routes.items()]);progress('selecting_phase_policy',panel=p,condition=c,alpha_new=chosen['alpha_new'])
 def components(ss):
  base=[];adapt=[]
  for s in ss:
   b=baseline.probability(s);r=routes[(s['panel'],s['condition'])];base.append(b)
   if not r['alpha_new']:adapt.append(b)
   else:
    q=models[(s['panel'],s['condition'])].predict(matrix(s),num_threads=4).astype('float32').reshape(6,-1);adapt.append(r['alpha_new']*q+(1-r['alpha_new'])*b)
  return base,adapt
 progress('calibration');cal_index=build('calibrate',progress);cal=[sample(e) for e in cal_index['samples']];base,adapt=components(cal);candidates=[dict(alpha_phase_policy=0.,thresholds=bt,proxy=score(cal,base,bt)['proxy_IoU'])]
 for alpha in [0.,.5,1.]:
  ps=[alpha*a+(1-alpha)*b for a,b in zip(adapt,base)]
  for t0 in [.1,.2,.35,.5]:
   for t1 in [.075,.1,.15,.25]:candidates.append(dict(alpha_phase_policy=alpha,thresholds=[t0,t1],proxy=score(cal,ps,[t0,t1])['proxy_IoU']))
 chosen=max(candidates,key=lambda a:a['proxy']);save('frozen-calibration.json',dict(selected=chosen,candidates=candidates,route_sha256=hashlib.sha256((ROOT/'routes.json').read_bytes()).hexdigest(),baseline_frozen_sha256=hashlib.sha256((R10/'frozen-calibration.json').read_bytes()).hexdigest(),proposal_hashes={f'{p}-{c}-expanded.txt':hashlib.sha256((R10/f'{p}-{c}-expanded.txt').read_bytes()).hexdigest() for p in PANELS for c in [0,1]},development_index_sha256=hashlib.sha256((ROOT/'tune-phase-index.json').read_bytes()).hexdigest(),calibration_index_sha256=hashlib.sha256((ROOT/'calibrate-phase-index.json').read_bytes()).hexdigest(),new_February_population_built_after_freeze=True,February_source_month_already_used=True,official_S_queue=None))
 progress('building_report_only_new_phase_audit');audit_index=build('audit',progress);audit=[sample(e) for e in audit_index['samples']];base,adapt=components(audit);final=[chosen['alpha_phase_policy']*a+(1-chosen['alpha_phase_policy'])*b for a,b in zip(adapt,base)];phase_reports={}
 for phase in [0,1,2]:
  ii=[i for i,s in enumerate(audit) if (pd.Timestamp(s['origin']).hour*12+pd.Timestamp(s['origin']).minute//5)%3==phase];phase_reports[str(phase)]={'round10':score([audit[i] for i in ii],[base[i] for i in ii],bt),'selected':score([audit[i] for i in ii],[final[i] for i in ii],chosen['thresholds'])}
 old=role_samples('audit');ob,oa=components(old);of=[chosen['alpha_phase_policy']*a+(1-chosen['alpha_phase_policy'])*b for a,b in zip(oa,ob)];reports={'new_phase_population_round10':score(audit,base,bt),'new_phase_population_selected':score(audit,final,chosen['thresholds']),'old410_round10':score(old,ob,bt),'old410_selected':score(old,of,chosen['thresholds'])};assert abs(reports['old410_round10']['proxy_IoU']-json.loads((R10/'proxy-results.json').read_text())['reports']['selected']['proxy_IoU'])<1e-12;save('proxy-results.json',dict(official_S_queue=None,reports=reports,by_phase=phase_reports,selected=chosen,new_groups=sum(r['alpha_new']>0 for r in routes.values()),seconds=time.time()-started,different_populations_not_directly_comparable=True,Task1_historical_S_state=.942828633,Task1_score_newly_measured=False))
 progress('predicting_official');checks={}
 for split in ['validation','private']:
  frames=[]
  for p in PANELS:
   d=DATA/'task2'/p/split;index=pd.read_csv(d/'window_index.csv');template=pd.read_csv(d/'sample_submission_queue.csv');parts=[]
   for row in index.itertuples():
    s=released_sample(DATA,p,split,row.window_id);bb,aa=components([s]);q=chosen['alpha_phase_policy']*aa[0]+(1-chosen['alpha_phase_policy'])*bb[0];keys=template[template.window_id==row.window_id].copy();ii={l:i for i,l in enumerate(s['network']['link_ids'])};h=((pd.to_datetime(keys.timestamp,utc=True)-pd.Timestamp(s['origin'])).dt.total_seconds()/300).to_numpy(dtype=int)-1;keys['queue_pred']=(q[h,keys.link_id.map(ii).to_numpy()]>=chosen['thresholds'][s['condition']]).astype(int);parts.append(keys)
   frame=pd.concat(parts,ignore_index=True);checks[split+'/'+p]=validate_queue(frame,template);frame['panel']=p;frames.append(frame)
  frame=pd.concat(frames,ignore_index=True);assert len(frame)==87000;frame.to_csv(ROOT/f'queue-{split}.csv',index=False)
 assert guard('after')==before;save('final-verification.json',dict(passed=True,rows=174000,checks=checks,Task1_package_unchanged=True,official_S_queue=None,new_phase_February_not_used_for_selection=True));progress('complete',new_phase_proxy=reports['new_phase_population_selected']['proxy_IoU'],old410_proxy=reports['old410_selected']['proxy_IoU'],seconds=time.time()-started)
if __name__=='__main__':
 import msvcrt
 lock=open(ROOT/'run.lock','a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
 try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 except OSError:raise SystemExit('Already running')
 try:main()
 except Exception:save('failure.json',dict(traceback=traceback.format_exc()));progress('failed');raise
