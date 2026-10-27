"""第七轮：结构化队列模式与历史状态重建，不改Task1冻结模型。"""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import sys,json,time,hashlib,traceback,copy,gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent;R6=ROOT.parent/'task2-round6-ramp-physics';R5=ROOT.parent/'task2-round5-distribution';R2=ROOT.parent/'task2-round2';WORK=ROOT.parents[1]/'work/round7-structured';WORK.mkdir(parents=True,exist_ok=True);sys.path[1:1]=[str(R6),str(R5),str(R2)]
from run_round2 import sample,arrays,role_samples,score,DATA,PANELS
from special_features import features,latest_state
from ramp_features import extras
from ramp_runtime import RampPredictor
from structured_model import StructuredQueue,queue_loss
from task2q.cache import atomic_json
from task2q.data import make_sample,released_sample
from task2q.baselines import official_persistence,persistence
from task2q.submission import validate_queue
TASK1=Path(r'C:\Users\18049\Documents\Codex\2026-10-24\trafficflowbench-186-global-road-adaptation\outputs\TrafficFlowBench-Best-20261026');SEED=20261027;STEPS=8000
def save(n,v):atomic_json(ROOT/n,v)
def progress(phase,**kw):
 v=dict(phase=phase,pid=os.getpid(),utc=pd.Timestamp.now(tz='UTC').isoformat(),**kw);save('progress.json',v);print(json.dumps(v,ensure_ascii=True),flush=True)
def guard(stage):
 m=json.loads((TASK1/'package-manifest.json').read_text());h={n:hashlib.sha256((TASK1/n).read_bytes()).hexdigest() for n in m['files']};assert all(h[n]==v['sha256'] for n,v in m['files'].items());save('task1-guard-'+stage+'.json',dict(passed=True,files=92,hashes=h,historical_S_state=.942828633,new_Task1_score_measured=False));return h
def input_features(s):
 n=s['x'].shape[1];r,p=extras(s);x=np.concatenate([features(s),r,p],axis=1)[:n];return np.concatenate([x[:,:185],x[:,191:]],axis=1).astype('float32')
def record(s,teacher):
 x=input_features(s);raw=s['x'].transpose(1,0,2).astype('float32');static=s['network']['static'];last,_=latest_state(s);flow=np.zeros(len(last),dtype='float32')
 for t in range(13):flow=np.where(s['x'][t,:,5]<.5,s['x'][t,:,1],flow)
 return dict(x=x,raw=raw,static=static,teacher=teacher.probability(s).T.astype('float32'),last=np.column_stack([last,flow]),y=s['y'].T,m=s['m'].T,speed=s['speed_y'].T,road=PANELS.index(s['panel']),condition=s['condition'])
def prepare_fit(teacher):
 index=WORK/'index.json'
 if index.exists():
  meta=json.loads(index.read_text());data={k:np.load(WORK/(k+'.npy'),mmap_mode='r') for k in meta['arrays']};return data,meta
 entries=json.loads((R2/'expanded-fit.json').read_text())['samples'];selected=[]
 for p in PANELS:
  for c in [0,1]:selected.extend([e for e in entries if e['panel']==p and e['condition']==c][::2])
 total=sum(len(arrays(e['network_key'])['link_ids']) for e in selected);dimension=record(sample(selected[0]),teacher)['x'].shape[1];shapes=dict(x=(total,dimension),raw=(total,13,13),static=(total,6),teacher=(total,6),last=(total,2),y=(total,6),m=(total,6),speed=(total,6),flow=(total,6),flow_m=(total,6));data={k:np.lib.format.open_memmap(WORK/(k+'.npy'),mode='w+',dtype='uint8' if k in ['m','flow_m'] else 'float16',shape=shape) for k,shape in shapes.items()};cursor=0;rows=[];sums=np.zeros(dimension);squares=np.zeros(dimension)
 for i,e in enumerate(selected):
  s=sample(e);r=record(s,teacher);n=len(r['x']);v=arrays(e['day_key']);sl=e['origin_slot'];future_flow=v['flow'][sl+1:sl+7].T/s['network']['capacity'][:,None];fm=s['m'].T&np.isfinite(future_flow);r['flow']=np.where(fm,future_flow,0);r['flow_m']=fm
  for k in shapes:data[k][cursor:cursor+n]=r[k]
  sums+=r['x'].sum(0,dtype='float64');squares+=(r['x'].astype('float64')**2).sum(0);rows.append(dict(entry=e,offset=cursor,nodes=n,road=r['road'],condition=r['condition'],positive=s['has_future_queue']));cursor+=n
  if (i+1)%256==0:progress('preparing_neural_cache',windows=i+1,total=len(selected))
 for v in data.values():v.flush()
 mean=sums/total;std=np.sqrt(np.maximum(squares/total-mean**2,1e-4));meta=dict(rows=rows,arrays=list(shapes),nodes=total,windows=len(rows),features=dimension,mean=mean.tolist(),std=std.tolist(),teacher_frozen_sha256=hashlib.sha256((R6/'frozen-calibration.json').read_bytes()).hexdigest());atomic_json(index,meta);return data,meta
def tensor(r,key,device='cuda'):
 return torch.as_tensor(np.asarray(r[key]).copy(),device=device,dtype=torch.bool if key in ['m','flow_m'] else torch.float32)
@torch.inference_mode()
def predict(model,records,head):
 model.eval();result=[]
 for r in records:result.append(model(tensor(r,'x'),r['road'],r['condition'])[head].T.cpu().numpy())
 return result
def main():
 started=time.time();torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED);rng=np.random.default_rng(SEED);assert torch.cuda.is_available();before=guard('before');teacher=RampPredictor(R6)
 save('experiment.json',dict(seed=SEED,optimizer_updates=STEPS,model='shared local MLP128 x2 residual; road/condition embeddings; independent6 vs pattern64; future speed/flow residual + masked historical speed/flow reconstruction',mask_policy='partial future-pattern likelihood; unknown bits marginalized, never zero-filled',fit_windows='every second window per panel-condition from same 8192 pool',primary_loss='class-balanced BCE + window soft IoU; pattern partial NLL; causal tree distillation; continuous state auxiliary losses',cpu_threads=4,GPU_serial=True,cpu_temperature=None,Task1='original 92-file package frozen; historical .942828633, not remeasured',official_S_queue=None))
 data,meta=prepare_fit(teacher);save('dataset-audit.json',dict(windows=meta['windows'],link_sequences=meta['nodes'],features=meta['features'],teacher='round6 frozen causal inputs; in-fit distillation, not independent labels',auxiliary_reconstruction='artificially masked T-5 observed speed/flow; diagnostic only, not official Task1 evaluation'));tune=role_samples('tune');tune_records=[record(s,teacher) for s in tune];model=StructuredQueue(meta['mean'],meta['std']).cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,eta_min=3e-5);first=0;best={'independent':-1.,'pattern':-1.};trace=[]
 groups={}
 for r in meta['rows']:groups.setdefault((r['road'],r['condition'],r['positive']),[]).append(r)
 keys=list(groups)
 if (ROOT/'last.pt').exists():
  ck=torch.load(ROOT/'last.pt',weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);first=ck['step'];best=ck['best'];trace=ck['trace'];rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
 progress('training_neural',step=first,total=STEPS)
 for step in range(first,STEPS):
  road=step%8;c=(step//8)%2;positive=bool(rng.random()<.8);g=groups.get((road,c,positive),groups[(road,c,True)]);row=g[int(rng.integers(len(g)))];sl=slice(row['offset'],row['offset']+row['nodes']);r={k:v[sl] for k,v in data.items()};r.update(road=road,condition=c);model.train();opt.zero_grad(set_to_none=True);out=model(tensor(r,'x'),road,c);y=tensor(r,'y');m=tensor(r,'m');loss,_=queue_loss(out,y,m,tensor(r,'teacher'),model.bits);target_speed=tensor(r,'speed');target_flow=tensor(r,'flow');fm=tensor(r,'flow_m');last=tensor(r,'last');future=out['future_residual'];speed=last[:,0,None]+future[:,:6];flow=last[:,1,None]+future[:,6:];aux=((F.smooth_l1_loss(speed,target_speed,reduction='none')*m).sum()/m.sum().clamp_min(1)+(F.smooth_l1_loss(flow,target_flow,reduction='none')*fm).sum()/fm.sum().clamp_min(1));raw=tensor(r,'raw');static=tensor(r,'static');valid=(raw[:,11,4]<.5)&(raw[:,11,5]<.5)&(raw[:,11,7]>.5)&(raw[:,11,3]>=.75);mask=valid&(torch.rand(len(raw),device='cuda')<.3);recon=model.reconstruct(raw,static,road,c,mask);rt=raw[:,11,:2];rec=(F.smooth_l1_loss(recon,rt,reduction='none').mean(1)*mask).sum()/mask.sum().clamp_min(1);loss=loss+.1*aux+.1*rec;assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();scheduler.step()
  if (step+1)%200==0:progress('training_neural',step=step+1,total=STEPS,loss=float(loss.detach()),state_aux=float(aux.detach()),reconstruction_aux=float(rec.detach()))
  if (step+1)%1000==0:
   report={}
   for head in ['independent','pattern']:
    ps=predict(model,tune_records,head);values=[dict(threshold=t,proxy=score(tune,ps,t)['proxy_IoU']) for t in [.05,.1,.2,.3,.4,.5,.6,.7]];winner=max(values,key=lambda a:a['proxy']);report[head]=winner
    if winner['proxy']>best[head]:best[head]=winner['proxy'];torch.save(dict(model=model.state_dict(),mean=meta['mean'],std=meta['std'],step=step+1,head=head),ROOT/f'best-{head}.pt')
   trace.append(dict(step=step+1,report=report));save('development-trace.json',trace);torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),step=step+1,best=best,trace=trace,rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),ROOT/'last.pt')
 progress('causal_acceptance');checks=[]
 for p in PANELS:
  for c in [0,1]:
   e=next(r['entry'] for r in meta['rows'] if r['entry']['panel']==p and r['condition']==c);s=sample(e);v=copy.deepcopy(arrays(e['day_key']));slot=e['origin_slot']
   for k in ['speed','flow','occupancy','quality','origin_speed','origin_flow','origin_occupancy','origin_quality']:v[k][slot+1:]=999
   v['eligible'][slot+1:]=False;s2=make_sample(v,s['network'],slot,p);assert np.array_equal(input_features(s),input_features(s2));checks.append(dict(panel=p,condition=c,passed=True))
 save('causality-check.json',dict(passed=True,groups=checks,auxiliary_state_branch_has_separate_masked_input=True,no_Task1_offline_output_inputs=True))
 models={}
 for head in ['independent','pattern']:
  ck=torch.load(ROOT/f'best-{head}.pt',weights_only=False);m2=StructuredQueue(ck['mean'],ck['std']).cuda();m2.load_state_dict(ck['model']);models[head]=m2
 progress('calibration');cal=role_samples('calibrate');cr=[record(s,teacher) for s in cal];base=[teacher.probability(s) for s in cal];bt=teacher.rule['thresholds'];pn={h:predict(m,cr,h) for h,m in models.items()};candidates=[dict(head='baseline',alpha_neural=0.,thresholds=bt,proxy=score(cal,base,bt)['proxy_IoU'])]
 for head,psn in pn.items():
  for alpha in [.25,.5,.75,1.]:
   ps=[alpha*a+(1-alpha)*b for a,b in zip(psn,base)]
   for t0 in [.1,.2,.3,.4,.5,.6,.7]:
    for t1 in [.1,.15,.2,.3,.4,.5,.6]:candidates.append(dict(head=head,alpha_neural=alpha,thresholds=[t0,t1],proxy=score(cal,ps,[t0,t1])['proxy_IoU']))
 winner=max(candidates,key=lambda a:a['proxy']);hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('best-*.pt')};save('frozen-calibration.json',dict(selected=winner,candidates=candidates,hashes=hashes,baseline_frozen_sha256=hashlib.sha256((R6/'frozen-calibration.json').read_bytes()).hexdigest(),february='reused report only',Task1_frozen=True))
 progress('evaluating');audit=role_samples('audit');ar=[record(s,teacher) for s in audit];base=[teacher.probability(s) for s in audit];pa={h:predict(m,ar,h) for h,m in models.items()};final=base if winner['head']=='baseline' else [winner['alpha_neural']*a+(1-winner['alpha_neural'])*b for a,b in zip(pa[winner['head']],base)];reports={'official_persistence':score(audit,[official_persistence(s,s['network']) for s in audit]),'carry_T':score(audit,[persistence(s,s['network'],True) for s in audit]),'round6_frozen':score(audit,base,bt),'independent_fixed05':score(audit,pa['independent']),'pattern_fixed05':score(audit,pa['pattern']),'selected':score(audit,final,winner['thresholds']),'selected_360min_spacing_diagnostic':score(audit,final,winner['thresholds'],True)};assert abs(reports['round6_frozen']['proxy_IoU']-json.loads((R6/'proxy-results.json').read_text())['reports']['selected']['proxy_IoU'])<1e-12;save('proxy-results.json',dict(official_S_queue=None,reports=reports,selected=winner,seconds=time.time()-started,Task1_historical_S_state=.942828633,Task1_score_newly_measured=False))
 progress('predicting_official');checks={}
 for split in ['validation','private']:
  frames=[]
  for p in PANELS:
   d=DATA/'task2'/p/split;index=pd.read_csv(d/'window_index.csv');template=pd.read_csv(d/'sample_submission_queue.csv');parts=[]
   for row in index.itertuples():
    s=released_sample(DATA,p,split,row.window_id);b=teacher.probability(s);pr=b if winner['head']=='baseline' else winner['alpha_neural']*predict(models[winner['head']],[record(s,teacher)],winner['head'])[0]+(1-winner['alpha_neural'])*b;keys=template[template.window_id==row.window_id].copy();ii={l:i for i,l in enumerate(s['network']['link_ids'])};h=((pd.to_datetime(keys.timestamp,utc=True)-pd.Timestamp(s['origin'])).dt.total_seconds()/300).to_numpy(dtype=int)-1;keys['queue_pred']=(pr[h,keys.link_id.map(ii).to_numpy()]>=winner['thresholds'][s['condition']]).astype(int);parts.append(keys)
   frame=pd.concat(parts,ignore_index=True);checks[split+'/'+p]=validate_queue(frame,template);frame['panel']=p;frames.append(frame)
  frame=pd.concat(frames,ignore_index=True);assert len(frame)==87000;frame.to_csv(ROOT/f'queue-{split}.csv',index=False)
 assert guard('after')==before;assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items());save('final-verification.json',dict(passed=True,rows=174000,checks=checks,frozen_hashes_unchanged=True,Task1_package_unchanged=True,official_S_queue=None));progress('complete',proxy=reports['selected']['proxy_IoU'],seconds=time.time()-started)
if __name__=='__main__':
 import msvcrt
 lock=open(ROOT/'run.lock','a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
 try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 except OSError:raise SystemExit('Already running')
 try:main()
 except Exception:save('failure.json',dict(traceback=traceback.format_exc()));progress('failed');raise
