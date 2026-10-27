"""0.9428组合完整入口：原始公开数据→三类从零训练→冻结路由→评估/Task1提交部分。

历史权重推理与新训练严格分开。不会将历史0.9428填入新训练结果。
"""
import os,sys,json,argparse,hashlib,shutil,subprocess,importlib.util,time,gc,copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'reproduction';VENDOR=HERE/'vendor'
sys.path.insert(0,str(ROOT))
KEYS=['panel','timestamp','station_id','link_id','mask_regime']
PANELS=list(json.loads((ROOT/'selected-road-models.json').read_text(encoding='utf8'))['routes'])
PLAN=json.loads((VENDOR/'jayden2/reference-split.json').read_text(encoding='utf8'))
SCALE=np.array([25.,600.])
LOGDIR=None
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf8');os.replace(t,p)
def log(**kw):print(json.dumps(dict(pid=os.getpid(),utc=pd.Timestamp.now(tz='UTC').isoformat(),**kw),ensure_ascii=False),flush=True)
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def execute(argv,env=None):
 log(command=[str(x) for x in argv]);LOGDIR.mkdir(parents=True,exist_ok=True);path=LOGDIR/(str(time.time_ns())+'.log')
 with path.open('wb') as f:
  process=subprocess.Popen([str(x) for x in argv],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  for line in iter(process.stdout.readline,b''):
   f.write(line);f.flush();sys.stdout.buffer.write(line);sys.stdout.buffer.flush()
  code=process.wait()
 if code:
  save(LOGDIR/'last-failure.json',dict(command=[str(x) for x in argv],returncode=code,log=str(path)));raise subprocess.CalledProcessError(code,argv)
def data_audit(a):
 expected=json.loads((HERE/'expected-data-sha256.json').read_text(encoding='utf8'));bad=[]
 for i,(name,digest) in enumerate(expected.items(),1):
  p=a.data_root/name
  if not p.is_file() or sha(p)!=digest:bad.append(name)
  if i%500==0:log(phase='raw_data_sha256',files=i,total=len(expected))
 report=dict(all_data_match=not bad,files=len(expected),mismatches=bad,data_root=str(a.data_root),expected_manifest_sha256=sha(HERE/'expected-data-sha256.json'));save(a.work_dir/'data-audit.json',report);assert not bad,bad[:10];return report
def history(a):
 from feature_runtime import Experiment
 for road in PANELS:
  dest=a.work_dir/'history'/road
  e=Experiment(SimpleNamespace(release_root=a.data_root,panel=road,output=dest,seed=2026,train_start='2030-06-29'))
  if not (dest/'history.npz').exists():e.profile();assert e.profile_dates==PLAN['history_dates']
  else:
   h=np.load(dest/'history.npz');assert list(h['stations'])==e.stations
  save(dest/'experiment.json',dict(panel=road,historical_start=PLAN['history_dates'][0],historical_end=PLAN['history_dates'][-1]))
  target=a.work_dir/'shared/cache'/road/'keys.parquet';target.parent.mkdir(parents=True,exist_ok=True)
  if not target.exists():pd.read_csv(a.data_root/'task1'/road/'train/sample_submission_state.csv',dtype=str,usecols=KEYS).to_parquet(target,index=False)
  log(phase='history_and_template_ready',panel=road,history_days=len(e.profile_dates))
def stage_jayden(a,group):
 assert a.device=='cuda','Jayden186原训练配方需要CUDA；CPU仅支持权重推理/评分及树模型。'
 dest=a.work_dir/group;dest.mkdir(parents=True,exist_ok=True)
 for p in (VENDOR/group).iterdir():
  if p.is_file():shutil.copy2(p,dest/p.name)
 shutil.copytree(ROOT/'feature-source',dest/'feature-source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 # 唯一运行时移植：旧机器绝对路径改为本次数据/工作目录；训练配方不变。
 p=dest/'train_global.py';source=p.read_text(encoding='utf8');lines=source.splitlines()
 for i,line in enumerate(lines):
  if line.startswith('ROOT=Path(__file__)'):lines[i]="ROOT=Path(__file__).resolve().parent;WORK=Path(os.environ['TFB_REPRO_WORK'])/'shared';CACHE=Path(os.environ['TFB_REPRO_WORK'])/'jayden-cache'"
  elif line.startswith('DATA=Path('):lines[i]="DATA=Path(os.environ['TFB_DATA_ROOT'])"
 if group=='jayden8':
  # 保留原ROADS顺序与+2种子偏移，仅略过两路最终未采用的Jayden模型。
  lines='\n'.join(lines).replace(' for road in ROADS:\n',' for road in ROADS:\n  if road in [\'D12_I405_S\',\'D7_I210_E\']:continue\n').splitlines()
 generated=('\n'.join(lines)+'\n').replace('mean_two_roads','mean_selected_roads').replace('eight newly trained roads; full ten-road summary is generated separately','selected Jayden roads only; complete combo must use reproduction evaluation')
 p.write_text(generated,encoding='utf8')
 save(dest/'source-portability.json',dict(vendor_sha256=sha(VENDOR/group/'train_global.py'),generated_sha256=sha(p),changes=['DATA/WORK/CACHE relative to configured roots','jayden8 skips two unselected roads without changing ROADS indices or per-road seeds'] if group=='jayden8' else ['DATA/WORK/CACHE relative to configured roots']))
 roads=['D12_I5_N','D7_I405_N'] if group=='jayden2' else ['D12_I405_N','D12_I405_S','D12_I5_S','D7_I10_E','D7_I10_W','D7_I210_E','D7_I210_W','D7_I405_S']
 for r in roads:
  target=dest/'history'/r;target.mkdir(parents=True,exist_ok=True);shutil.copy2(a.work_dir/'history'/r/'history.npz',target/'history.npz')
 env=os.environ.copy();env.update(TFB_DATA_ROOT=str(a.data_root),TFB_REPRO_WORK=str(a.work_dir),PYTHONUTF8='1')
 execute([sys.executable,'-X','utf8','-u',p,'smoke'],env)
 if not a.smoke:execute([sys.executable,'-X','utf8','-u',p,'run'],env)
def stage_lgb(a):
 dest=a.work_dir/'global186/outputs';dest.mkdir(parents=True,exist_ok=True);shutil.copytree(ROOT/'feature-source',dest/'feature-source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 source=(VENDOR/'global186.py').read_text(encoding='utf8');source='\n'.join("DATA=Path(os.environ['TFB_DATA_ROOT'])" if line.startswith('DATA=Path(') else line for line in source.splitlines())+'\n';p=dest/'train_global.py';p.write_text(source,encoding='utf8');shutil.copy2(a.work_dir/'data-audit.json',dest/'data-audit.json');os.environ['TFB_DATA_ROOT']=str(a.data_root)
 m=load('repro_global186',p)
 # 此组合只采用该配方的树分支；不训练未选用的旧全局小Transformer。
 if a.smoke:m.prepare(True);m.smoke();return
 m.prepare();models=m.fit_lgb(m.PANELS,m.OUT/'global/lightgbm');m.fit_lgb(['D12_I405_S'],m.OUT/'adapted/D12_I405_S/lightgbm',models)
def original_args(a):
 template=json.loads((HERE/'historical-evidence/full_dataset_training_v2/run_config.json').read_text(encoding='utf8'));out=a.work_dir/'original51/D7_I210_E';out.mkdir(parents=True,exist_ok=True)
 template.update(release_root=str(a.data_root),output=str(out),history=str(out/'history.npz'),resume=str(out/'last.pt') if (out/'last.pt').exists() else None,device=a.device,prepare_only=a.smoke)
 return SimpleNamespace(**template)
def original_tree(a,e):
 import lightgbm as lgb
 out=a.work_dir/('original51/D7_I210_E/trees-smoke' if a.smoke else 'original51/D7_I210_E/trees');out.mkdir(parents=True,exist_ok=True);arrays={}
 for role,dates in [('training',PLAN['training_dates']),('internal',PLAN['internal_validation_dates'])]:
  if a.smoke:dates=dates[:1]
  xx=[];yy=[]
  for date in dates:
   keys=e.day_targets[date];x,y,b,_,_,names=e.features(date,keys);xx.append(x);yy.append((y/SCALE).astype('float32').astype('float64')*SCALE-b)
  arrays[role]=(np.concatenate(xx),np.concatenate(yy))
 params=json.loads((HERE/'historical-evidence/lightgbm_time80_20/experiment.json').read_text(encoding='utf8'))['parameters']
 for c,name in enumerate(['speed_kmh','flow_per_lane']):
  if (out/(name+'.txt')).exists() and not a.smoke:continue
  train=lgb.Dataset(arrays['training'][0],label=arrays['training'][1][:,c],feature_name=names);valid=lgb.Dataset(arrays['internal'][0],label=arrays['internal'][1][:,c],reference=train)
  model=lgb.train(params,train,num_boost_round=3 if a.smoke else 2000,valid_sets=[valid],callbacks=[lgb.early_stopping(100,verbose=False)]);(out/(name+'.txt')).write_text(model.model_to_string(),encoding='utf8')
 save(out/'fit.json',dict(parameters=params,rounds_cap=2000,patience=100,smoke=a.smoke,label_precision='float32 normalized then restored to physical residual',rows={r:len(v[0]) for r,v in arrays.items()}))
def stage_original(a):
 sys.path.insert(0,str(VENDOR/'original51'));worker=load('upstream_worker',VENDOR/'original51/full_train_worker.py');args=original_args(a)
 worker.run(args)
 e=worker.Experiment(args);e.profile();e.targets();original_tree(a,e)
def assemble(a):
 dest=a.work_dir/'trained-bundle';shutil.copytree(ROOT,dest,dirs_exist_ok=True,ignore=shutil.ignore_patterns('models','runs','predictions','__pycache__','reproduction','audit','最新成绩.txt','中文说明.txt','.venv-task1-repro','release-cache'));routes=json.loads((ROOT/'selected-road-models.json').read_text(encoding='utf8'))
 for road,route in routes['routes'].items():
  folder=dest/'models'/road;folder.mkdir(parents=True,exist_ok=True)
  if route['model']=='jayden_186':
   group='jayden2' if road in ['D12_I5_N','D7_I405_N'] else 'jayden8';base=a.work_dir/group;sources=dict(transformer=base/'results'/road/'best.pt',history=base/'history'/road/'history.npz',normalization=a.work_dir/'jayden-cache'/road/'normalization.npz')
  elif route['model']=='adapted_lightgbm':sources={k:a.work_dir/'global186/outputs/adapted'/road/'lightgbm'/(k+'.txt') for k in ['speed','flow']}
  else:
   base=a.work_dir/'original51/D7_I210_E';sources=dict(transformer=base/'best.pt',history=base/'history.npz',speed=base/'trees/speed_kmh.txt',flow=base/'trees/flow_per_lane.txt')
  route['paths']={};route['sha256']={}
  for key,src in sources.items():
   target=folder/src.name;shutil.copy2(src,target);route['paths'][key]=target.relative_to(dest).as_posix();route['sha256'][key]=sha(target)
  if route['model']=='adapted_lightgbm':
   target=folder/'history.npy';shutil.copy2(a.work_dir/'global186/work/cache'/road/'history.npy',target);route['history']=target.relative_to(dest).as_posix();route['history_sha256']=sha(target)
  route['historical_score_common55']=route.pop('score_common55',None);route['validation']='newly trained weights; must evaluate, historical score is not new score'
 save(dest/'selected-road-models.json',routes);save(dest/'training-origin.json',dict(from_scratch=True,work_dir=str(a.work_dir),fixed_routes=True,actual_score=None))
 text='新训练的冻结路由组合，尚未评估。历史0.942828633只是来源，不是本次训练分数。请执行pipeline.py evaluate --bundle并查看新metrics.json。\n'
 for name in ['最新成绩.txt','中文说明.txt','README.md']:(dest/name).write_text(text,encoding='utf8')
 save(dest/'package-manifest.json',dict(files={p.relative_to(dest).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in dest.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='package-manifest.json'}));log(phase='assembled',bundle=str(dest))
def metric(rows):
 regs=[]
 for regime in ['R1','R2','R3']:
  values=[r for r in rows if r['regime']==regime];n=sum(r['rows'] for r in values);
  if not n:continue
  speed=(sum(r['speed_sse'] for r in values)/n)**.5;flow=(sum(r['flow_sse'] for r in values)/n)**.5;regs.append(dict(regime=regime,rows=n,speed_rmse_kmh=speed,flow_rmse_per_lane=flow,S_state=.54*max(0,1-speed/25)+.46*max(0,1-flow/600)))
 return dict(S_state=float(np.mean([r['S_state'] for r in regs])),regimes=regs,rows=sum(r['rows'] for r in regs))
def infer(a,split):
 import torch
 torch.set_num_threads(4);bundle=a.bundle or ROOT;sys.path.insert(0,str(bundle));p=load('frozen_predict',bundle/'predict_selected.py');routes=p.load_routes();days=PLAN['holdout_dates'] if split=='train' else pd.date_range('2031-03-01','2031-03-31').strftime('%Y-%m-%d').tolist() if split=='validation' else pd.date_range('2031-04-01','2031-04-30').strftime('%Y-%m-%d').tolist()
 if a.smoke:days=days[:1]
 dest=a.work_dir/('evaluation' if split=='train' else 'submissions')/split;dest.mkdir(parents=True,exist_ok=True);signature=dict(bundle=str(bundle),routes_sha256=sha(bundle/'selected-road-models.json'),predict_source_sha256=sha(bundle/'predict_selected.py'),data_root=str(a.data_root),split=split,days=days,smoke=a.smoke)
 lock=dest/'input-signature.json'
 if lock.exists():assert json.loads(lock.read_text(encoding='utf8'))==signature,'预测输入变化，拒绝复用旧结果'
 else:save(lock,signature)
 panel_results=[];frames=[]
 for panel in PANELS:
  parts=[]
  for chunk in pd.read_csv(a.data_root/'task1'/panel/split/'sample_submission_state.csv',dtype=str,usecols=KEYS,chunksize=250000):parts.append(chunk[chunk.timestamp.str[:10].isin(days)])
  keys=pd.concat(parts,ignore_index=True);assert not keys.duplicated(KEYS).any();by_date={d:g.reset_index(drop=True) for d,g in keys.groupby(keys.timestamp.str[:10])};p.keys_for=lambda data,road,which,date:by_date[date].copy();daily=[]
  for date in days:
   file=dest/(panel+'_'+date+'.parquet')
   if file.exists():pred=pd.read_parquet(file)
   else:
    # 推理时拦截任何未掩码真值读取，避免提交流程误用标签。
    reader=pd.read_parquet
    def guarded(path,*args,**kw):
     if '/mainline_states/' in str(path).replace('\\','/'):raise RuntimeError('推理禁止读取未掩码真值')
     return reader(path,*args,**kw)
    pd.read_parquet=guarded
    try:pred=p.predict(panel,date,routes[panel],a.data_root,dest,a.device)
    finally:pd.read_parquet=reader
   assert pred[KEYS].astype(str).reset_index(drop=True).equals(by_date[date][KEYS].astype(str));assert np.isfinite(pred[['speed_kmh','flow_vph']].to_numpy()).all();assert (pred[['speed_kmh','flow_vph']]>=0).all().all()
   if split=='train':
    source=next((a.data_root/'corridors'/panel/'train/mainline_states').rglob('synthetic_mainline_'+date.replace('-','_')+'.parquet'));truth=pd.read_parquet(source);truth.timestamp=truth.timestamp.astype(str);m=pred.merge(truth[['timestamp','station_id','link_id','speed_kmh','flow_vph','pct_observed']],on=['timestamp','station_id','link_id'],validate='one_to_one',suffixes=('_pred','_true'));assert len(m)==len(pred) and (m.pct_observed>=75).all();lanes=pd.read_csv(a.data_root/'corridors'/panel/'network/fd_parameters.csv').set_index('link_id').lanes;v=m.speed_kmh_pred-m.speed_kmh_true;q=(m.flow_vph_pred-m.flow_vph_true)/m.link_id.map(lanes);regime=pred.mask_regime.unique();assert len(regime)==1;daily.append(dict(date=date,regime=regime[0],rows=len(m),speed_sse=float(np.square(v).sum()),flow_sse=float(np.square(q).sum())))
   log(phase='evaluating' if split=='train' else 'predicting_submission',panel=panel,date=date,rows=len(pred))
  if split=='train':panel_results.append(dict(panel=panel,**metric(daily),days=daily))
  else:
   # 逐道路写CSV，避免把所有道路预测同时放入内存。
   output=dest/(panel+'.csv');first=True
   for date in days:
    pred=pd.read_parquet(dest/(panel+'_'+date+'.parquet'));pred[KEYS+['speed_kmh','flow_vph']].to_csv(output,mode='w' if first else 'a',header=first,index=False);first=False
   assert sum(len(pd.read_parquet(dest/(panel+'_'+d+'.parquet'))) for d in days)==len(keys)
  del parts,keys,by_date;gc.collect()
 if split=='train':
  result=dict(mean=float(np.mean([r['S_state'] for r in panel_results])),roads=panel_results,targets=sum(r['rows'] for r in panel_results),source='published frozen weights' if bundle==ROOT else 'supplied bundle; inspect training-origin.json',scope='previously viewed55day model-selection validation, not independent test',smoke=a.smoke);save(dest/'metrics.json',result)
  if not a.smoke:assert result['targets']==6636705
  if (bundle/'training-origin.json').exists() and not a.smoke:
   origin=json.loads((bundle/'training-origin.json').read_text(encoding='utf8'));origin['actual_score']=result['mean'];origin['evaluation_report']=str(dest/'metrics.json');save(bundle/'training-origin.json',origin)
   (bundle/'最新成绩.txt').write_text('本次新训练实际55日S_state：'+str(result['mean'])+'\n已查看过的选模验证，不是官方Kaggle分数。\n',encoding='utf8')
   save(bundle/'package-manifest.json',dict(files={p.relative_to(bundle).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in bundle.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not any(k in p.relative_to(bundle).parts for k in ['runs','predictions']) and p.name!='package-manifest.json'}))
  log(phase='evaluation_complete',mean=result['mean'],targets=result['targets'])
 else:save(dest/'part-manifest.json',dict(split=split,complete=not a.smoke,days=days,panels=PANELS,scope='Task1 state part; not whole competition submission',routes_sha256=signature['routes_sha256']))
def train_all(a):
 stages=['audit','history','jayden2','jayden8','global186','original51','assemble']
 for stage in stages:
  command=[sys.executable,'-X','utf8','-u',Path(__file__),stage,'--data-root',a.data_root,'--work-dir',a.work_dir,'--device',a.device]
  if a.smoke:command.append('--smoke')
  if a.smoke and stage=='assemble':continue
  execute(command)
def main():
 global LOGDIR
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['audit','history','jayden2','jayden8','global186','original51','assemble','train-all','evaluate','submit']);parser.add_argument('--data-root',type=Path,required=True);parser.add_argument('--work-dir',type=Path,default=ROOT/'runs/reproduction');parser.add_argument('--device',choices=['cpu','cuda'],default='cuda');parser.add_argument('--bundle',type=Path);parser.add_argument('--smoke',action='store_true');a=parser.parse_args();a.data_root=a.data_root.resolve();a.work_dir=a.work_dir.resolve();a.bundle=a.bundle.resolve() if a.bundle else None
 assert (a.data_root/'task1').is_dir();assert not a.work_dir.is_relative_to(a.data_root);assert a.work_dir!=ROOT and not ROOT.is_relative_to(a.work_dir);a.work_dir.mkdir(parents=True,exist_ok=True)
 LOGDIR=a.work_dir/'logs'
 if a.stage in ['audit','history','train-all','jayden2','jayden8','global186','original51','assemble'] and not a.smoke:
  import importlib.metadata
  sources=list((HERE/'vendor').rglob('*.py'))+list((ROOT/'feature-source').rglob('*.py'))+[Path(__file__),ROOT/'feature_runtime.py',ROOT/'jayden_model.py',ROOT/'observation_features.py',ROOT/'github_graph.py']
  identity=dict(data_root=str(a.data_root),sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources if '__pycache__' not in p.parts},routes_sha256=sha(ROOT/'selected-road-models.json'),raw_expected_sha256=sha(HERE/'expected-data-sha256.json'),versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','pyarrow','torch','lightgbm']})
  frozen=a.work_dir/'reproduction-input-lock.json'
  if frozen.exists():assert json.loads(frozen.read_text(encoding='utf8'))==identity,'数据路径/源码/环境变化，请使用新工作目录，禁止混用旧断点'
  else:save(frozen,identity)
 import msvcrt
 lock_name='compute.lock' if a.stage in ['jayden2','jayden8','global186','original51','evaluate','submit'] else a.stage+'.lock'
 guard=(a.work_dir/lock_name).open('a+b');guard.write(b'0');guard.flush();guard.seek(0);msvcrt.locking(guard.fileno(),msvcrt.LK_NBLCK,1)
 if a.stage=='audit':data_audit(a)
 elif a.stage=='history':history(a)
 elif a.stage in ['jayden2','jayden8']:stage_jayden(a,a.stage)
 elif a.stage=='global186':stage_lgb(a)
 elif a.stage=='original51':stage_original(a)
 elif a.stage=='assemble':assemble(a)
 elif a.stage=='train-all':train_all(a)
 elif a.stage=='evaluate':infer(a,'train')
 else:
  for split in ['validation','private']:infer(a,split)
  if not a.smoke:
   destination=a.work_dir/'submissions/state_submission.csv';first=True;total=0
   for split in ['validation','private']:
    for panel in PANELS:
     for chunk in pd.read_csv(a.work_dir/'submissions'/split/(panel+'.csv'),chunksize=200000):
      chunk.to_csv(destination,mode='w' if first else 'a',header=first,index=False);first=False;total+=len(chunk)
   assert total==6735795
   save(a.work_dir/'submissions/state-submission-checks.json',dict(passed=True,rows=total,validation_private_complete=True,output=str(destination),sha256=sha(destination),scope='Task1 state rows only; merge with Task2 and Task4 for Kaggle'))
if __name__=='__main__':main()

