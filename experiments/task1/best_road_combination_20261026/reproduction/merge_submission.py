"""按官方submission_key逐键合并Task1/2/4，缺任意部分报错，不用占位0伪装完整提交。"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
COLS=['submission_id','task','speed_kmh','flow_vph','queue_pred','path_flow']
KEYS={'state':['panel','timestamp','station_id','link_id','mask_regime'],'queue':['panel','window_id','timestamp','link_id'],'odme':['panel','departure_time','path_id']}
VALUES={'state':['speed_kmh','flow_vph'],'queue':['queue_pred'],'odme':['path_flow']}
def normalized(frame,task):
 f=frame.copy()
 for c in KEYS[task]:
  if c not in f:raise ValueError('预测缺列 '+c)
  f[c]=f[c].astype(str)
 if 'timestamp' in KEYS[task]:f['timestamp']=pd.to_datetime(f.timestamp,utc=True,errors='raise').dt.strftime('%Y-%m-%dT%H:%M:%SZ')
 return f
def table(frame,task):
 f=normalized(frame,task)
 if f.duplicated(KEYS[task]).any():raise ValueError(task+'预测键重复')
 for c in VALUES[task]:
  f[c]=pd.to_numeric(f[c],errors='raise');assert np.isfinite(f[c]).all() and (f[c]>=0).all(),c
 if task=='queue':assert f.queue_pred.isin([0,1]).all()
 return f.set_index(KEYS[task])[VALUES[task]]
def merge(data,state_dir,queue_paths,odme_paths,output):
 tables={task:table(pd.concat([pd.read_csv(p,dtype={c:str for c in KEYS[task]},keep_default_na=False) for p in paths],ignore_index=True),task) for task,paths in [('queue',queue_paths),('odme',odme_paths)]}
 used={};sizes={};cached={};counts={t:0 for t in KEYS};first=True;total=0;last_id=None;output.parent.mkdir(parents=True,exist_ok=True);tmp=output.with_suffix('.csv.tmp')
 with pd.read_csv(data/'submission_key.csv',dtype=str,keep_default_na=False,chunksize=200000) as reader:
  for chunk in reader:
   ids=pd.to_numeric(chunk.submission_id,errors='raise').to_numpy();assert np.all(np.diff(ids)>0) and (last_id is None or ids[0]>last_id),'submission_id跨分片重复或顺序错误';last_id=ids[-1]
   assert not chunk.submission_id.duplicated().any();out=pd.DataFrame(dict(submission_id=chunk.submission_id,task=chunk.task,speed_kmh=0.,flow_vph=0.,queue_pred=0,path_flow=0.))
   for (task,split,panel),group in chunk.groupby(['task','split','panel'],sort=False):
    if task not in KEYS:raise ValueError('未知task '+task)
    identity=(task,split,panel)
    if task=='state':
     if identity not in cached:
      cached.clear();path=state_dir/split/(panel+'.csv');cached[identity]=table(pd.read_csv(path,dtype=str,keep_default_na=False),task);sizes[identity]=len(cached[identity])
     source=cached[identity]
    else:source=tables[task]
    normalized_group=normalized(group,task);index=pd.MultiIndex.from_frame(normalized_group[KEYS[task]]);selected=source.reindex(index)
    if selected.isna().any().any():raise ValueError('预测缺键：'+str(identity))
    out.loc[group.index,VALUES[task]]=selected.to_numpy();used[identity]=used.get(identity,0)+len(group);counts[task]+=len(group)
   assert out.submission_id.tolist()==chunk.submission_id.tolist();out[COLS].to_csv(tmp,mode='w' if first else 'a',header=first,index=False);first=False;total+=len(out)
 for identity,n in sizes.items():assert used[identity]==n,('多余state预测行',identity,n,used[identity])
 provided={('state',split,p.stem) for split in ['validation','private'] for p in (state_dir/split).glob('*.csv')};assert provided==set(sizes),'存在未使用的state道路文件'
 for task in ['queue','odme']:assert counts[task]==len(tables[task]),('多余预测行',task)
 tmp.replace(output);result=dict(passed=True,rows=total,counts=counts,keys='official submission_key order; one-to-one, finite/nonnegative, queue binary',output=str(output))
 h=hashlib.sha256()
 with output.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 result['sha256']=h.hexdigest();output.with_suffix('.checks.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result));return result
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--state-dir',type=Path,required=True);p.add_argument('--queue',type=Path,action='append',required=True);p.add_argument('--odme',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();merge(a.data_root,a.state_dir,a.queue,a.odme,a.output)
