"""Apply frozen public-input severe-mask residual mixture locally."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,lightgbm as lgb
from residual_features import build_features,KEYS
from context_features import extra_features
HERE=Path(__file__).resolve().parent

def load_panel(panel):
 policy=json.loads((HERE/'policy.json').read_text())['policy'][panel];parent=[lgb.Booster(model_file=str(HERE/'models'/panel/'parent'/n)) for n in ['speed.txt','flow_per_lane.txt']];new=[lgb.Booster(model_file=str(HERE/'models'/panel/'augmented'/n)) for n in ['speed.txt','flow_per_lane.txt']] if policy['gamma']>0 else None;return policy,dict(parent=parent,augmented=new)

def correct(data_root,panel,date,frame,split,policy,models):
 p=frame[KEYS+['speed_kmh','flow_vph']].copy()
 if p.duplicated(KEYS).any():raise ValueError('Duplicate state keys')
 fd=pd.read_csv(Path(data_root)/'corridors'/panel/'network/fd_parameters.csv',dtype={'link_id':str}).groupby('link_id').lanes.mean();p['lanes']=p.link_id.map(fd)
 if p.lanes.isna().any():raise ValueError('Unknown link')
 X=build_features(data_root,panel,date,p,split)
 if policy['parent_features']==61 or policy['gamma']>0:X=np.column_stack([X,extra_features(data_root,panel,date,p,split)])
 base=p[['speed_kmh','flow_vph']].to_numpy();scale=np.column_stack([np.full(len(p),25.),600*p.lanes.to_numpy()])
 def apply(mm,alpha,nfeatures):
  if any(m.num_feature()!=nfeatures for m in mm):raise ValueError('Feature schema mismatch')
  residual=np.column_stack([m.predict(X[:,:nfeatures],num_threads=4) for m in mm]);return np.maximum(base+alpha*residual*scale,0)
 parent=apply(models['parent'],policy['parent_alpha'],policy['parent_features']);gamma=policy['gamma'];v=parent if gamma==0 else parent+gamma*(apply(models['augmented'],1.,61)-parent)
 if not np.isfinite(v).all() or (v<0).any():raise ValueError('Invalid state prediction')
 p['speed_kmh']=v[:,0];p['flow_vph']=v[:,1];return p[KEYS+['speed_kmh','flow_vph']]

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--state-csv',type=Path,required=True);ap.add_argument('--panel',required=True);ap.add_argument('--split',choices=['train','validation','private'],required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.resolve()==a.state_csv.resolve():raise ValueError('Preserve baseline; choose another output')
 parts=[chunk[chunk.panel==a.panel] for chunk in pd.read_csv(a.state_csv,dtype={k:str for k in KEYS},chunksize=250000)];f=pd.concat(parts,ignore_index=True)
 if f.empty or f.duplicated(KEYS).any():raise ValueError('Empty or duplicate state input')
 policy,models=load_panel(a.panel);result=pd.concat([correct(a.data_root,a.panel,date,p,a.split,policy,models) for date,p in f.groupby(f.timestamp.str[:10],sort=False)],ignore_index=True);result=f[KEYS].merge(result,on=KEYS,how='left',validate='one_to_one',sort=False);assert len(result)==len(f);a.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(a.output,index=False);print(json.dumps(dict(rows=len(result),official_score=None,kaggle_upload=False)))
if __name__=='__main__':main()
