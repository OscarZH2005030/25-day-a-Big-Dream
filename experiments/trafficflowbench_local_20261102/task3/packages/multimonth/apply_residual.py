"""Apply frozen context residuals to Task1 state CSVs, local only."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,lightgbm as lgb
from residual_features import build_features,KEYS
from context_features import extra_features
HERE=Path(__file__).resolve().parent
def load_panel(panel):
 policy=json.loads((HERE/'policy.json').read_text())['policy'][panel];models=[lgb.Booster(model_file=str(HERE/'models'/panel/n)) for n in ['speed.txt','flow_per_lane.txt']];return policy,models
def correct(data_root,panel,date,frame,split,policy,models):
 p=frame[KEYS+['speed_kmh','flow_vph']].copy()
 if p.duplicated(KEYS).any():raise ValueError('Duplicate state keys')
 fd=pd.read_csv(Path(data_root)/'corridors'/panel/'network/fd_parameters.csv',dtype={'link_id':str}).groupby('link_id').lanes.mean();p['lanes']=p.link_id.map(fd)
 if p.lanes.isna().any():raise ValueError('Unknown link')
 X=build_features(data_root,panel,date,p,split)
 if policy['features']==61:X=np.column_stack([X,extra_features(data_root,panel,date,p,split)])
 if X.shape[1]!=policy['features'] or any(m.num_feature()!=X.shape[1] for m in models):raise ValueError('Feature schema mismatch')
 z=np.column_stack([m.predict(X,num_threads=4) for m in models]);v=p[['speed_kmh','flow_vph']].to_numpy();v[:,0]+=policy['alpha']*z[:,0]*25;v[:,1]+=policy['alpha']*z[:,1]*600*p.lanes.to_numpy();v=np.maximum(v,0)
 if not np.isfinite(v).all():raise ValueError('Non-finite prediction')
 p['speed_kmh']=v[:,0];p['flow_vph']=v[:,1];return p[KEYS+['speed_kmh','flow_vph']]
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--state-csv',type=Path,required=True);ap.add_argument('--panel',required=True);ap.add_argument('--split',choices=['train','validation','private'],required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.resolve()==a.state_csv.resolve():raise ValueError('Preserve baseline; choose another output')
 parts=[]
 for chunk in pd.read_csv(a.state_csv,dtype={k:str for k in KEYS},chunksize=250000):parts.append(chunk[chunk.panel==a.panel])
 f=pd.concat(parts,ignore_index=True)
 if f.empty or f.duplicated(KEYS).any():raise ValueError('Empty or duplicate state input')
 policy,models=load_panel(a.panel);out=[correct(a.data_root,a.panel,date,p,a.split,policy,models) for date,p in f.groupby(f.timestamp.str[:10],sort=True)];result=pd.concat(out,ignore_index=True);assert len(result)==len(f);a.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(a.output,index=False);print(json.dumps(dict(rows=len(result),official_score=None,kaggle_upload=False)))
if __name__=='__main__':main()
