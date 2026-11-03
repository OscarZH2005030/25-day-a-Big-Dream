"""Causal current-state reconstruction auxiliary; not a queue forecast model."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import lightgbm as lgb
HERE=Path(__file__).resolve().parent
def features(x,static,up,down):
    x=np.asarray(x,dtype='float32').copy();static=np.asarray(static,dtype='float32');n=x.shape[1]
    if x.shape!=(13,n,16) or static.shape!=(n,8):raise ValueError('Expected x[13,N,16], static[N,8]')
    original=x[-1].copy();valid=(x[:,:,7]>.5)&(x[:,:,3]>=.75)&(x[:,:,4]<.5);last=np.full(n,np.nan);age=np.full(n,65.)
    for t in range(12):last=np.where(valid[t],x[t,:,0],last);age=np.where(valid[t],(12-t)*5,age)
    x[-1,:,0]=1;x[-1,:,4]=1;x[-1,:,9:11]=0;neighbor=[]
    for matrix in [up,down]:
        matrix=np.asarray(matrix,dtype='float32').copy()
        if matrix.shape!=(n,n) or not np.isfinite(matrix).all() or (matrix<0).any():raise ValueError('Expected finite nonnegative neighbor matrix')
        np.fill_diagonal(matrix,0);den=matrix@valid[-1].astype('float32');num=matrix@np.where(valid[-1],original[:,0],0);value=np.divide(num,den,out=np.full(n,np.nan),where=den>0);neighbor.extend([value,den])
    return np.column_stack([x.transpose(1,0,2).reshape(n,-1),static,*neighbor,last,age,original[:,4],original[:,7]]).astype('float32')
def load_model():
    info=json.loads((HERE/'state-bridge-manifest.json').read_text());path=HERE/info['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256'];return lgb.Booster(model_file=str(path))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sample',required=True,type=Path);ap.add_argument('--network',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);args=ap.parse_args()
    with np.load(args.sample,allow_pickle=False) as z:x=z['x']
    with np.load(args.network,allow_pickle=False) as z:f=features(x,z['static'],z['up'],z['down'])
    ratio=load_model().predict(f);args.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.output,estimated_speed_to_free_speed=ratio);print(json.dumps(dict(output=str(args.output),nodes=len(ratio),queue_forecast=False,official_score=None)))
if __name__=='__main__':main()
