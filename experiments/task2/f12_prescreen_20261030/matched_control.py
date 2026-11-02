"""Matched-control pre-screen: fit Jun-Sep, hold out Oct-Nov, same params/rounds; arms: control (X555+position), +F1, +F1+F2."""
import sys,json
from pathlib import Path
import numpy as np,lightgbm as lgb
HERE=Path(r'C:\Users\ROG\Desktop\IEEE Big data 2026\task2_features\experiments\task2\r127_reproduction_20261027');TI=Path(r'C:\Users\ROG\tfb-r127-assets\TrainingInputs-R127');EX=Path(r'C:\Users\ROG\tfb-r127-work\f12_extra')
man=json.loads((TI/'manifest.json').read_text());P=next(m for m in man['models'] if m['file']=='D7_I10_E-position.txt')['params'];ROUNDS=int(sys.argv[2]) if len(sys.argv)>2 else 600
def wap(y,p,w):
    o=np.argsort(-p);y=y[o];w=w[o];tp=np.cumsum(w*y);fp=np.cumsum(w*(1-y));prec=tp/np.maximum(tp+fp,1e-12);return float((prec*w*y).sum()/max((w*y).sum(),1e-12))
def wll(y,p,w):
    p=np.clip(p,1e-6,1-1e-6);return float((w*-(y*np.log(p)+(1-y)*np.log(1-p))).sum()/w.sum())
def run(panel):
    rec=json.loads((HERE/'training/recipes'/(panel+'.json')).read_text());d=TI/'position'/panel
    X=np.load(d/'X555.npy');pos=np.load(d/'position.npy');y=np.load(d/'y.npy').astype(np.float64);w=np.load(d/'w.npy');F=np.load(EX/(panel+'_F.npy'))
    month=np.empty(len(y),'U7')
    for r in rec['windows']:month[r['start']:r['end']]=r['entry']['origin'][:7] if 'origin' in r['entry'] else r['origin'][:7]
    fit=np.isin(month,['2030-06','2030-07','2030-08','2030-09']);hold=np.isin(month,['2030-10','2030-11']);print(panel,'fit',fit.sum(),'hold',hold.sum(),'hold pos',y[hold].sum(),flush=True);out={}
    arms={'control':np.hstack([X,pos]),'F1':np.hstack([X,pos,F[:,:8]]),'F1F2':np.hstack([X,pos,F])}
    for name,A in arms.items():
        b=lgb.train(dict(P,num_threads=4),lgb.Dataset(A[fit],y[fit],weight=w[fit]),ROUNDS);p=b.predict(A[hold]);out[name]=dict(wLL=wll(y[hold],p,w[hold]),wAP=wap(y[hold],p,w[hold]));print(panel,name,out[name],flush=True)
    return out
res={p:run(p) for p in sys.argv[1].split(',')};Path(r'C:\Users\ROG\Desktop\IEEE Big data 2026\task2_features\f12\prescreen_'+sys.argv[1].replace(',','_')+'.json').write_text(json.dumps(res,indent=1))
