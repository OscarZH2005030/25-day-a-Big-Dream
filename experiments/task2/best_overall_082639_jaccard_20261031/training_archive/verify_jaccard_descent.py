"""Check actual train-loss decrease, independently of the GBDT curvature proxy."""
import json
import numpy as np
import lightgbm as lgb
import train_ongoing_jaccard as t
from ongoing_jaccard_core import loss_gradient

def main():
    panel='D7_I10_E';seed=t.SEEDS[0];source=t.old.RUN/'fit'/panel;meta=t.read(source/'complete.json');base=np.load(source/'X.npy',mmap_mode='r');extra=np.load(t.speed.RUN/'fit'/panel/'extra.npy',mmap_mode='r');y0=np.load(source/'y.npy');m=np.load(source/'m.npy');month=np.load(source/'month.npy');row,hs=np.nonzero(m&(month<203010)[:,None]);x=np.column_stack([base[row],extra[row],(hs+1)*5]).astype('float32');y=y0[row,hs];nodew=np.load(source/'w.npy');q=np.zeros(len(base));pos=np.zeros(len(base),bool);groups=np.zeros(len(base),dtype='int32')
    with np.load(t.BASE/'outputs/ongoing_analog_20261031/library'/f'{panel}.npz') as z:values={t.old.prior.old.normtime(o):v for o,v in zip(z['origin'],z['future'])}
    for i,w in enumerate(meta['windows']):
        v=values[t.old.prior.old.normtime(w['origin'])];known=np.isfinite(v);a,b=w['start'],w['end'];q[a:b]=((v<=0)&known).sum()/max(int(known.sum()),1);pos[a:b]=w['positive'];groups[a:b]=i
    pos=pos[row];group=groups[row];weight=nodew[row].astype(float)*np.where(pos,np.clip(.1/(q[row]+.01),.5,8.),1.)
    for label,mass in [(True,.8),(False,.2)]:
        use=pos==label;weight[use]*=mass/weight[use].sum()
    weight/=weight.mean();initial=t.BASE/'outputs/ongoing_macro_20261031'/f'development_inverse_{seed}'/f'{panel}.txt';before=lgb.Booster(model_file=str(initial)).predict(x,raw_score=True,num_threads=2);reports={}
    for cap in t.CAPACITY:
        strength=t.CAPACITY[cap]['strength'];after=lgb.Booster(model_file=str(t.OUT/f'development_{cap}_{seed}'/f'{panel}.txt')).predict(x,raw_score=True,num_threads=2);a=loss_gradient(before,y,weight,group,strength)[0]/len(y);b=loss_gradient(after,y,weight,group,strength)[0]/len(y);assert b<a,(cap,a,b);reports[cap]=dict(objective_before=a,objective_after=b,decreased=True,rows=len(y))
    t.write(t.OUT/'training_descent_check.json',dict(panel=panel,phase='June-September fitting',seed=seed,heads=reports));print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
