"""Three-state not-arrived/queued/post-arrival model and conditional paths."""
import argparse,json,gc
from collections import Counter
from pathlib import Path
import numpy as np
import lightgbm as lgb
from core import ROOT,r,softmax,partial_loss,Context,verifier_features
from run import Generator
from joint import joint_features

BASE=Path(__file__).resolve().parent;HERE=BASE/'onset_arrival';HERE.mkdir(exist_ok=True)
BITS=((np.arange(64)[:,None]>>np.arange(6))&1).astype(int)
PHASE=np.where(BITS==1,1,np.where(np.cumsum(BITS,axis=1)>0,2,0));FIRST=np.where(BITS.any(1),np.argmax(BITS,axis=1),6)

def phase_allowed(y,m):
    y=np.asarray(y,bool);m=np.asarray(m,bool);a=np.ones((*y.shape,3),bool)
    for h in range(6):
        known=m[:,h];positive=y[:,h]&known;negative=~y[:,h]&known;earlier=(y[:,:h]&m[:,:h]).any(1);prefix=m[:,:h].all(1)
        a[positive,h]=[0,1,0];a[negative&earlier,h]=[0,0,1];a[negative&~earlier&prefix,h]=[1,0,0];a[negative&~earlier&~prefix,h]=[1,0,1]
    return a

def fit(regime):
    dest=HERE/f'{regime}-phase.txt'
    if dest.exists():return lgb.Booster(model_file=str(dest))
    x=np.load(BASE/f'{regime}X.npy',mmap_mode='r');f=np.load(BASE/f'{regime}F.npy',mmap_mode='r');y=np.load(BASE/'y.npy');m=np.load(BASE/'m.npy');w=np.load(BASE/'wc.npy');keep=np.load(BASE/'early.npy') if regime=='early' else np.ones(len(y),bool);keep&=(y&m).any(1);ri,hi=np.nonzero(m&keep[:,None]);xx=np.column_stack([x[ri],np.eye(6,dtype='float32')[hi],f[ri,hi]]);a=phase_allowed(y,m)[ri,hi];weights=w[ri]/m.sum(1)[ri];weights/=weights.mean();state={}
    def objective(raw,ds):
        loss,g,h=partial_loss(raw,a,weights);state['loss']=loss;return g.astype('float32'),h.astype('float32')
    params=dict(objective=objective,num_class=3,num_leaves=15,learning_rate=.04,min_data_in_leaf=40,lambda_l2=5,feature_fraction=.9,num_threads=4,verbosity=-1,seed=20261029,deterministic=True,force_col_wise=True,max_bin=63)
    model=lgb.train(params,lgb.Dataset(xx,label=np.zeros(len(xx))),num_boost_round=160);model.save_model(str(dest));r.write(dest.with_suffix('.json'),dict(rows=len(xx),features=xx.shape[1],phases=['not_arrived','queued','post_arrival_clear'],unknown_prefix_zero='ambiguous not-arrived or post-arrival',rounds_per_class=160,total_trees=model.num_trees(),conditional_on_observed_event=True,no_class_weight_boost=True,threshold=.5,loss=state['loss']));print(json.dumps(dict(stage='onset_phase_fit',regime=regime,rows=len(xx),loss=state['loss'])),flush=True);return model

def decode(prob,ep):
    """Conditional nonempty paths weighted by first-arrival mass; recurrence allowed."""
    n=len(prob);logp=np.log(np.maximum(prob,1e-12));score=np.zeros((n,63))
    for h in range(6):score+=logp[:,h,PHASE[1:,h]]
    event=np.maximum(1-ep[:,0],1e-12);first=ep[:,1:]/event[:,None];count=2.**(5-FIRST[1:]);score+=np.log(np.maximum(first[:,FIRST[1:]],1e-12))-np.log(count)[None]
    post=softmax(score);conditional=post@BITS[1:];firstpost=post@np.eye(6)[FIRST[1:]];before=np.stack([(post*(PHASE[1:]==0)[:,h]).sum(1) for h in range(6)],axis=1);after=np.stack([(post*(PHASE[1:]==2)[:,h]).sum(1) for h in range(6)],axis=1)
    return conditional.astype('float32'),(conditional*event[:,None]).astype('float32'),firstpost.astype('float32'),before.astype('float32'),after.astype('float32')

def reshape_parent(b,q,use):
    result=b.copy();peak=b.max(0);ids=np.flatnonzero(use&(peak>=.5)&(q.max(0)>0));shape=q[:,ids]/np.maximum(q[:,ids].max(0),1e-20);result[:,ids]=shape*peak[ids];result[np.argmax(shape,axis=0),ids]=peak[ids];return result

def generate(args,backend,regime,pop):
    dest=HERE/f'{regime}-{pop}.npz'
    if dest.exists():return r.read(dest.with_suffix('.json')),dict(np.load(dest))
    ctx=Context(args,backend,ROOT/'work/task2-rescue-v2/profiles',r.read,r.write,lambda *a,**kw:None);gen=Generator(regime,ctx,backend[2]);phase=fit(regime);meta=r.read(BASE/f'{regime}-{pop}.json');old=np.load(BASE/f'{regime}-{pop}.npz');store={}
    for i,e in enumerate(meta):
        b=old[f'{i}_base'];store[f'{i}_base']=b
        for k in ['y','m']:store[f'{i}_{k}']=old[f'{i}_{k}']
        if e['condition']!=0:continue
        slot=int(e['origin'][11:13])*12+int(e['origin'][14:16])//5;s=backend[1](dict(e,origin_slot=slot));x=backend[2](s);extra,future=ctx.extra(s,regime);xx=np.column_stack([x,extra]);ep=old[f'{i}_event'];pool=old[f'{i}_links'];ids=np.flatnonzero(pool);probs=np.zeros((len(x),6,3),dtype='float32');probs[:,:,0]=1
        if len(ids):
            matrix=np.column_stack([np.tile(xx[ids],(6,1)),np.repeat(np.eye(6,dtype='float32'),len(ids),axis=0),future[ids].transpose(1,0,2).reshape(6*len(ids),10)]);p=softmax(phase.predict(matrix,raw_score=True,num_threads=4)).reshape(6,len(ids),3).transpose(1,0,2);probs[ids]=p
        conditional,joint,first,before,after=decode(probs,ep);conditional=conditional.T;joint=joint.T;conditional[:,~pool]=0;joint[:,~pool]=0;use=backend[-1](s);shaped=reshape_parent(b,conditional,use)
        # Whole-link timing proposals retain the parent's event peak exactly.
        ids=np.flatnonzero(((shaped>=.5)!=(b>=.5)).any(0));vf,_=verifier_features(s,ctx,ep,conditional,b,regime);ph,_,_=joint_features(s);features=[];ri=[]
        for link in ids:
            oldbits=b[:,link]>=.5;newbits=shaped[:,link]>=.5;firstold=np.flatnonzero(oldbits);firstnew=np.flatnonzero(newbits);summary=np.r_[vf[link].mean(0),vf[link,:,103:].min(0),vf[link,:,103:].max(0),ph[link].mean(0),b[:,link],conditional[:,link],joint[:,link],first[link],before[link],after[link],oldbits,newbits,firstold[0]/5,firstnew[0]/5,oldbits.mean(),newbits.mean()];features.append(summary);ri.append(link)
        width=122+38+ph.shape[2]+48+4
        for k,v in dict(conditional=conditional,joint=joint,shaped=shaped,ri=np.array(ri,int),Z=np.array(features,dtype='float32').reshape(-1,width),unseen=use,first=first,before=before,after=after).items():store[f'{i}_{k}']=v
        if (i+1)%256==0:print(json.dumps(dict(stage='arrival_phase_predictions',regime=regime,pop=pop,done=i+1)),flush=True)
    np.savez_compressed(dest,**store);r.write(dest.with_suffix('.json'),meta);return meta,store

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();backend=r.setup(args)
    for regime,pops in [('early',['october','november']),('full',['december','january','february_new','february_old'])]:
        fit(regime)
        for pop in pops:generate(args,backend,regime,pop)

if __name__=='__main__':main()
