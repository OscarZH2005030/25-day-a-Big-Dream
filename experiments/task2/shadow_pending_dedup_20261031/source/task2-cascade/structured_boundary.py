"""Semantic arrival/clearance experts and an out-of-fold linear-chain CRF.

Only training labels enter transition fitting. Thresholds stay .25/.5/.5;
decoders are fixed posterior marginal .5 and joint MAP, with all 64 edit
patterns allowed. No monotonic queue or single-episode assumption.
"""
import argparse,json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import lightgbm as lgb
from core import r
import phase_experiment as phase
import ongoing_gate as gate
from history_aligned_phase import load_aligned,ORIGINAL
from supply_phase_experiment import HERE as SUPPLY
from boundary_context import directed_features

HERE=SUPPLY/'structured';HERE.mkdir(exist_ok=True)
BITS=((np.arange(64)[:,None]>>np.arange(6))&1).astype(int)

def extra(backend,pop,regime):
    dest=HERE/f'{regime}-{pop}-context.npz'
    if dest.exists():return dict(np.load(dest))
    meta,z,c=load_aligned(pop,regime);out={};names=None
    for i,e in enumerate(meta):
        if e['condition']!=1:continue
        ri=c[f'{i}_ri'];hi=c[f'{i}_hi']
        if len(ri):
            slot=int(e['origin'][11:13])*12+int(e['origin'][14:16])//5;s=backend[1](dict(e,origin_slot=slot));a,names=directed_features(s,z[f'{i}_base']);out[f'{i}_D']=a[ri,hi]
        else:out[f'{i}_D']=None
    for k,v in out.items():
        if v is None:out[k]=np.zeros((0,len(names)),dtype='float32')
    np.savez_compressed(dest,**out);r.write(HERE/'context-schema.json',dict(names=names,inputs='history/masked T, directed public graph, frozen parent forecast only'));return out

def collect(months,backend,regime='early'):
    xs=[];ys=[];knowns=[];groups=[];dates=[];weights=[];sequences=[];cursor=0
    for pop in months:
        meta,z,c=load_aligned(pop,regime);a=dict(np.load(SUPPLY/f'phase-{regime}-{pop}.npz'));d=extra(backend,pop,regime);counts=Counter(e['panel'] for e in meta)
        for i,e in enumerate(meta):
            if e['condition']!=1:continue
            rows=np.flatnonzero(a[f'{i}_phase']==1);ri=c[f'{i}_ri'][rows];hi=c[f'{i}_hi'][rows]
            if not len(rows):continue
            x=np.column_stack([phase.features(c,a,i)[rows],d[f'{i}_D'][rows]]);old=z[f'{i}_base'][hi,ri]>=.25;y=old!=z[f'{i}_y'][hi,ri].astype(bool);known=z[f'{i}_m'][hi,ri];role=(a[f'{i}_A'][rows,1]>.5).astype(int);group=4*role+2*(hi>=3)+old.astype(int);w=np.full(len(rows),1/counts[e['panel']]/max(known.sum(),1));xs.append(x);ys.append(y);knowns.append(known);groups.append(group);dates.extend([e['origin'][:10]]*len(rows));weights.append(w)
            links=np.unique(ri)
            for link in links:
                take=np.flatnonzero(ri==link);ids=np.full(6,-1,int);ids[hi[take]]=cursor+take;mask=ids>=0;target=np.zeros(6,bool);target[hi[take]]=y[take];obs=np.zeros(6,bool);obs[hi[take]]=known[take]
                if obs.any():sequences.append(dict(ids=ids,mask=mask,target=target,known=obs,base=z[f'{i}_base'][:,link]>=.25,role=int(role[take[0]]),weight=1/counts[e['panel']]/len(links),date=e['origin'][:10]))
            cursor+=len(rows)
    return dict(X=np.concatenate(xs),y=np.concatenate(ys),known=np.concatenate(knowns),group=np.concatenate(groups),date=np.array(dates),w=np.concatenate(weights),sequences=sequences)

def fit_experts(data,mask,name=None):
    x=data['X'];y=data['y'];w=data['w'];g=data['group'];known=data['known'];models={};counts=[]
    masks=[('fallback',mask&known)]+[(str(j),mask&known&(g==j)) for j in range(8)]
    for key,take in masks:
        if take.sum()<60 or len(np.unique(y[take]))<2:continue
        params=dict(objective='binary',num_leaves=7,learning_rate=.04,min_data_in_leaf=20,lambda_l2=15,feature_fraction=.9,num_threads=4,verbosity=-1,seed=20261029,deterministic=True,force_col_wise=True,max_bin=31)
        ww=w[take]/w[take].mean();model=lgb.train(params,lgb.Dataset(x[take],label=y[take],weight=ww),num_boost_round=160);models[key]=model;counts.append(dict(expert=key,rows=int(take.sum())))
        if name:model.save_model(str(HERE/f'{name}-{key}.txt'))
    if name:r.write(HERE/f'{name}-experts.json',dict(experts=counts,features=x.shape[1],threshold=.5,role='latest valid queued vs currently clear propagation boundary',no_class_weights=True))
    return models

def predict_raw(models,x,g):
    out=np.full(len(x),-30.,float)
    for group in np.unique(g):
        take=g==group;model=models.get(str(group),models.get('fallback'))
        if model is not None:out[take]=model.predict(x[take],raw_score=True,num_threads=4)
    return out

def chain_arrays(sequence,raw):
    mask=np.asarray(sequence['mask']);ids=np.asarray(sequence['ids']);unary=np.zeros(6);unary[mask]=raw[ids[mask]];allowed=(BITS[:,~mask]==0).all(1);known=np.asarray(sequence.get('known',np.zeros(6,bool)));target=np.asarray(sequence.get('target',np.zeros(6,bool)));compatible=(BITS[:,known]==target[known]).all(1)&allowed
    # Pairwise edit potentials depend on inherited parent states and observed
    # local queue role. Their zero-edit baseline is subtracted identically.
    f=np.zeros((64,32),float);base=np.asarray(sequence['base'],int);role=int(sequence['role'])
    for t in range(5):
        group=role*16+(base[t]*2+base[t+1])*4;code=BITS[:,t]*2+BITS[:,t+1];f[np.arange(64),group+code]+=1;f[:,group]-=1
    return BITS@unary,f,allowed,compatible

def fit_chain(data,raw,name):
    import torch
    torch.set_num_threads(4);arrays=[chain_arrays(s,raw) for s in data['sequences']];unary=torch.tensor(np.array([a[0] for a in arrays]),dtype=torch.float64);f=torch.tensor(np.array([a[1] for a in arrays]),dtype=torch.float64);allowed=torch.tensor(np.array([a[2] for a in arrays]));compatible=torch.tensor(np.array([a[3] for a in arrays]));weights=torch.tensor([s['weight'] for s in data['sequences']],dtype=torch.float64);weights/=weights.sum();theta=torch.zeros(32,dtype=torch.float64,requires_grad=True);optimizer=torch.optim.LBFGS([theta],lr=1.,max_iter=100,line_search_fn='strong_wolfe');state={}
    def closure():
        optimizer.zero_grad();score=unary+f@theta;logz=torch.logsumexp(score.masked_fill(~allowed,-torch.inf),1);logtarget=torch.logsumexp(score.masked_fill(~compatible,-torch.inf),1);loss=((logz-logtarget)*weights).sum()+.02*(theta**2).mean();loss.backward();state['loss']=float(loss.detach());return loss
    optimizer.step(closure);values=theta.detach().numpy();assert np.isfinite(values).all() and np.isfinite(state['loss']);r.write(HERE/f'{name}-chain.json',dict(theta=values.tolist(),parameters=32,sequences=len(arrays),loss=state['loss'],objective='partial edit-sequence NLL plus 0.02 mean squared transition potential',oof='3 folds by global date; all roads on held-out date excluded from unary-model training',threshold=.5,no_monotonic_or_single_episode_constraint=True));return values

def train(months,backend,name):
    data=collect(months,backend);mapping={d:i%3 for i,d in enumerate(sorted(set(data['date'])))};fold=np.array([mapping[d] for d in data['date']]);oof=np.zeros(len(data['X']));checks=[]
    for k in range(3):
        tr=fold!=k;te=~tr;models=fit_experts(data,tr);oof[te]=predict_raw(models,data['X'][te],data['group'][te]);assert not set(data['date'][tr])&set(data['date'][te]);checks.append(dict(fold=k,heldout_dates=sorted(set(data['date'][te]))))
    theta=fit_chain(data,oof,name);models=fit_experts(data,np.ones(len(oof),bool),name);r.write(HERE/f'{name}-training.json',dict(months=months,rows=len(oof),known_rows=int(data['known'].sum()),date_folds=checks,parent_threshold=.25,proposal_threshold=.5,verifier_threshold=.5,global_date_splits=True));return models,theta

def decode(sequence,raw,theta,mode):
    unary,f,allowed,_=chain_arrays(sequence,raw);score=unary+f@theta;score[~allowed]=-np.inf
    if mode=='map':return BITS[np.argmax(score)].astype(bool)
    p=np.exp(score-np.max(score));p/=p.sum();return (p@BITS)>=.5

def evaluate(backend,pop,regime,models,theta,variant):
    meta,z,c=load_aligned(pop,regime);a=dict(np.load(SUPPLY/f'phase-{regime}-{pop}.npz'));d=dict(np.load(HERE/f'{regime}-{pop}-context.npz'));cc=dict(c)
    class Identity:
        def predict(self,x,**kw):return x[:,0]
    for i,e in enumerate(meta):
        if e['condition']!=1:continue
        rows=np.flatnonzero(a[f'{i}_phase']==1);ri=c[f'{i}_ri'][rows];hi=c[f'{i}_hi'][rows];accept=np.zeros(len(c[f'{i}_ri']),bool)
        if len(rows) and variant!='base':
            x=np.column_stack([phase.features(c,a,i)[rows],d[f'{i}_D'][rows]]);old=z[f'{i}_base'][hi,ri]>=.25;role=(a[f'{i}_A'][rows,1]>.5).astype(int);g=4*role+2*(hi>=3)+old.astype(int);raw=predict_raw(models,x,g)
            if variant=='semantic_pointwise':accept[rows]=raw>=0
            else:
                for link in np.unique(ri):
                    take=np.flatnonzero(ri==link);ids=np.full(6,-1,int);ids[hi[take]]=take;seq=dict(ids=ids,mask=ids>=0,base=z[f'{i}_base'][:,link]>=.25,role=int(role[take[0]]));decision=decode(seq,raw,theta,'map' if variant=='structured_map' else 'marginal');accept[rows[take]]=decision[hi[take]]
        cc[f'{i}_Z']=accept[:,None].astype(float)
    return gate.metrics(meta,z,cc,backend,{'all':Identity()},'base' if variant=='base' else 'shared')

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();backend=r.setup(args)
    for pop in ['october','november']:extra(backend,pop,'early')
    models,theta=train(['october'],backend,'probe');variants=['base','semantic_pointwise','structured_marginal','structured_map'];internal={k:evaluate(backend,'november','early',models,theta,k)[0] for k in variants};b=internal['base'];eligible=[k for k in variants[1:] if internal[k]['ongoing_macro']>b['ongoing_macro'] and internal[k]['FP']<=b['FP'] and internal[k]['TP']>=b['TP']];selected=max(eligible,key=lambda k:internal[k]['ongoing_macro']) if eligible else 'base';r.write(args.output/'task2_structured_boundary_internal.json',dict(scores=internal,selected=selected,later_month_selection=False,design_informed_by_reused_cases=True,inherited_routing_selected_later=True,internal_not_blind=True));r.write(HERE/'frozen-policy.json',dict(selected=selected,later_month_selection=False,thresholds=dict(parent=.25,proposal=.5,verifier=.5)));models,theta=train(['october','november'],backend,'full');reports={}
    for pop in ['december','january','february_new','february_old']:
        extra(backend,pop,'full');reports[pop]={}
        for k in variants:
            scores,rows=evaluate(backend,pop,'full',models,theta,k);reports[pop][k]=dict(scores=scores,windows=rows)
        r.write(args.output/'task2_structured_boundary_external.json',dict(reports=reports,frozen_selection=selected,all_evaluations_reused_development=True,official_truth_available=False));print(json.dumps(dict(population=pop,selected=selected,scores={k:{j:v['scores'][j] for j in ['ongoing_macro','TP','FP','FN','known_changed_cells']} for k,v in reports[pop].items()})),flush=True)

if __name__=='__main__':main()
