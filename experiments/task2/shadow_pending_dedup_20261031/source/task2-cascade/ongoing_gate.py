"""Time-out trained verifier of proposed ongoing corrections."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
import lightgbm as lgb
from core import ROOT,r,Context
import ongoing as o

HERE=o.HERE

def cache(args,backend,ctx,regime,pop):
    dest=HERE/f'gate-{regime}-{pop}.npz'
    meta=r.read(HERE/f'{regime}-{pop}.json');z=dict(np.load(HERE/f'{regime}-{pop}.npz'))
    if dest.exists():return meta,z,dict(np.load(dest))
    out={}
    for i,e in enumerate(meta):
        if e['condition']!=1:continue
        b=z[f'{i}_base'];new=z[f'{i}_binary'];proposal=((b>=.25)!=(new>=.5))&z[f'{i}_dynamic'][None];ri,hi=np.nonzero(proposal.T)
        if len(ri):
            slot=int(e['origin'][11:13])*12+int(e['origin'][14:16])//5;s=backend[1](dict(e,origin_slot=slot));a,_,_=o.features(s,ctx,regime);prev=np.vstack([b[:1],b[:-1]]);nxt=np.vstack([b[1:],b[-1:]]);last=np.column_stack([b[hi,ri],new[hi,ri],z[f'{i}_speed'][hi,ri],prev[hi,ri],nxt[hi,ri],b.max(0)[ri],b.mean(0)[ri],(b>=.25).mean(0)[ri],new.max(0)[ri],new.mean(0)[ri],(b[hi,ri]>=.25)])
            x=np.column_stack([a[ri,hi],last])
        else:x=np.zeros((0,309),dtype='float32')
        for k,v in dict(ri=ri,hi=hi,Z=x).items():out[f'{i}_{k}']=v
        if (i+1)%256==0:print(json.dumps(dict(phase='ongoing_correction_candidates',pop=pop,done=i+1,total=len(meta))),flush=True)
    width=next(v.shape[1] for k,v in out.items() if k.endswith('_Z') and len(v))
    for k,v in out.items():
        if k.endswith('_Z') and not len(v):out[k]=np.zeros((0,width),dtype='float32')
    np.savez_compressed(dest,**out);return meta,z,out

def fit(months,name):
    xs=[];ys=[];ws=[];directions=[]
    for pop in months:
        meta=r.read(HERE/f'early-{pop}.json');z=np.load(HERE/f'early-{pop}.npz');c=np.load(HERE/f'gate-early-{pop}.npz');counts=Counter(e['panel'] for e in meta)
        for i,e in enumerate(meta):
            ri=c[f'{i}_ri'];hi=c[f'{i}_hi'];m=z[f'{i}_m'][hi,ri]
            if not m.any():continue
            old=z[f'{i}_base'][hi,ri]>=.25;y=old!=z[f'{i}_y'][hi,ri].astype(bool);xs.append(c[f'{i}_Z'][m]);ys.append(y[m]);directions.append(old[m]);ws.append(np.full(m.sum(),1/counts[e['panel']]/m.sum()))
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);w/=w.mean();direction=np.concatenate(directions);models={}
    for kind,mask in [('all',np.ones(len(x),bool)),('clear',direction),('revive',~direction)]:
        params=dict(objective='binary',num_leaves=7,learning_rate=.04,min_data_in_leaf=30,lambda_l2=10,feature_fraction=.9,num_threads=4,verbosity=-1,seed=20261029,deterministic=True,force_col_wise=True,max_bin=31)
        model=lgb.train(params,lgb.Dataset(x[mask],label=y[mask],weight=w[mask]),num_boost_round=160);model.save_model(str(HERE/f'{name}-{kind}.txt'));models[kind]=model
    r.write(HERE/f'{name}.json',dict(months=months,rows=len(y),correctable=int(y.sum()),incorrect_proposals=int((~y).sum()),features=x.shape[1],parent_threshold=.25,proposal_threshold=.5,verifier_threshold=.5,teachers_fit='June-September; October/November held out by both proposal models',class_weight_boost=False))
    return models

def metrics(meta,z,c,backend,models,variant):
    rows=[];TP=FP=FN=0;changes=0
    for i,e in enumerate(meta):
        if e['condition']!=1 or not e['has_future_queue']:continue
        old=z[f'{i}_base']>=.25;p=old.copy();y=z[f'{i}_y'].astype(bool);m=z[f'{i}_m'];ri=c[f'{i}_ri'];hi=c[f'{i}_hi'];x=c[f'{i}_Z']
        if len(ri) and variant!='base':
            if variant=='shared':accept=models['all'].predict(x,num_threads=4)>=.5
            else:
                clear=old[hi,ri];accept=np.zeros(len(ri),bool)
                for kind,take in [('clear',clear),('revive',~clear)]:
                    if take.any():accept[take]=models[kind].predict(x[take],num_threads=4)>=.5
            p[hi[accept],ri[accept]]=~old[hi[accept],ri[accept]];changes+=int((accept&m[hi,ri]).sum())
        rows.append(dict(panel=e['panel'],condition=1,window_id=e['origin'],date=e['origin'][:10],iou=backend[4](y,p,m)));TP+=int((p&y&m).sum());FP+=int((p&~y&m).sum());FN+=int((~p&y&m).sum())
    score=backend[5](rows);score.update(ongoing_macro=float(np.mean(list(score['by_panel_condition'].values()))),TP=TP,FP=FP,FN=FN,known_changed_cells=changes);return score,rows

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();backend=r.setup(args);ctx=Context(args,backend,o.V2/'profiles',r.read,r.write,lambda *a,**kw:None)
    for pop in ['october','november']:cache(args,backend,ctx,'early',pop)
    models=fit(['october'],'gate-probe');meta,z,c=cache(args,backend,ctx,'early','november');internal={k:metrics(meta,z,c,backend,models,k)[0] for k in ['base','shared','separate']};b=internal['base'];eligible=[k for k in ['shared','separate'] if internal[k]['ongoing_macro']>b['ongoing_macro'] and internal[k]['FP']<=b['FP'] and internal[k]['TP']>=.995*b['TP']];selected=max(eligible,key=lambda k:internal[k]['ongoing_macro']) if eligible else 'base'
    r.write(args.output/'task2_ongoing_gate_internal.json',dict(scores=internal,selected=selected,later_month_selection=False,thresholds=dict(parent=.25,proposal=.5,verifier=.5),parent='June-September R10 surrogate'))
    r.write(HERE/'gate-frozen-policy.json',dict(selected=selected,thresholds=dict(parent=.25,proposal=.5,verifier=.5),later_month_selection=False));models=fit(['october','november'],'gate-full');reports={}
    for pop in ['december','january','february_new','february_old']:
        meta,z,c=cache(args,backend,ctx,'full',pop);reports[pop]={}
        for k in ['base','shared','separate']:
            score,rows=metrics(meta,z,c,backend,models,k);reports[pop][k]=dict(scores=score,windows=rows)
        r.write(args.output/'task2_ongoing_gate_external.json',dict(reports=reports,frozen_selection=selected,all_evaluations_reused_development=True,official_truth_available=False));print(json.dumps(dict(population=pop,selected=selected,scores={k:{j:v['scores'][j] for j in ['ongoing_macro','TP','FP','FN','known_changed_cells']} for k,v in reports[pop].items()})),flush=True)

if __name__=='__main__':main()
