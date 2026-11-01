"""Time-out trained three-stage rescue; every inference threshold is fixed .5."""
import argparse,copy,gc,json,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import lightgbm as lgb
from core import ROOT,r,Context,first_allowed,softmax,partial_loss,verifier_features
HERE=Path(__file__).resolve().parent;BASE=ROOT/'work/task2-research';GATE=ROOT/'work/task2-gate';V2=ROOT/'work/task2-rescue-v2'

def progress(stage,**kw):
    d=dict(stage=stage,time=time.time(),**kw);r.write(HERE/'progress.json',d);print(json.dumps(d),flush=True)

def prepare(args,backend,ctx):
    if (HERE/'matrix.json').exists():return
    entries=r.read(BASE/'selected-training-windows.json');counts=Counter((e['panel'],e['origin'][:7]) for e in entries);rng=np.random.default_rng(20261029);chunks=defaultdict(list);windows=[];cursor=0;checks=[]
    _,sample,features,*_=backend
    for i,e in enumerate(entries):
        s=sample(e);y=s['y'].T.astype(bool);m=s['m'].T.astype(bool);valid=m.any(1);positive=(y&m).any(1);pos=np.flatnonzero(valid&positive);neg=np.flatnonzero(valid&~positive);kn=min(32,len(neg));ids=np.r_[pos,rng.choice(neg,kn,replace=False) if kn else np.array([],int)];inv=np.r_[np.ones(len(pos)),np.full(kn,len(neg)/kn if kn else 0)]
        mass=1/counts[(e['panel'],e['origin'][:7])];we=inv/max(int(valid.sum()),1)*mass;wc=np.zeros(len(ids));wc[:len(pos)]=mass/max(len(pos),1)
        x=features(s)
        for regime in ['early','full']:
            extra,future=ctx.extra(s,regime);chunks[regime+'X'].append(np.column_stack([x,extra])[ids]);chunks[regime+'F'].append(future[ids])
        for k,v in dict(y=y[ids],m=m[ids],we=we.astype('float32'),wc=wc.astype('float32'),early=np.full(len(ids),e['origin'][:10]<='2030-09-30')).items():chunks[k].append(v)
        windows.append(dict(panel=e['panel'],origin=e['origin'],start=cursor,end=cursor+len(ids),positive_views=len(pos),sampled_negative_views=kn,eligible_link_views=int(valid.sum())));cursor+=len(ids)
        if e['panel'] not in [a['panel'] for a in checks]:
            a,b=ctx.extra(s,'full');ss=copy.deepcopy(s);ss['y']=1-ss['y'];ss['m']=~ss['m'];ss['has_future_queue']=not ss['has_future_queue'];c,d=ctx.extra(ss,'full');assert np.array_equal(a,c) and np.array_equal(b,d)
            checks.append(dict(panel=e['panel'],target_fields_ignored_by_generator_features=True,training_profiles_leave_whole_day_out=True))
        if (i+1)%128==0:progress('cascade_training_data',windows=i+1,total=len(entries))
    for k,v in chunks.items():np.save(HERE/(k+'.npy'),np.concatenate(v))
    r.write(HERE/'matrix.json',dict(link_views=cursor,windows=windows,weights='inverse link inclusion and equal window mass per road/month only; NO label-class weighting',checks=checks))

def train(args,regime):
    X=np.load(HERE/(regime+'X.npy'),mmap_mode='r');F=np.load(HERE/(regime+'F.npy'),mmap_mode='r');Y=np.load(HERE/'y.npy');M=np.load(HERE/'m.npy');WE=np.load(HERE/'we.npy').astype(float);WC=np.load(HERE/'wc.npy').astype(float);keep=np.load(HERE/'early.npy') if regime=='early' else np.ones(len(X),bool)
    ep=HERE/(regime+'-event.txt')
    params=dict(num_leaves=15,learning_rate=.04,min_data_in_leaf=40,lambda_l2=5,feature_fraction=.9,num_threads=4,verbosity=-1,seed=20261029,deterministic=True,force_col_wise=True,max_bin=63)
    if not ep.exists():
        xx=np.column_stack([X[keep],F[keep].reshape(int(keep.sum()),60)]);a=first_allowed(Y[keep],M[keep]);w=WE[keep];w/=w.mean();info=~a.all(1);xx=xx[info];a=a[info];w=w[info];state={}
        def objective(raw,ds):
            loss,g,h=partial_loss(raw,a,w);state['loss']=loss;return g.astype('float32'),h.astype('float32')
        def callback(env):
            if (env.iteration+1)%64==0:progress('event_first_arrival_fit',regime=regime,round=env.iteration+1,total=256,loss=state.get('loss'))
        model=lgb.train(dict(params,objective=objective,num_class=7),lgb.Dataset(xx,label=np.zeros(len(xx))),num_boost_round=256,callbacks=[callback]);model.save_model(str(ep));r.write(ep.with_suffix('.json'),dict(rows=len(xx),classes=7,rounds=256,features=xx.shape[1],threshold=.5,loss='partial first-arrival likelihood; positive weights not inflated'))
        del xx,model;gc.collect()
    tp=HERE/(regime+'-temporal.txt')
    if not tp.exists():
        selected=keep&(Y&M).any(1);ri,hi=np.nonzero(M&selected[:,None]);xx=np.column_stack([X[ri],np.eye(6,dtype='float32')[hi],F[ri,hi]]);yy=Y[ri,hi].astype(float);ww=WC[ri]/M.sum(1)[ri];ww/=ww.mean()
        model=lgb.train(dict(params,objective='binary'),lgb.Dataset(xx,label=yy,weight=ww),num_boost_round=448);model.save_model(str(tp));r.write(tp.with_suffix('.json'),dict(event_link_views=int(selected.sum()),observed_frames=len(ri),rounds=448,features=xx.shape[1],threshold=.5,target='queue at horizon given observed-positive event; all observed zero frames retained'))
        progress('conditional_temporal_fit_done',regime=regime,rows=len(ri));del xx,model;gc.collect()

class Generator:
    def __init__(self,regime,ctx,features):
        self.regime=regime;self.ctx=ctx;self.features=features;self.event=lgb.Booster(model_file=str(HERE/(regime+'-event.txt')));self.temporal=lgb.Booster(model_file=str(HERE/(regime+'-temporal.txt')))
    def predict(self,s,b):
        x=self.features(s);extra,future=self.ctx.extra(s,self.regime);base=np.column_stack([x,extra]);ep=softmax(self.event.predict(np.column_stack([base,future.reshape(len(x),60)]),raw_score=True,num_threads=4)).astype('float32');links=((1-ep[:,0])>=.5)|(b>=.5).any(0);ids=np.flatnonzero(links);q=np.zeros_like(b)
        if len(ids):
            xx=np.column_stack([np.tile(base[ids],(6,1)),np.repeat(np.eye(6,dtype='float32'),len(ids),axis=0),future[ids].transpose(1,0,2).reshape(6*len(ids),10)])
            q[:,ids]=self.temporal.predict(xx,num_threads=4).reshape(6,-1)
        return ep,q,links

def generate(args,backend,ctx,regime,pop):
    dest=HERE/(regime+'-'+pop+'.npz');mp=dest.with_suffix('.json')
    if dest.exists():return r.read(mp),dict(np.load(dest))
    parent,sample,features,*_=backend;gen=Generator(regime,ctx,features)
    source={'october':'clean-2030-10','november':'clean-2030-11','december':'evaluation-december','january':'evaluation-january','february_new':'february-new_phase_population','february_old':'february-old410'}[pop]
    meta=r.read(GATE/(source+'.json'));old=np.load(GATE/(source+'.npz'));store={};schema=None
    for i,e in enumerate(meta):
        b=old[f'{i}_base'];store[f'{i}_base']=b
        for k in ['y','m']:store[f'{i}_{k}']=old[f'{i}_{k}']
        if e['condition']==0:
            ee=dict(e,origin_slot=int(e['origin'][11:13])*12+int(e['origin'][14:16])//5);s=sample(ee);ep,q,links=gen.predict(s,b);proposal=(q>=.5)&(b<.5);ri,hi=np.nonzero(proposal.T);vf,schema=verifier_features(s,ctx,ep,q,b,regime)
            for k,v in dict(event=ep,q=q,links=links,ri=ri,hi=hi,Z=vf[ri,hi]).items():store[f'{i}_{k}']=v
        if (i+1)%128==0:progress('cascade_time_out_candidates',regime=regime,population=pop,windows=i+1,total=len(meta))
    np.savez_compressed(dest,**store);r.write(mp,meta);r.write(HERE/'verifier-schema.json',schema);return meta,store

def fit_verifier(months,name):
    dest=HERE/(name+'.txt')
    if dest.exists():return lgb.Booster(model_file=str(dest))
    XX=[];YY=[];WW=[]
    for month in months:
        meta=r.read(HERE/('early-'+month+'.json'));counts=Counter(e['panel'] for e in meta)
        with np.load(HERE/('early-'+month+'.npz')) as z:
            for i,e in enumerate(meta):
                if e['condition']!=0:continue
                ri=z[f'{i}_ri'];hi=z[f'{i}_hi'];known=z[f'{i}_m'][hi,ri]
                if not known.any():continue
                XX.append(z[f'{i}_Z'][known]);YY.append(z[f'{i}_y'][hi[known],ri[known]]);WW.append(np.full(int(known.sum()),1/counts[e['panel']]/int(known.sum())))
    xx=np.concatenate(XX);yy=np.concatenate(YY).astype(float);ww=np.concatenate(WW);ww/=ww.mean()
    params=dict(objective='binary',num_leaves=7,learning_rate=.04,min_data_in_leaf=30,lambda_l2=10,feature_fraction=.9,num_threads=4,verbosity=-1,seed=20261029,deterministic=True,force_col_wise=True,max_bin=31)
    model=lgb.train(params,lgb.Dataset(xx,label=yy,weight=ww),num_boost_round=160);model.save_model(str(dest));r.write(dest.with_suffix('.json'),dict(source_months=months,candidate_frames=len(yy),true_queue_frames=int(yy.sum()),false_queue_frames=int((yy==0).sum()),features=xx.shape[1],threshold=.5,weights='equal candidate-window mass, no label-class weighting',generator_fit='June-September; both source months held out by generators'))
    progress('physical_pattern_verifier_fit',name=name,candidates=len(yy),TP=int(yy.sum()),FP=int((yy==0).sum()));return model

def metrics(meta,z,backend,stage,verifier=None):
    *_,iou,aggregate,unused=backend;rows=[];neg=defaultdict(list);TP=FP=FN=0;events=Counter()
    for i,e in enumerate(meta):
        b=z[f'{i}_base'];pred=b>=(.5 if e['condition']==0 else .25);y=z[f'{i}_y'].astype(bool);m=z[f'{i}_m']
        if e['condition']==0:
            t=(y&m).any(0);ep=(1-z[f'{i}_event'][:,0])>=.5;union=z[f'{i}_links'];parent=(b>=.5).any(0);complete=m.all(0)
            events['observed_positive_event_links']+=int(t.sum());events['event_head_recalled_links']+=int((ep&t).sum());events['parent_recalled_event_links']+=int((parent&t).sum());events['union_recalled_event_links']+=int((union&t).sum());events['new_event_links_beyond_parent']+=int((ep&t&~parent).sum());events['event_head_complete_negative_links']+=int((ep&~t&complete).sum());events['event_head_unknown_support_links']+=int((ep&~t&~complete).sum())
        if e['condition']==0 and stage!='base':
            links=z[f'{i}_links']
            if stage=='proposal':pred|=np.tile(links,(6,1))
            elif stage=='temporal':pred|=z[f'{i}_q']>=.5
            else:
                ri=z[f'{i}_ri'];hi=z[f'{i}_hi'];a=verifier.predict(z[f'{i}_Z'],num_threads=4)>=.5 if len(ri) else np.array([],bool);pred[hi[a],ri[a]]=True
        if not e['has_future_queue']:
            if e['condition']==0:neg[e['panel']].append(bool((pred&m).any()))
            continue
        row=dict(panel=e['panel'],condition=e['condition'],window_id=e['origin'],date=e['origin'][:10],iou=iou(y,pred,m));rows.append(row)
        if e['condition']==0:TP+=int((pred&y&m).sum());FP+=int((pred&~y&m).sum());FN+=int((~pred&y&m).sum())
    result=aggregate(rows);result['onset_macro']=float(np.mean([v for k,v in result['by_panel_condition'].items() if k.endswith('queue_onset')]))
    result.update(TP=TP,FP=FP,FN=FN,cell_recall=TP/max(TP+FN,1),cell_precision=TP/max(TP+FP,1),negative_false_any=float(np.mean([np.mean(v) for v in neg.values()])) if neg else None,event_diagnostics=dict(events),proposal_stage_is_candidate_coverage_not_a_submission=stage=='proposal')
    return result,rows

def pattern_report(args,months):
    schema=r.read(HERE/'verifier-schema.json');start=schema['physical_start'];extra_indices=[37,47,48,49,63,68,69,73,74,66,67];extra_names=['position_fraction','lane_drop','capacity_drop','isolated','weekend','historical_timeband_queue_rate','historical_global_queue_rate','demand_deviation','demand_ratio','AM_peak_intensity','PM_peak_intensity'];indices=list(range(start,start+len(schema['physical_names'])))+extra_indices;names=schema['physical_names']+extra_names;reports={}
    for month in months:
        true=[];false=[];tw=[];fw=[];groups=defaultdict(lambda:dict(TP=0,FP=0));meta=r.read(HERE/('early-'+month+'.json'))
        with np.load(HERE/('early-'+month+'.npz')) as z:
            for i,e in enumerate(meta):
                ri=z[f'{i}_ri'];hi=z[f'{i}_hi'];m=z[f'{i}_m'][hi,ri];y=z[f'{i}_y'][hi,ri]>0;full=z[f'{i}_Z'];X=full[:,indices]
                if (m&y).any():true.append(X[m&y]);tw.append(X[m&y].mean(0))
                if (m&~y).any():false.append(X[m&~y]);fw.append(X[m&~y].mean(0))
                for row,label in zip(full[m],y[m]):
                    zone=int(row[39:43].argmax());band=int(row[51:56].argmax());day='weekend' if row[63]>.5 else 'weekday';groups[f'{day}/band{band}/zone{zone}']['TP' if label else 'FP']+=1
        a=np.concatenate(true);b=np.concatenate(false);aw=np.array(tw);bw=np.array(fw);stats=[]
        for j,name in enumerate(names):
            scale=np.sqrt((aw[:,j].var()+bw[:,j].var())/2);stats.append(dict(feature=name,true_mean=float(a[:,j].mean()),false_mean=float(b[:,j].mean()),true_median=float(np.median(a[:,j])),false_median=float(np.median(b[:,j])),true_window_mean=float(aw[:,j].mean()),false_window_mean=float(bw[:,j].mean()),window_standardized_mean_difference=float((aw[:,j].mean()-bw[:,j].mean())/max(scale,1e-6))))
        reports[month]=dict(true_candidates=len(a),false_candidates=len(b),windows_with_true=len(aw),windows_with_false=len(bw),patterns=sorted(stats,key=lambda d:-abs(d['window_standardized_mean_difference'])),time_region_daytype_groups=dict(groups))
    r.write(args.output/'task2_cascade_true_false_patterns.json',dict(reports=reports,scope='time-out generator proposals; descriptive correlated cell differences, not causal proof'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','train','internal','evaluate']);ap.add_argument('--regime',choices=['early','full'])
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();backend=r.setup(args);ctx=Context(args,backend,V2/'profiles',r.read,r.write,progress)
    if args.action=='prepare':prepare(args,backend,ctx);return
    if args.action=='train':train(args,args.regime);return
    if args.action=='internal':
        for pop in ['october','november']:generate(args,backend,ctx,'early',pop)
        probe=fit_verifier(['october'],'verifier-probe');meta,z=generate(args,backend,ctx,'early','november');reports={k:metrics(meta,z,backend,k,probe)[0] for k in ['base','proposal','temporal','verified']};r.write(args.output/'task2_cascade_november_internal.json',dict(reports=reports,thresholds_fixed=.5,official_S_queue=None));pattern_report(args,['october','november']);fit_verifier(['october','november'],'verifier-full');r.write(HERE/'frozen-policy.json',dict(event=.5,temporal=.5,verifier=.5,no_class_weight_boost=True,no_later_month_selection=True));print(json.dumps(reports),flush=True);return
    verifier=lgb.Booster(model_file=str(HERE/'verifier-full.txt'));reports={}
    for pop in ['december','january','february_new','february_old']:
        meta,z=generate(args,backend,ctx,'full',pop);scores={};rows={}
        for stage in ['base','proposal','temporal','verified']:scores[stage],rows[stage]=metrics(meta,z,backend,stage,verifier)
        reports[pop]=dict(scores=scores,windows=rows);r.write(args.output/'task2_learned_cascade_external.json',dict(reports=reports,all_evaluations_reused_development=True,official_S_queue=None));print(json.dumps(dict(population=pop,scores={k:dict(onset=v['onset_macro'],overall=v['proxy_IoU'],recall=v['cell_recall'],precision=v['cell_precision'],FP=v['FP'],FN=v['FN'],negative=v['negative_false_any']) for k,v in scores.items()})),flush=True)
if __name__=='__main__':main()
