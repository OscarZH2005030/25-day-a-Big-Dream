"""Causal, utility-weighted rescue gate with rolling-origin detector predictions."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import argparse,copy,json,sys,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import lightgbm as lgb
HERE=Path(__file__).resolve().parent
RESEARCH=HERE.parent/'task2-research';sys.path.insert(0,str(RESEARCH))
import research as r
from summarize import paired_ci
SEED=20261029
GATE_FEATURES=['latest_speed_ratio','carried_speed_Tminus15','speed_trend_15min','queue_speed_margin',
 'history_min_speed','history_mean_speed','history_speed_coverage','latest_speed_quality',
 'history_near_queue_fraction','T_speed_missing','T_observation_age','T_quality','history_mean_flow',
 'history_mean_occupancy','T_flow','T_occupancy','T_flow_missing','T_occupancy_missing','sin_time','cos_time',
 'detector_peak','detector_mean','detector_first','detector_last','detector_peak_horizon','detector_positive_fraction',
 'parent_peak','parent_mean','parent_first','parent_last','peak_difference','window_detector_event_fraction',
 'window_parent_event_fraction','window_detector_max','window_parent_max','window_mean_speed','window_low_speed',
 'window_T_coverage','length_km','lanes_div8','free_speed_div130','capacity_div12000','threshold_ratio','isolated']

def progress(stage,**kw):
    d=dict(stage=stage,utc_epoch=time.time(),**kw);r.write(HERE/'progress.json',d);print(json.dumps(d),flush=True)

def gate_features(s,b,q):
    """Only history/static and predictions; target/quality-future fields ignored."""
    x=s['x'];n=x.shape[1];valid=x[:,:,4]<.5;speed=x[:,:,0];cut=s['network']['threshold']/s['network']['free_speed']
    latest=np.ones(n);old15=np.ones(n);latest_quality=np.zeros(n)
    for t in range(13):
        latest[valid[t]]=speed[t,valid[t]];latest_quality[valid[t]]=x[t,valid[t],3]
        if t<=9:old15[valid[t]]=speed[t,valid[t]]
    count=valid.sum(0);mn=np.min(np.where(valid,speed,np.inf),axis=0);mn[count==0]=1
    mean=np.divide(np.where(valid,speed,0).sum(0),count,out=np.ones(n),where=count>0)
    vflow=x[:,:,5]<.5;vocc=x[:,:,6]<.5
    flow=np.divide(np.where(vflow,x[:,:,1],0).sum(0),vflow.sum(0),out=np.zeros(n),where=vflow.sum(0)>0)
    occ=np.divide(np.where(vocc,x[:,:,2],0).sum(0),vocc.sum(0),out=np.zeros(n),where=vocc.sum(0)>0)
    qp=q.max(0);bp=b.max(0)
    values=[latest,old15,latest-old15,cut-latest,mn,mean,count/13,latest_quality,
            ((speed<=cut+.15)&valid).sum(0)/13,x[-1,:,4],x[-1,:,8],x[-1,:,3],
            flow,occ,x[-1,:,1],x[-1,:,2],x[-1,:,5],x[-1,:,6],x[-1,:,11],x[-1,:,12],
            qp,q.mean(0),q[0],q[-1],q.argmax(0)/5,(q>=.5).sum(0)/6,
            bp,b.mean(0),b[0],b[-1],qp-bp,
            np.full(n,(qp>=.5).mean()),np.full(n,(bp>=.5).mean()),
            np.full(n,qp.max()),np.full(n,bp.max()),np.full(n,np.mean(latest)),
            np.full(n,np.quantile(latest,.1)),np.full(n,valid[-1].mean())]
    out=np.column_stack(values+[s['network']['static'][:,j] for j in range(6)]).astype('float32')
    assert out.shape==(n,44) and np.isfinite(out).all()
    return out

def utility_targets(y,m,b,q,use):
    """Exact one-link marginal change in observed-cell window IoU."""
    truth=y.astype(bool)&m;old=(b>=.5)&m;inter=int((old&truth).sum());union=int((old|truth).sum());old_iou=inter/union if union else 1.
    proposal=(q>=.5)&m
    tp=(proposal&truth).sum(0);fp=(proposal&~truth).sum(0)
    if union:delta=(inter+tp)/(union+fp)-old_iou
    else:delta=np.where(fp>0,-1.,0.)
    informative=use&(proposal.sum(0)>0)&(np.abs(delta)>1e-12)
    return delta,informative

def prepare(args,backend):
    done=HERE/'row-roster.json'
    if done.exists():return
    _,sample,_,_,_,_,unseen=backend
    entries=r.read(RESEARCH/'selected-training-windows.json');orig_y=np.load(RESEARCH/'matrix/y.npy',mmap_mode='r');orig_m=np.load(RESEARCH/'matrix/m.npy',mmap_mode='r')
    rng=np.random.default_rng(r.SEED);rows=[];cursor=0;checks=[]
    for i,e in enumerate(entries):
        s=sample(e);y=s['y'].T.astype(bool);m=s['m'].T.astype(bool);eligible=unseen(s)&m.any(1);positive=(y&m).any(1)
        pos=np.flatnonzero(eligible&positive);neg=np.flatnonzero(eligible&~positive);kn=min(32,len(neg));ids=np.r_[pos,rng.choice(neg,kn,replace=False) if kn else np.array([],int)]
        end=cursor+len(ids);assert np.array_equal(orig_y[cursor:end],y[ids]) and np.array_equal(orig_m[cursor:end],m[ids])
        rows.append(dict(start=cursor,end=end,panel=e['panel'],origin=e['origin'],month=e['origin'][:7],has_future_queue=e['has_future_queue']));cursor=end
        if e['panel'] not in [c['panel'] for c in checks]:
            rng2=np.random.default_rng(0);b=rng2.random(s['y'].shape);q=rng2.random(s['y'].shape);a=gate_features(s,b,q)
            changed=copy.deepcopy(s);changed['y']=1-changed['y'];changed['m']=~changed['m'];changed['has_future_queue']=not changed['has_future_queue'];assert np.array_equal(a,gate_features(changed,b,q))
            values=copy.deepcopy(sample.load_day(e['panel'],e['origin'][:10]));slot=e['origin_slot']
            for key in ['speed','flow','occupancy','quality','origin_speed','origin_flow','origin_occupancy','origin_quality']:values[key][slot+1:]=9999
            values['eligible'][slot+1:]=False;future=sample.make_sample(values,s['network'],slot,e['panel']);assert np.array_equal(a,gate_features(future,b,q))
            checks.append(dict(panel=e['panel'],target_fields_invariant=True,future_mainline_invariant=True))
        if (i+1)%128==0:progress('reconstructing_row_roster',windows=i+1,total=len(entries))
    assert cursor==len(orig_y)
    r.write(done,dict(rows=cursor,windows=rows,causality_checks=checks))
    r.write(args.output/'task2_gate_causality.json',dict(passed=True,checks=checks,features=44))

def fit_detector(cutoff):
    path=HERE/('detector-through-'+cutoff+'.txt')
    if path.exists():return lgb.Booster(model_file=str(path))
    roster=r.read(HERE/'row-roster.json');keep=np.zeros(roster['rows'],bool)
    for e in roster['windows']:
        if e['month']<=cutoff:keep[e['start']:e['end']]=True
    X=np.load(RESEARCH/'matrix/X.npy',mmap_mode='r');y=np.load(RESEARCH/'matrix/y.npy');m=np.load(RESEARCH/'matrix/m.npy');w=np.load(RESEARCH/'matrix/w.npy').astype(float)
    ri,hi=np.nonzero(m&keep[:,None]);xx=np.column_stack([X[ri],np.eye(6,dtype='float32')[hi]]);yy=y[ri,hi].astype(float);ww=w[ri]/m.sum(1)[ri];ww/=ww.mean()
    params=dict(objective='binary',num_leaves=31,learning_rate=.05,min_data_in_leaf=40,lambda_l2=4,feature_fraction=.85,num_threads=4,verbosity=-1,seed=r.SEED,deterministic=True,force_col_wise=True,max_bin=63)
    def cb(env):
        if (env.iteration+1)%224==0:progress('crossfit_detector',cutoff=cutoff,step=env.iteration+1,total=896)
    model=lgb.train(params,lgb.Dataset(xx,label=yy,weight=ww),num_boost_round=896,callbacks=[cb]);model.save_model(str(path));return model

def predict_head(head,x):
    xx=np.column_stack([np.tile(x,(6,1)),np.repeat(np.eye(6,dtype='float32'),len(x),axis=0)])
    return head.predict(xx,num_threads=4).reshape(6,-1).astype('float32')

def crossfit(args,backend):
    parent,sample,features,_,_,_,unseen=backend;entries=r.read(RESEARCH/'selected-training-windows.json')
    for cutoff,month in [('2030-08','2030-09'),('2030-09','2030-10'),('2030-10','2030-11')]:
        dest=HERE/('oof-'+month+'.npz')
        if dest.exists():continue
        head=fit_detector(cutoff);es=[e for e in entries if e['origin'].startswith(month)];store={};meta=[]
        for i,e in enumerate(es):
            s=sample(e);b=parent.probability(s);q=predict_head(head,features(s));g=gate_features(s,b,q);use=unseen(s)&(b.max(0)<.5)&(q.max(0)>=.5)
            for k,v in dict(base=b,q=q,g=g,y=s['y'],m=s['m'],use=use).items():store[f'{i}_{k}']=v
            meta.append(dict(panel=e['panel'],origin=e['origin'],condition=0,has_future_queue=e['has_future_queue']))
            if (i+1)%64==0:progress('crossfit_predictions',month=month,windows=i+1,total=len(es))
        np.savez_compressed(dest,**store);r.write(dest.with_suffix('.json'),meta)

def train_gate(months,name,prefix='oof-',feature_indices=None):
    path=HERE/(name+'.txt')
    if path.exists():return lgb.Booster(model_file=str(path))
    X=[];Y=[];W=[];diagnostics=[]
    for month in months:
        meta=r.read(HERE/(prefix+month+'.json'));counts=Counter((e['panel'],e['has_future_queue']) for e in meta)
        with np.load(HERE/(prefix+month+'.npz')) as store:
            for i,e in enumerate(meta):
                use=store[f'{i}_use'];delta,keep=utility_targets(store[f'{i}_y'],store[f'{i}_m'],store[f'{i}_base'],store[f'{i}_q'],use)
                if not keep.any():continue
                mass=(.8 if e['has_future_queue'] else .2)/counts[(e['panel'],e['has_future_queue'])]
                X.append(store[f'{i}_g'][keep]);Y.append((delta[keep]>0).astype(float));W.append(np.abs(delta[keep])*mass/max(int(use.sum()),1))
                diagnostics.append(dict(panel=e['panel'],month=month,rows=int(keep.sum()),profitable=int((delta[keep]>0).sum()),window_positive=e['has_future_queue']))
    xx=np.concatenate(X);yy=np.concatenate(Y);ww=np.concatenate(W);ww/=ww.mean();assert len(np.unique(yy))==2
    indices=list(range(44)) if feature_indices is None else list(feature_indices);xx=xx[:,indices]
    params=dict(objective='binary',num_leaves=7,learning_rate=.04,min_data_in_leaf=12,lambda_l2=5,feature_fraction=1.,num_threads=4,verbosity=-1,seed=SEED,deterministic=True,force_col_wise=True,max_bin=31)
    names=[GATE_FEATURES[i] for i in indices]
    model=lgb.train(params,lgb.Dataset(xx,label=yy,weight=ww,feature_name=names),num_boost_round=120);model.save_model(str(path))
    r.write(HERE/(name+'-training.json'),dict(months=months,rows=len(yy),profitable=int(yy.sum()),unprofitable=int((yy==0).sum()),features=len(indices),feature_indices=indices,rounds=model.current_iteration(),windows=diagnostics,weighted_positive_fraction=float(np.average(yy,weights=ww)),target='positive exact one-link marginal proxy-IoU change; absolute gain weighted'))
    r.write(HERE/(name+'-feature-importance.json'),sorted([dict(feature=k,gain=float(v)) for k,v in zip(names,model.feature_importance(importance_type='gain'))],key=lambda v:-v['gain']))
    progress('gate_fit_complete',name=name,rows=len(yy),profitable=int(yy.sum()));return model

def evaluate_store(backend,meta,store,policy,gate=None):
    *_,iou,aggregate,unseen=backend;records=[];negative=defaultdict(list);rows=[];added=0;kept=0
    for i,e in enumerate(meta):
        b=store[f'{i}_base'];p=b.copy();use=store[f'{i}_use'].copy();q=store[f'{i}_q']
        if policy=='confidence':use &= q.max(0)>=.75
        if policy=='learned':use &= gate.predict(store[f'{i}_g'],num_threads=4)>=.5
        if policy!='base':p[:,use]=q[:,use]
        else:use[:]=False
        added+=int(store[f'{i}_use'].sum());kept+=int(use.sum());pred=p>=(.5 if e['condition']==0 else .25)
        y=store[f'{i}_y'];m=store[f'{i}_m']
        if not e['has_future_queue']:
            if e['condition']==0:negative[e['panel']].append(bool((pred&m).any()))
            continue
        v=iou(y,pred,m);rec=dict(panel=e['panel'],condition=e['condition'],window_id=e['origin'],iou=v,date=e['origin'][:10]);records.append(rec);rows.append(rec)
    report=aggregate(records);report['onset_macro']=float(np.mean([v for k,v in report['by_panel_condition'].items() if k.endswith('queue_onset')]))
    report['negative_false_any']=float(np.mean([np.mean(g) for g in negative.values()])) if negative else None
    report['eligible_rescue_links']=added;report['retained_rescue_links']=kept
    return report,rows

def select(args,backend):
    gate=train_gate(['2030-09','2030-10'],'gate-probe');meta=r.read(HERE/'oof-2030-11.json')
    with np.load(HERE/'oof-2030-11.npz') as z:store=dict(z)
    reports={};rows={}
    for policy in ['base','rescue','confidence','learned']:reports[policy],rows[policy]=evaluate_store(backend,meta,store,policy,gate)
    base=reports['base'];ungated_gain=reports['rescue']['onset_macro']-base['onset_macro'];minimum=max(.004,.5*ungated_gain)
    eligible=[p for p in ['confidence','learned'] if reports[p]['onset_macro']>=base['onset_macro']+minimum and reports[p]['negative_false_any']<=base['negative_false_any']+.03]
    selected=max(eligible,key=lambda p:reports[p]['onset_macro']) if eligible else 'base'
    r.write(args.output/'task2_gate_internal_selection.json',dict(reports=reports,selected=selected,minimum_onset_gain=minimum,legacy_parent_not_crossfitted=True,detector_temporally_crossfitted=True,month='November',official_S_queue=None))
    if selected=='learned':train_gate(['2030-09','2030-10','2030-11'],'gate-final')
    r.write(HERE/'frozen-selection.json',dict(policy=selected,threshold=.5,confidence_threshold=.75,selected_only_on='November rolling-origin detector predictions',no_December_January_selection=True))
    print(json.dumps(dict(selected=selected,reports={k:dict(onset=v['onset_macro'],negative=v['negative_false_any'],kept=v['retained_rescue_links']) for k,v in reports.items()})),flush=True)

def final_cache(args,backend,month):
    cache=HERE/('evaluation-'+month+'.npz');mp=cache.with_suffix('.json')
    if cache.exists():return r.read(mp),dict(np.load(cache))
    parent,sample,features,_,_,_,unseen=backend
    if month in ['december','january']:
        meta=r.read(RESEARCH/('predictions-'+month+'.json'));old=np.load(RESEARCH/('predictions-'+month+'.npz'))
        entries=r.read(RESEARCH/(month+'-evaluation-roster.json'));store={}
        for i,e in enumerate(entries):
            b=old[f'{i}_base'];q=old[f'{i}_bce'] if e['condition']==0 else b.copy()
            for k in ['y','m']:store[f'{i}_{k}']=old[f'{i}_{k}']
            use=old[f'{i}_unseen']&(b.max(0)<.5)&(q.max(0)>=.5)&(e['condition']==0)
            s=sample(e);g=gate_features(s,b,q)
            for k,v in dict(base=b,q=q,g=g,use=use).items():store[f'{i}_{k}']=v
            if (i+1)%96==0:progress('external_features',month=month,windows=i+1,total=len(entries))
    else:raise ValueError('No new February access in this stage')
    np.savez_compressed(cache,**store);r.write(mp,meta);return meta,store

def evaluate(args,backend):
    selection=r.read(HERE/'frozen-selection.json');policy=selection['policy']
    if policy=='base':
        r.write(args.output/'task2_gate_decision.json',dict(selected='R127',reason='No gate passed the internal November criteria; no external-policy selection performed',official_S_queue=None));return
    gate=lgb.Booster(model_file=str(HERE/'gate-final.txt')) if policy=='learned' else None
    reports={}
    for month in ['december','january']:
        meta,store=final_cache(args,backend,month);record={};rows={}
        for name in ['base','rescue',policy]:record[name],rows[name]=evaluate_store(backend,meta,store,name,gate)
        b=record['base'];c=record[policy];u=record['rescue'];target=max(.004,.5*(u['onset_macro']-b['onset_macro']))
        passed=c['onset_macro']>=b['onset_macro']+target and c['negative_false_any']<=b['negative_false_any']+.03
        reports[month]=dict(scores=record,passed=passed,required_gain=target,uncertainty=paired_ci(rows['base'],rows[policy]),windows=rows)
        r.write(args.output/'task2_gate_external_results.json',dict(policy=selection,reports=reports,official_S_queue=None,all_months_reused_development=True))
        print(json.dumps(dict(month=month,policy=policy,passed=passed,scores={k:dict(onset=v['onset_macro'],negative=v['negative_false_any'],kept=v['retained_rescue_links']) for k,v in record.items()})),flush=True)
    r.write(args.output/'task2_gate_decision.json',dict(selected='gated research candidate' if all(v['passed'] for v in reports.values()) else 'R127',policy=selection,December_January_both_passed=all(v['passed'] for v in reports.values()),official_S_queue=None))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','crossfit','select','evaluate'])
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();backend=r.setup(args)
    if args.action=='prepare':prepare(args,backend)
    elif args.action=='crossfit':crossfit(args,backend)
    elif args.action=='select':select(args,backend)
    else:evaluate(args,backend)
if __name__=='__main__':main()
