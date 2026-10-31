"""R127-anchored scalar event residual and matched hard-negative control.

Native R127 weights are frozen. A bounded scalar log-odds residual changes each
link's curve amplitude, preserving its normalized six-step shape. Binary .5
crossings can change; no first-time, duration or clearing head is retrained.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='4'
import argparse,copy,hashlib,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import run_onset_experiment as ref

BASE=Path(__file__).resolve().parents[1]
RUN=BASE/'work/anchored_residual'
OUT=BASE/'outputs/anchored_residual'
SEEDS=[20261029,20261030,20261031]
EPOCHS=45
ALPHAS=[0.,.25,.5,.75,1.]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2),encoding='utf8');tmp.replace(p)
def log(**v):print(json.dumps(dict(seconds=round(time.time()-START,1),**v)),flush=True)

def curve_features(x,parent):
    parent=np.asarray(parent,float);peak=parent.max(1)
    shape=np.divide(parent,peak[:,None],out=np.zeros_like(parent),where=peak[:,None]>0)
    b=np.log(np.clip(peak,1e-7,1-1e-7))-np.log1p(-np.clip(peak,1e-7,1-1e-7))
    return np.column_stack([x,b,shape]).astype('float32')

def compose(parent,delta,alpha=1.):
    p=np.asarray(parent,dtype=float);d=np.asarray(delta,dtype=float)*alpha
    if p.ndim!=2 or p.shape[1]!=6 or d.shape!=(len(p),):raise ValueError('Expected [N,6] curve and [N] residual')
    peak=p.max(1);base=np.clip(peak,1e-7,1-1e-7)
    z=np.log(base)-np.log1p(-base)+d
    new=np.clip(np.exp(-np.logaddexp(0.,-z))+peak-base,0,1)
    shape=np.divide(p,peak[:,None],out=np.zeros_like(p),where=peak[:,None]>0)
    q=(shape*new[:,None]).astype('float32')
    q[d==0]=np.asarray(parent,dtype='float32')[d==0]
    return q

def apply_condition(parent,delta,condition,alpha=1.):
    return compose(parent,delta,alpha) if condition==0 else np.asarray(parent,dtype='float32').copy()

def prepare():
    RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(anchor='Unchanged R127',fit_dates=['2030-12-01','2030-12-20'],held_dates=['2030-12-21','2030-12-31'],
        rationale='December is after native R127 June-November weight fitting. It was historically used for R127 routing selection, so it is not untouched parent-validation data.',
        additional_windows='All 8 roads, every December day, fixed 06:00/12:00/18:00 origins; include only history-defined onset windows; union with original frozen December positive roster; deduplicate exact origins',
        negatives='Retain all links with any observed future cell, including all positives and all observed negatives. Hard variant upweights only observed negative cells with parent probability>=.25 by4. Unknown cells are masked, not event negatives.',
        control='Identical network/seeds/rows/epochs; hard-negative multiplier1 instead of4',
        loss='Window/road-balanced masked cell BCE plus .03*delta^2 trust penalty; no positive-class upweighting',
        architecture=[947,64,32,1],activation='GELU',dropout=.1,residual='5*tanh(raw), added to clipped R127 peak log-odds with endpoint correction; normalized six-step shape preserved',
        epochs=EPOCHS,seeds=SEEDS,optimizer='AdamW lr.0003 decay.0001',batch=2048,
        selection='January only: maximize onset macro IoU over alpha [0,.25,.5,.75,1] subject to both total FP cells and road-macro eligible-negative FPR <= R127; ties prefer smaller alpha. Alpha0 is explicit fallback.',
        evaluation='Original same 183/200 February positive-onset windows, separate populations; also report all356 older onset windows including natural negatives',
        fixed_queue_threshold=.5,official_S_queue=None,February_reused_development=True)
    write(OUT/'protocol.json',protocol)
    if (RUN/'complete.json').exists():log(phase='PREPARE_ALREADY_COMPLETE');return
    model,sample,r2,feature,audit=ref.backend()
    sys.path.insert(0,str(ref.REPRO))
    from task2q.data import network,day,make_sample
    old,oldmeta=ref.load('december');oldp=np.load(ref.RUN/'december/parent.npy')
    xx=[];yy=[];mm=[];pp=[];ww=[];meta=[];seen=set();cursor=0
    rng=np.random.default_rng(SEEDS[0]);checks=[]
    def append(s,z,p,source):
        nonlocal cursor
        key=(s['panel'],s['origin'][:19])
        if key in seen:return
        seen.add(key)
        y=np.asarray(s['y']).T;m=np.asarray(s['m']).T;p=np.asarray(p)
        ids=np.flatnonzero(m.any(1));positive=(y*m).any(1);risk=p.max(1)>=.1
        keep=ids;ipw=np.ones(len(keep))
        if not len(keep):return
        count=int(m.sum());weights=ipw/max(count,1)
        x=curve_features(z,p);a,b=cursor,cursor+len(keep);cursor=b
        xx.append(x[keep]);yy.append(y[keep]);mm.append(m[keep]);pp.append(p[keep]);ww.append(weights.astype('float32'))
        meta.append(dict(panel=s['panel'],origin=s['origin'],positive=bool(s['has_future_queue']),start=a,end=b,source=source,selected_links=keep.tolist(),full_nodes=len(y),observed_cells=count,fit=s['origin'][:10]<='2030-12-20'))
    for e in oldmeta['windows']:
        a,b=e['start'],e['end']
        s=dict(panel=e['panel'],origin=e['origin'],has_future_queue=e['positive'],y=old['y'][a:b].T,m=old['m'][a:b].T)
        append(s,old['X'][a:b],oldp[a:b],'frozen_positive_roster')
    for panel in r2.PANELS:
        net=network(ref.DATA,panel)
        for daynum in range(1,32):
            date=f'2030-12-{daynum:02d}';v=day(ref.DATA,panel,date,cache_dir=RUN/'daycache')
            for slot in [72,144,216]:
                s=make_sample(v,net,slot,panel)
                if s['condition']!=0 or not s['m'].any():continue
                if (panel,s['origin'][:19]) in seen:continue
                p=model.probability(s).T;z=feature(s)
                if not any(c['panel']==panel for c in checks):
                    altered=copy.deepcopy(s);altered['y']=1-s['y'];altered['m']=~s['m'];altered['has_future_queue']=not s['has_future_queue']
                    np.testing.assert_allclose(z,feature(altered),equal_nan=True)
                    np.testing.assert_array_equal(p,model.probability(altered).T)
                    future=copy.deepcopy(v)
                    for k in ['speed','flow','occupancy','quality','origin_speed','origin_flow','origin_occupancy','origin_quality']:future[k][slot+1:]=999
                    future['eligible'][slot+1:]=False
                    q=make_sample(future,net,slot,panel)
                    np.testing.assert_allclose(z,feature(q),equal_nan=True);np.testing.assert_array_equal(p,model.probability(q).T)
                    checks.append(dict(panel=panel,passed=True))
                append(s,z,p,'fixed_history_only_onset_origins')
            if daynum%10==0:log(phase='prepare',panel=panel,date=date,windows=len(meta),rows=cursor)
    X=np.concatenate(xx);y=np.concatenate(yy);m=np.concatenate(mm);p=np.concatenate(pp);w=np.concatenate(ww)
    fit=np.zeros(len(X),bool);counts={}
    for e in meta:counts[(e['panel'],e['fit'])]=counts.get((e['panel'],e['fit']),0)+1
    for e in meta:
        a,b=e['start'],e['end'];fit[a:b]=e['fit'];w[a:b]/=counts[(e['panel'],e['fit'])]
    for key,value in [('X',X),('y',y),('m',m),('parent',p),('w',w),('fit',fit)]:np.save(RUN/(key+'.npy'),value)
    positives=((y>0)&m).any(1);complete_negative=m.all(1)&~positives;hard=(p>=.25)&(y==0)&m
    stats=dict(rows=len(X),fit_rows=int(fit.sum()),held_rows=int((~fit).sum()),windows=len(meta),fit_windows=sum(e['fit'] for e in meta),held_windows=sum(not e['fit'] for e in meta),known_positive_links=int(positives.sum()),complete_negative_links=int(complete_negative.sum()),unknown_event_links=int((~(positives|complete_negative)).sum()),hard_negative_cells=int(hard.sum()),hard_complete_negative_links=int((complete_negative&(p.max(1)>=.1)).sum()))
    write(RUN/'complete.json',dict(windows=meta,statistics=stats,causality=checks))
    write(OUT/'training_data.json',dict(statistics=stats,causality=checks))
    log(phase='PREPARE_COMPLETE',**stats)

class Residual(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.Sequential(nn.Linear(947,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,32),nn.GELU(),nn.Dropout(.1),nn.Linear(32,1))
        nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)
    def forward(self,x):return 5*torch.tanh(self.layers(x).squeeze(-1))

def torch_curve(parent,delta):
    peak=parent.max(1).values;base=peak.clamp(1e-7,1-1e-7)
    e=(torch.sigmoid(torch.logit(base)+delta)+peak-base).clamp(0,1)
    shape=parent/peak[:,None].clamp_min(1e-30)
    return shape*e[:,None]

def residual_loss(delta,parent,y,m,hard_multiplier):
    y=torch.where(m,y,torch.zeros_like(y));q=torch_curve(parent,delta).clamp(1e-7,1-1e-7)
    weights=torch.where((parent>=.25)&(y==0)&m,hard_multiplier,1.)
    cell=F.binary_cross_entropy(q,y,reduction='none')*m*weights
    # Row weights already include window observed-cell denominator and IPW.
    return cell.sum(1)+.03*delta.square()*m.sum(1)

def normalize(x,mean,scale):return np.nan_to_num((np.asarray(x)-mean)/scale,nan=0,posinf=0,neginf=0).clip(-8,8).astype('float32')

def train():
    torch.set_num_threads(4);meta=read(RUN/'complete.json');fit=np.load(RUN/'fit.npy');X=np.load(RUN/'X.npy',mmap_mode='r')
    z=np.asarray(X[fit],float);valid=np.isfinite(z);count=valid.sum(0);mean=np.where(valid,z,0).sum(0)/np.maximum(count,1)
    scale=np.sqrt(np.maximum(np.where(valid,(z-mean)**2,0).sum(0)/np.maximum(count,1),1e-4)).astype('float32');mean=mean.astype('float32');del z
    np.savez(OUT/'normalization.npz',mean=mean,scale=scale)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    data={k:torch.as_tensor(np.array(np.load(RUN/(k+'.npy'))[fit]),device=device,dtype=torch.bool if k=='m' else torch.float32) for k in ['y','m','parent','w']}
    data['x']=torch.from_numpy(normalize(X[fit],mean,scale)).to(device);data['w']/=data['w'].mean()
    n=len(data['x'])
    write(OUT/'environment.json',dict(device=device,torch=torch.__version__,gpu=torch.cuda.get_device_name() if device=='cuda' else None))
    for seed in SEEDS:
        for variant,multiplier in [('hard_negatives',4.),('matched_standard',1.)]:
            folder=OUT/f'{variant}_{seed}';folder.mkdir(parents=True,exist_ok=True)
            if (folder/'complete.json').exists():continue
            torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
            if device=='cuda':torch.cuda.manual_seed_all(seed)
            model=Residual().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001);rng=np.random.default_rng(seed);trace=[]
            for epoch in range(EPOCHS):
                model.train();order=rng.permutation(n);total=0.
                for j in range(0,n,2048):
                    ids=torch.as_tensor(order[j:j+2048],device=device)
                    opt.zero_grad(set_to_none=True);d=model(data['x'][ids])
                    loss=(residual_loss(d,data['parent'][ids],data['y'][ids],data['m'][ids],multiplier)*data['w'][ids]).mean()
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);opt.step();total+=float(loss.detach())*len(ids)
                trace.append(dict(epoch=epoch+1,loss=total/n))
                if (epoch+1)%15==0:log(phase='train',seed=seed,variant=variant,**trace[-1])
            torch.save(dict(model=model.state_dict(),mean=mean,scale=scale,variant=variant,seed=seed,epochs=EPOCHS),folder/'model.pt')
            write(folder/'trace.json',trace);write(folder/'complete.json',dict(trained=True,epochs=EPOCHS,fit_rows=n,parameters=sum(v.numel() for v in model.parameters())))
    log(phase='TRAIN_COMPLETE')

def delta_predict(model,X,mean,scale,device):
    model.eval();out=[]
    with torch.inference_mode():
        for j in range(0,len(X),4096):out.append(model(torch.from_numpy(normalize(X[j:j+4096],mean,scale)).to(device)).cpu().numpy())
    return np.concatenate(out)

def score(p,y,m,meta,positive_only=True):
    data=dict(y=y,m=m)
    s=ref.score_population(p,data,meta,.5,positive_only)
    roads={};windows=[]
    for e in meta['windows']:
        if positive_only and not e['positive']:continue
        a,b=e['start'],e['end'];yy=np.asarray(y[a:b])>0;mm=np.asarray(m[a:b],bool);q=p[a:b]>=.5
        fp=int((q&~yy&mm).sum());tn=int((~yy&mm).sum());tp=int((q&yy&mm).sum());fn=int((~q&yy&mm).sum())
        totals=roads.setdefault(e['panel'],[0,0]);totals[0]+=fp;totals[1]+=tn
        windows.append(dict(panel=e['panel'],origin=e['origin'],FP=fp,TP=tp,FN=fn,eligible_negative=tn,false_any=bool((q&mm).any()),positive=e['positive']))
    s['macro_FPR']=float(np.mean([fp/max(den,1) for fp,den in roads.values()]))
    s['window_errors']=windows;s['per_road_FPR']={k:fp/max(den,1) for k,(fp,den) in roads.items()}
    return s

def fp_bootstrap(a,b,reps=2000):
    aa=a['window_errors'];bb=b['window_errors'];assert [(e['panel'],e['origin']) for e in aa]==[(e['panel'],e['origin']) for e in bb]
    days=sorted({e['origin'][:10] for e in aa});byday={d:sum(x['FP']-y['FP'] for x,y in zip(aa,bb) if x['origin'][:10]==d) for d in days}
    rng=np.random.default_rng(42);values=[sum(byday[d] for d in rng.choice(days,len(days))) for _ in range(reps)]
    return dict(delta=a['cell_counts']['FP_cells']-b['cell_counts']['FP_cells'],day_cluster_95CI=np.quantile(values,[.025,.975]).tolist())

def evaluate():
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    populations={}
    for name in ['january','new_phase_population','old410']:
        data,meta=ref.load(name);parent=np.load(ref.RUN/name/'parent.npy')
        populations[name]=dict(X=curve_features(data['X'],parent),y=data['y'],m=data['m'],parent=parent,meta=meta)
    meta=read(RUN/'complete.json');fit=np.load(RUN/'fit.npy');held=~fit
    # Heldout metadata offsets are rebuilt after row selection, without splitting days.
    heldmeta=[];cursor=0
    for e in meta['windows']:
        if e['fit']:continue
        n=e['end']-e['start'];heldmeta.append(dict(e,start=cursor,end=cursor+n));cursor+=n
    populations['December_held']=dict(X=np.load(RUN/'X.npy',mmap_mode='r')[held],y=np.load(RUN/'y.npy')[held],m=np.load(RUN/'m.npy')[held],parent=np.load(RUN/'parent.npy')[held],meta=dict(windows=heldmeta))
    results={};deltas={};checks=[]
    for variant in ['hard_negatives','matched_standard']:
        for seed in SEEDS:
            folder=OUT/f'{variant}_{seed}';ck=torch.load(folder/'model.pt',map_location=device,weights_only=False);model=Residual().to(device);model.load_state_dict(ck['model'])
            for pop,v in populations.items():
                d=delta_predict(model,v['X'],ck['mean'],ck['scale'],device);deltas[(variant,seed,pop)]=d;np.save(folder/(pop+'-residual.npy'),d)
    for pop,v in populations.items():
        parent=v['parent'];baseline=score(parent,v['y'],v['m'],v['meta']);report={'R127':baseline}
        for variant in ['hard_negatives','matched_standard']:
            jan=populations['january'];jd=np.mean([deltas[(variant,s,'january')] for s in SEEDS],axis=0)
            jb=score(jan['parent'],jan['y'],jan['m'],jan['meta'])
            selection=[]
            for alpha in ALPHAS:
                p=compose(jan['parent'],jd,alpha);s=score(p,jan['y'],jan['m'],jan['meta'])
                eligible=s['cell_counts']['FP_cells']<=jb['cell_counts']['FP_cells'] and s['macro_FPR']<=jb['macro_FPR']+1e-15
                selection.append(dict(alpha=alpha,IoU=s['onset_IoU'],FP=s['cell_counts']['FP_cells'],macro_FPR=s['macro_FPR'],eligible=bool(eligible)))
            choice=max([x for x in selection if x['eligible']],key=lambda x:(x['IoU'],-x['alpha']))['alpha']
            d=np.mean([deltas[(variant,s,pop)] for s in SEEDS],axis=0)
            current={}
            for name,alpha in [('raw_alpha1',1.),('January_selected',choice)]:
                q=compose(parent,d,alpha);s=score(q,v['y'],v['m'],v['meta'])
                peak0=parent.max(1);peakq=q.max(1);use=(peak0>0)&(peakq>0)
                shape_error=float(np.max(np.abs(q[use]/peakq[use,None]-parent[use]/peak0[use,None]))) if use.any() else 0.
                assert shape_error<2e-6
                s['IoU_vs_R127']=ref.bootstrap(s,baseline);s['FP_vs_R127']=fp_bootstrap(s,baseline)
                s['shape_max_abs_error']=shape_error;s['alpha']=choice if name=='January_selected' else 1.
                current[name]=s;np.save(OUT/f'{pop}-{variant}-{name}-probability.npy',q)
                if pop in ['old410','December_held']:current[name+'_all_onset_windows']=score(q,v['y'],v['m'],v['meta'],False)
                checks.append(dict(population=pop,variant=variant,policy=name,shape_error=shape_error))
            current['January_selection_grid']=selection
            seed_results=[]
            for seed in SEEDS:
                z=score(compose(parent,deltas[(variant,seed,pop)]),v['y'],v['m'],v['meta'])
                seed_results.append(dict(seed=seed,IoU=z['onset_IoU'],FP=z['cell_counts']['FP_cells']))
            current['individual_seed_raw']=seed_results
            report[variant]=current
        report['hard_vs_standard']={policy:dict(IoU=ref.bootstrap(report['hard_negatives'][policy],report['matched_standard'][policy]),FP=fp_bootstrap(report['hard_negatives'][policy],report['matched_standard'][policy])) for policy in ['raw_alpha1','January_selected']}
        if pop in ['old410','December_held']:report['R127_all_onset_windows']=score(parent,v['y'],v['m'],v['meta'],False)
        results[pop]=report
        log(phase='evaluated',population=pop,baseline_IoU=baseline['onset_IoU'],baseline_FP=baseline['cell_counts']['FP_cells'],hard_raw_IoU=report['hard_negatives']['raw_alpha1']['onset_IoU'],hard_raw_FP=report['hard_negatives']['raw_alpha1']['cell_counts']['FP_cells'],alpha=report['hard_negatives']['January_selected']['alpha'],hard_selected_IoU=report['hard_negatives']['January_selected']['onset_IoU'],hard_selected_FP=report['hard_negatives']['January_selected']['cell_counts']['FP_cells'])
    write(OUT/'results.json',dict(populations=results,shape_checks=checks,official_S_queue=None,February_reused_development=True))
    log(phase='EVALUATION_COMPLETE')

if __name__=='__main__':
    START=time.time();ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','train','evaluate']);args=ap.parse_args()
    {'prepare':prepare,'train':train,'evaluate':evaluate}[args.action]()
