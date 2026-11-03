"""Same features/capacity, genuinely denser training origins; expanded heldout validation."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
import json,time,hashlib,gc
from pathlib import Path
import numpy as np
import lightgbm as lgb
import train_ongoing_speed as speed
from speed_distribution_core import PANELS,odds_mix
B=speed.old.BASE;O=B/'outputs/dense_ongoing_v29_20261102';DATA=B/'outputs/dense_ongoing_v28_20261102';VALID=B/'outputs/expanded_ongoing_v25_20261102';START=time.time();SEEDS=speed.old.SEEDS;CONFIGS=[dict(name=f'dense_{int(alpha*100)}',alpha=alpha) for alpha in [.5,1.]]
def save(n,x):speed.old.write(O/n,x)
def log(**x):print(json.dumps(dict(seconds=round(time.time()-START,1),**x)),flush=True)
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def parents(panel,seed,h):
    a=B/'outputs/ongoing_jaccard_20261031'/f'development_j075_{seed}'/f'{panel}.txt';b=B/'outputs/ongoing_horizon_20261031'/f'development_bce_{seed}'/f'{panel}-h{(h+1)*5}.txt';return lgb.Booster(model_file=str(a)),lgb.Booster(model_file=str(b))
def frame(x,h):return np.column_stack([x,np.full((len(x),30),np.nan),np.full(len(x),(h+1)*5)]).astype('float32')
def train(seed,panel):
    folder=DATA/'fit'/panel;meta=speed.old.read(folder/'complete.json');x=np.load(folder/'X.npy',mmap_mode='r');y=np.load(folder/'y.npy');m=np.load(folder/'m.npy');w=np.load(folder/'w.npy');months=np.load(folder/'month.npy');assert x.shape[1]==274 and (months>=203006).all() and (months<203010).all();positive=np.zeros(len(x),bool);density=np.zeros(len(x))
    for win in meta['windows']:positive[win['start']:win['end']]=win['positive'];density[win['start']:win['end']]=win['density'];assert '2030-06'<=win['origin'][:7]<='2030-09'
    for h in range(6):
        ids=np.flatnonzero(m[:,h]);tx=frame(x[ids],h);oldx=np.column_stack([tx[:,:274],tx[:,-1]]);base,residual=parents(panel,seed,h);assert base.num_feature()==275 and residual.num_feature()==275;initial=base.predict(oldx,raw_score=True,num_threads=4)+residual.predict(oldx,raw_score=True,num_threads=4);pos=positive[ids];weight=w[ids].astype(float)*np.where(pos,np.clip(.1/(density[ids]+.01),.5,8.),1.)
        for label,mass in [(True,.8),(False,.2)]:
            take=pos==label;assert take.any();weight[take]*=mass/weight[take].sum()
        weight/=weight.mean();params=dict(objective='binary',learning_rate=.03,num_leaves=15,max_depth=4,min_data_in_leaf=100,lambda_l2=50,max_bin=127,feature_fraction=1.,bagging_fraction=.9,bagging_freq=1,num_threads=4,seed=seed,deterministic=True,force_col_wise=True,verbosity=-1);model=lgb.train(params,lgb.Dataset(tx,label=y[ids,h],weight=weight,init_score=initial),num_boost_round=100);path=O/'models'/f'dense_{seed}'/f'{panel}-h{(h+1)*5}.txt';path.parent.mkdir(parents=True,exist_ok=True);model.save_model(str(path));log(stage='TRAIN_DENSIFIED',seed=seed,panel=panel,horizon=(h+1)*5,rows=len(ids));del model,tx,oldx,base,residual;gc.collect()
def raw(seed,kind,d,meta):
    cache=O/'cache'/f'{kind}_{seed}.npy';cache.parent.mkdir(exist_ok=True)
    if cache.exists():return np.load(cache)
    out=np.zeros(d['y'].shape,dtype='float64')
    for panel in PANELS:
        ids=np.concatenate([np.arange(w['start'],w['end']) for w in meta['windows'] if w['panel']==panel]);x=d['X'][ids]
        for h in range(6):
            if kind=='base':
                base,residual=parents(panel,seed,h);xx=np.column_stack([x,np.full(len(x),(h+1)*5)]);out[ids,h]=base.predict(xx,raw_score=True,num_threads=4)+residual.predict(xx,raw_score=True,num_threads=4)
            else:
                root=B/'outputs/ongoing_control_confirmation_v23_20261102' if kind=='control' else O;name=f'control_{seed}' if kind=='control' else f'dense_{seed}';model=lgb.Booster(model_file=str(root/'models'/name/f'{panel}-h{(h+1)*5}.txt'));out[ids,h]=model.predict(frame(x,h),raw_score=True,num_threads=4)
    np.save(cache,out);return out
def predict(seeds,kind,alpha,d,meta):
    prob=[]
    for seed in seeds:
        z=raw(seed,'base',d,meta)
        if kind!='base':z=z+alpha*raw(seed,kind,d,meta)
        prob.append(np.exp(-np.logaddexp(0.,-z)))
    return odds_mix(d['parent'],np.mean(prob,axis=0).astype('float32'))
def main():
    O.mkdir(exist_ok=False);assert speed.old.read(DATA/'verification.json')['passed'];assert speed.old.read(VALID/'verification.json')['passed'];statistics=speed.old.read(DATA/'statistics.json');assert all(v['new_origins']>v['original_windows'] for v in statistics.values());save('decision_before_training.json',dict(basis='ExpandedOctober error audit identifies entryFN/clearanceFP,not lack of anotherarchitecture. v28 adds actualnew5/10min origins withnewmaskedlabels andidenticaloldfeatures/priors;everyoldrow reproducedexactly. Adjacentorigins arecorrelated,notnewindependentdates.',scope='Matcheddata augmentation study,not newfeatures/capacity',fit='June-Sep only',data_statistics=statistics,configs=CONFIGS,seeds=SEEDS,model='Same additional100tree15leafdepth4min100L250lr.03,maxbin127,fraction1,bag.9 asv23matchedoriginalcontrol;305layout original274+30NaN+horizon',init='Original900+100incumbent,in-sample/nearbyorigins not OOF',matched_control='All144v23controlmodels read-only reused,no redundantfit;controlalpha.5 fixed',weights='SameIPW/inversedensity80positive20aux;unknowncells excluded;no futuredensity feature',October='Expanded673positivewindows,fixed.25;top2 IoU>=max(incumbent,matchedcontrol)+.001 andFP/FN<=1.05incumbent',November='OnlyOctselectedoriginal3seeds,expanded673positivewindows,noreselection',January_February='Not accessed',production_onset_and_default_unchanged=True,Kaggle_upload=False));files=[Path(__file__),DATA/'verification.json',DATA/'statistics.json',VALID/'metadata.json',VALID/'X.npy',VALID/'parent.npy',VALID/'y.npy',VALID/'m.npy',B/'outputs/current_task2_best.json']
    for panel in PANELS:
        files.extend(DATA/'fit'/panel/name for name in ['complete.json','X.npy','y.npy','m.npy','w.npy','month.npy'])
        for seed in SEEDS:
            files.append(B/'outputs/ongoing_jaccard_20261031'/f'development_j075_{seed}'/f'{panel}.txt')
            for h in range(5,31,5):files.extend([B/'outputs/ongoing_horizon_20261031'/f'development_bce_{seed}'/f'{panel}-h{h}.txt',B/'outputs/ongoing_control_confirmation_v23_20261102/models'/f'control_{seed}'/f'{panel}-h{h}.txt'])
    hashes={str(f.relative_to(B)):sha(f) for f in files};save('predeclared_hashes.json',hashes);d={k:np.load(VALID/f'{k}.npy',mmap_mode='r') for k in ['X','y','m','parent']};meta=speed.old.read(VALID/'metadata.json');october=speed.old.subset(meta,'2030-10');baseline=speed.old.score(predict(SEEDS[:1],'base',0,d,meta),d,october,.25);control=speed.old.score(predict(SEEDS[:1],'control',.5,d,meta),d,october,.25);save('October_controls.json',dict(incumbent=baseline,matched_original_data_control=control))
    for panel in PANELS:train(SEEDS[0],panel)
    rows=[]
    for cfg in CONFIGS:
        prob=predict(SEEDS[:1],'dense',cfg['alpha'],d,meta);np.save(O/f"expanded_{cfg['name']}_firstseed_probability.npy",prob);s=speed.old.score(prob,d,october,.25);a,b=s['cell_counts'],baseline['cell_counts'];ok=s['onset_IoU']>=max(baseline['onset_IoU'],control['onset_IoU'])+.001 and a['FP_cells']<=1.05*b['FP_cells'] and a['FN_cells']<=1.05*b['FN_cells'];rows.append(dict(config=cfg,October=s,eligible=bool(ok),vs_original_data_control=speed.old.prior.REF.bootstrap(s,control)));save('October_search.json',rows);log(stage='OCTOBER',name=cfg['name'],IoU=s['onset_IoU'],FP=a['FP_cells'],FN=a['FN_cells'],eligible=bool(ok))
    selected=sorted([r for r in rows if r['eligible']],key=lambda r:r['October']['onset_IoU'],reverse=True)[:2];save('selected_candidates.json',selected)
    if selected:
        for seed in SEEDS[1:]:
            for panel in PANELS:train(seed,panel)
        baseprob=predict(SEEDS,'base',0,d,meta);gold=np.load(VALID/'incumbent900plus100_odds50_probability.npy');error=float(np.max(abs(baseprob-gold)));assert error<=1e-7;save('incumbent_parity.json',dict(passed=True,max_probability_error=error));nov=speed.old.subset(meta,'2030-11');baseline=speed.old.score(baseprob,d,nov,.25);controlprob=predict(SEEDS,'control',.5,d,meta);control=speed.old.score(controlprob,d,nov,.25);reports={}
        for row in selected:
            cfg=row['config'];prob=predict(SEEDS,'dense',cfg['alpha'],d,meta);np.save(O/f"expanded_{cfg['name']}_probability.npy",prob);s=speed.old.score(prob,d,nov,.25);reports[cfg['name']]=dict(score=s,vs_incumbent=speed.old.prior.REF.bootstrap(s,baseline),vs_matched_original_data=speed.old.prior.REF.bootstrap(s,control))
        save('November_confirmation.json',dict(incumbent=baseline,matched_original_data_control=control,candidates=reports))
    for f,h in hashes.items():assert sha(B/f)==h
    save('verification.json',dict(passed=True,all_frozen_sources_unchanged=True,actual_new_origin_training=True,June_September_only_fit=True,expanded_October_only_selection=True,expanded_November_no_reselection=True,no_January_February=True,production_onset_and_default_unchanged=True,independent_raw_acceptance_pending=True,Kaggle_upload=False));log(stage='COMPLETE',advanced=len(selected))
if __name__=='__main__':main()
