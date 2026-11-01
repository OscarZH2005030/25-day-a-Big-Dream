"""Full-horizon BCE and window Soft-IoU continuation; frozen onset and January-only thresholds."""
import gc, hashlib, json, shutil, time, zipfile,sys,subprocess,concurrent.futures,argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import train_ongoing_speed as speed
from speed_distribution_core import odds_mix,PANELS
from ongoing_jaccard_core import objective,sigmoid
old=speed.old;BASE=old.BASE;OUT=BASE/'outputs/ongoing_horizon_20261031'
SEEDS=old.SEEDS;START=time.time();read=old.read;write=old.write
CAPACITY={'bce':dict(strength=0.),'soft':dict(strength=.75)}
CONFIGS=[dict(name=f'{c}_a{int(alpha*100)}',capacity=c,alpha=alpha,mix='odds50') for c in CAPACITY for alpha in [.5,1.]]
def log(**kw):print(json.dumps(dict(seconds=round(time.time()-START,1),**kw)),flush=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check():
    for p,h in read(OUT/'frozen_onset_hashes.json').items():assert sha(BASE/p)==h
def init():
    OUT.mkdir(exist_ok=True)
    write(OUT/'protocol.json',dict(date='2026-10-31',method='Frozen900-tree perroad parent + six5-30minute residual specialists. Residuals use external init_score; saved residual contains100newtrees only.',
      inputs='Same274 causal features,8143June-Novwindows,selectednodes andIPW;allknownhorizons. Unknownfuturecells excluded. Perhorizon fit avoids sharing all short/long horizon corrections.',
      fitting='Perroad/horizon/seed100trees,15leavesdepth4min100L2=50lr.03; same inverse-densitywindowfactor and80/20positive/auxmass,renormalized perhorizon. BCE or BCE+.75SoftIoU over nodes within window/horizon. Base900 frozen.',
      configs=CONFIGS,capacities=CAPACITY,seeds=SEEDS,
      October='June-Sep900-tree j075parent andJune-Sep residuals;fixed.25 picktop2 of4(BCE/Soft x residual logit shrinkage.5/1) byIoU, FP/FN<=1.15R127 preferred. No January choice of head or shrinkage. R127fitthese dates,developmentnotpureOOS.',
      November='Selected2three seeds,reportonly,noreselection. FinalJune-Nov900parent andresiduals.',
      January='Only scalar threshold .05.. .75 step.025 maximizingIoU;record priorFP/FNguardrail feasibility and bestfeasiblethreshold but primaryunconstrained as higherIoUpriority.',
      February='Allmodel/params/source+Januarythresholds hashfrozen before originalsame184/210ongoingpopulations. No tuning. Calendar64audit uses alreadyfrozen prior model andisreportonly,neverparameterselection.',
      caveat='Reliableobservation proxy,not officialS_queue;Februaryreuseddevelopment,notblind. Preserve0.82639,0.82245,0.82186,bestonsetearly_full unchanged.'))
    shutil.copy2(old.OUT/'frozen_onset_hashes.json',OUT/'frozen_onset_hashes.json');shutil.copy2(BASE/'outputs/ongoing_jaccard_20261031/overall_leaders.json',OUT/'incumbent.json')

def train_local(phase,capacities,seeds,panel):
    jobs={(cap,seed,h) for cap in capacities for seed in seeds for h in range(6) if not (OUT/f'{phase}_{cap}_{seed}'/f'{panel}-h{(h+1)*5}.txt').exists()}
    if not jobs:return
    source=old.RUN/'fit'/panel;meta=read(source/'complete.json');base=np.load(source/'X.npy',mmap_mode='r');extra=np.load(speed.RUN/'fit'/panel/'extra.npy',mmap_mode='r');xx=np.column_stack([base,extra]).astype('float32');yy=np.load(source/'y.npy');mm=np.load(source/'m.npy');month=np.load(source/'month.npy');use=month<203010 if phase=='development' else np.ones(len(month),bool);nodew=np.load(source/'w.npy');density=np.zeros(len(month));positive=np.zeros(len(month),bool);groups=np.zeros(len(month),dtype='int32')
    with np.load(BASE/'outputs/ongoing_analog_20261031/library'/f'{panel}.npz') as z:values={old.prior.old.normtime(o):v for o,v in zip(z['origin'],z['future'])}
    for i,w in enumerate(meta['windows']):
        v=values[old.prior.old.normtime(w['origin'])];known=np.isfinite(v);a,b=w['start'],w['end'];density[a:b]=((v<=0)&known).sum()/max(int(known.sum()),1);positive[a:b]=w['positive'];groups[a:b]=i
    for seed in seeds:
        initial=BASE/'outputs/ongoing_jaccard_20261031'/f'{phase}_j075_{seed}'/f'{panel}.txt';parent=lgb.Booster(model_file=str(initial))
        for h in range(6):
            pending=[cap for cap in capacities if (cap,seed,h) in jobs]
            if not pending:continue
            row=np.flatnonzero(mm[:,h]&use);x=np.column_stack([xx[row],np.full(len(row),(h+1)*5)]).astype('float32');y=yy[row,h];pos=positive[row];w=nodew[row].astype(float)*np.where(pos,np.clip(.1/(density[row]+.01),.5,8.),1.);group=groups[row]
            for label,mass in [(True,.8),(False,.2)]:
                take=pos==label;w[take]*=mass/w[take].sum()
            w/=w.mean();initial_score=parent.predict(x,raw_score=True,num_threads=8)
            for cap in pending:
                strength=CAPACITY[cap]['strength'];params=dict(objective='binary' if strength==0 else objective(w,group,strength),learning_rate=.03,num_leaves=15,max_depth=4,min_data_in_leaf=100,lambda_l2=50,max_bin=127,feature_fraction=.9,bagging_fraction=.9,bagging_freq=1,num_threads=8,seed=seed,deterministic=True,force_col_wise=True,verbosity=-1);model=lgb.train(params,lgb.Dataset(x,label=y,weight=w,init_score=initial_score),num_boost_round=100);assert model.num_trees()==100;file=OUT/f'{phase}_{cap}_{seed}'/f'{panel}-h{(h+1)*5}.txt';file.parent.mkdir(exist_ok=True);model.save_model(str(file));del model;gc.collect()
            log(stage='HORIZON',phase=phase,panel=panel,seed=seed,minutes=(h+1)*5,heads=pending,rows=len(row))

def train(phase,capacities,seeds):
    write(OUT/'execution_notes.json',dict(concurrent_road_processes=2,threads_per_process=8,model_parameters_unchanged=True))
    panels=[p for p in PANELS if any(not (OUT/f'{phase}_{c}_{seed}'/f'{p}.txt').exists() for c in capacities for seed in seeds)]
    def run(panel):
        subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--fit-panel',panel,'--phase',phase,'--capacities',','.join(capacities),'--seeds',','.join(str(x) for x in seeds)],check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,panels))

def raw_predictions(phase,seeds,pop,d,m,cap=None):
    parts=[]
    for seed in seeds:
        file=OUT/'prediction_cache'/f'{pop}_{phase}_{cap or "base"}_{seed}.npy';file.parent.mkdir(exist_ok=True)
        if file.exists():parts.append(np.load(file));continue
        result=np.empty(d['y'].shape,dtype='float64')
        for panel in PANELS:
            ids=np.concatenate([np.arange(w['start'],w['end']) for w in m['windows'] if w['panel']==panel]);x=d['X'][ids]
            parent=None if cap else lgb.Booster(model_file=str(BASE/'outputs/ongoing_jaccard_20261031'/f'{phase}_j075_{seed}'/f'{panel}.txt'))
            for h in range(6):
                model=parent if cap is None else lgb.Booster(model_file=str(OUT/f'{phase}_{cap}_{seed}'/f'{panel}-h{(h+1)*5}.txt'));result[ids,h]=model.predict(np.column_stack([x,np.full(len(x),(h+1)*5)]),raw_score=True,num_threads=8)
        np.save(file,result);parts.append(result)
    return np.stack(parts)

def preds(pop,phase,configs,seeds):
    d,m=speed.query(pop);base=raw_predictions(phase,seeds,pop,d,m);cache={};prob={}
    for c in configs:
        cap=c['capacity']
        if cap not in cache:cache[cap]=raw_predictions(phase,seeds,pop,d,m,cap)
        p=np.mean(sigmoid(base+c['alpha']*cache[cap]),axis=0).astype('float32');prob[c['name']]=odds_mix(d['parent'],p);np.save(OUT/f"{pop}_{c['name']}_probability.npy",prob[c['name']])
    return prob,d,m

def screen():
    train('development',list(CAPACITY),SEEDS[:1]);p,d,m=preds('development','development',CONFIGS,SEEDS[:1]);octm=old.subset(m,'2030-10');base=old.score(d['parent'],d,octm);rows=[]
    for cfg in CONFIGS:
        score=old.score(p[cfg['name']],d,octm);c=score['cell_counts'];b=base['cell_counts'];ok=c['FP_cells']<=1.15*b['FP_cells'] and c['FN_cells']<=1.15*b['FN_cells'];rows.append(dict(config=cfg,eligible=bool(ok),October=score));log(stage='OCTOBER',name=cfg['name'],**old.compact(score))
    write(OUT/'October_search.json',rows);write(OUT/'selected_candidates.json',sorted(rows,key=lambda x:(x['eligible'],x['October']['onset_IoU']),reverse=True)[:2])

def confirm():
    configs=[r['config'] for r in read(OUT/'selected_candidates.json')];train('development',sorted({c['capacity'] for c in configs}),SEEDS);p,d,m=preds('development','development',configs,SEEDS);novm=old.subset(m,'2030-11');rows={}
    for cfg in configs:
        name=cfg['name'];s=old.score(p[name],d,novm);rows[name]=s;log(stage='NOVEMBER',name=name,**old.compact(s))
    write(OUT/'November_confirmation.json',dict(R127=old.score(d['parent'],d,novm),candidates=rows))
def calibrate():
    configs=[r['config'] for r in read(OUT/'selected_candidates.json')];train('final',sorted({c['capacity'] for c in configs}),SEEDS);p,d,m=preds('january','final',configs,SEEDS);r127=old.score(d['parent'],d,m);base=read(BASE/'outputs/ongoing_jaccard_20261031/January_results.json')['candidates']['j075_odds50']['January'];rows={}
    for cfg in configs:
        name=cfg['name'];best=None;anybest=None;grid=[]
        for threshold in np.round(np.arange(.05,.7501,.025),3):
            s=old.score(p[name],d,m,float(threshold));c=s['cell_counts'];b=base['cell_counts'];ok=c['FP_cells']<=1.05*b['FP_cells'] and c['FN_cells']<=1.05*b['FN_cells'] and min(s['per_road'][r]-base['per_road'][r] for r in base['per_road'])>=-.03;grid.append(dict(threshold=float(threshold),eligible=bool(ok),**old.compact(s)));item=(float(threshold),s)
            if anybest is None or s['onset_IoU']>anybest[1]['onset_IoU']:anybest=item
            if ok and (best is None or s['onset_IoU']>best[1]['onset_IoU']):best=item
        t,s=anybest;selected_ok=next(g['eligible'] for g in grid if g['threshold']==t);rows[name]=dict(config=cfg,threshold=t,feasible=bool(selected_ok),best_feasible_threshold=best[0] if best else None,threshold_objective='maximum January IoU',January=s,grid=grid);log(stage='JANUARY',name=name,threshold=t,feasible=bool(selected_ok),**old.compact(s))
    write(OUT/'January_results.json',dict(R127=r127,incumbent=base,candidates=rows));check();files=list(OUT.glob('final_*/*.txt'))+list((BASE/'outputs/ongoing_jaccard_20261031').glob('final_j075_*/*.txt'))+[OUT/n for n in ['protocol.json','selected_candidates.json','January_results.json','frozen_onset_hashes.json']]+[Path(__file__),BASE/'outputs/ongoing_jaccard_core.py']
    write(OUT/'frozen_before_February.json',{str(p.relative_to(BASE)):sha(p) for p in files})
def evaluate():
    for p,h in read(OUT/'frozen_before_February.json').items():assert sha(BASE/p)==h
    cal=read(OUT/'January_results.json');configs=[r['config'] for r in cal['candidates'].values()];incumbent=read(BASE/'outputs/ongoing_jaccard_20261031/February_results.json');onset=read(BASE/'outputs/early_onset_20261030/February_results.json');results={};rows=[];roadrows=[];hrows=[];retained=[]
    for pop in ['new_phase_population','old410']:
        prob,d,m=preds(pop,'final',configs,SEEDS);base=old.score(d['parent'],d,m);inc=incumbent[pop]['candidates']['j075_odds50']['score'];fixed=onset[pop]['candidates']['early_full']['score']['onset_IoU'];res={}
        for name,r in cal['candidates'].items():
            s=old.score(prob[name],d,m,r['threshold']);res[name]=dict(score=s,threshold=r['threshold'],overall_IoU=(s['onset_IoU']+fixed)/2,fixed_onset_IoU=fixed,IoU_vs_R127=old.prior.REF.bootstrap(s,base),IoU_vs_incumbent=old.prior.REF.bootstrap(s,inc),exceeds_current_best=s['onset_IoU']>inc['onset_IoU']);log(stage='FEBRUARY',population=pop,name=name,overall=res[name]['overall_IoU'],**old.compact(s))
        results[pop]=dict(R127=base,incumbent=inc,candidates=res)
        for name,s,t in [('R127',base,.25),('incumbent_j075_odds50',inc,.375)]+[(n,r['score'],r['threshold']) for n,r in res.items()]:
            rows.append(dict(population=pop,model=name,ongoing_IoU=s['onset_IoU'],overall_IoU=(s['onset_IoU']+fixed)/2,FP=s['cell_counts']['FP_cells'],FN=s['cell_counts']['FN_cells'],threshold=t))
            for panel,v in s['per_road'].items():roadrows.append(dict(population=pop,model=name,panel=panel,IoU=v))
        preds0={'R127':d['parent']>=.25,'incumbent':np.load(BASE/'outputs/ongoing_jaccard_20261031'/f'{pop}_j075_odds50_probability.npy')>=.375,**{name:prob[name]>=r['threshold'] for name,r in cal['candidates'].items()}}
        for name,p in preds0.items():
            for h in range(6):
                y=np.asarray(d['y'][:,h],bool);known=np.asarray(d['m'][:,h],bool);hrows.append(dict(population=pop,model=name,minutes=5*(h+1),FP=int((p[:,h]&~y&known).sum()),FN=int((~p[:,h]&y&known).sum())))
    write(OUT/'February_results.json',results)
    for c in configs:
        name=c['name']
        if any(results[p]['candidates'][name]['exceeds_current_best'] for p in results):retained.append(dict(candidate='ongoing_horizon_20261031_'+name,config=c,threshold=cal['candidates'][name]['threshold'],populations={p:results[p]['candidates'][name] for p in results},retained=True))
    write(OUT/'retained_candidates.json',retained)
    if retained:
        f=BASE/'outputs/retained_ongoing_candidates.json';reg=read(f);reg['candidates']+=retained;write(f,reg)
    for name,data in [('comparison',rows),('per_road',roadrows),('per_horizon',hrows)]:pd.DataFrame(data).to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')
    check();write(OUT/'verification.json',dict(onset_unchanged=True,models_and_threshold_hashes_verified=True,February_tuning=False,same_causal_feature_cache=True))
    (OUT/'report.txt').write_text('分时距残差专家：十月选配置，十一月仅报告，一月只选全局门槛，二月冻结比较。\n最好onset固定，真值为可靠观测代理，二月非独立盲测。\n\n'+pd.DataFrame(rows).to_string(index=False)+'\n\n超过此前最好已保留：'+(', '.join(r['candidate'] for r in retained) or '无'),encoding='utf8')
    code=OUT/'code';shutil.copytree(speed.OUT/'code',code,dirs_exist_ok=True);shutil.copy2(__file__,code/Path(__file__).name);shutil.copy2(BASE/'outputs/ongoing_jaccard_core.py',code/'ongoing_jaccard_core.py');shutil.copy2(BASE/'work/ongoing_horizon_tests.txt',OUT/'unit_tests.txt');shutil.copytree(speed.OUT/'final_reference',OUT/'final_reference',dirs_exist_ok=True);shutil.copytree(speed.OUT/'basis',OUT/'basis',dirs_exist_ok=True)
    write(OUT/'manifest.json',{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'})
    with zipfile.ZipFile(OUT.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in OUT.rglob('*'):
            if p.is_file():z.write(p,Path(OUT.name)/p.relative_to(OUT))
    log(stage='COMPLETE',retained=[r['candidate'] for r in retained])
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--fit-panel');ap.add_argument('--phase');ap.add_argument('--capacities');ap.add_argument('--seeds');args=ap.parse_args()
    if args.fit_panel:train_local(args.phase,args.capacities.split(','),[int(v) for v in args.seeds.split(',')],args.fit_panel)
    else:init();screen();confirm();calibrate();evaluate()
