"""Road-specialist capacity experiment, frozen onset and January-only thresholds."""
import gc, hashlib, json, shutil, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import train_ongoing_speed as speed
from speed_distribution_core import odds_mix,PANELS
old=speed.old;BASE=old.BASE;OUT=BASE/'outputs/ongoing_roads_20261031'
SEEDS=old.SEEDS;START=time.time();read=old.read;write=old.write
CAPACITY={'compact':dict(rounds=400,num_leaves=31,max_depth=6,min_data_in_leaf=100),
          'larger':dict(rounds=600,num_leaves=63,max_depth=7,min_data_in_leaf=60)}
CONFIGS=[dict(name=f'{c}_{mix}',capacity=c,mix=mix) for c in CAPACITY for mix in ['independent','odds50']]
def log(**kw):print(json.dumps(dict(seconds=round(time.time()-START,1),**kw)),flush=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check():
    for p,h in read(OUT/'frozen_onset_hashes.json').items():assert sha(BASE/p)==h
def init():
    OUT.mkdir(exist_ok=True)
    write(OUT/'protocol.json',dict(date='2026-10-31',method='Separate shared-horizon binary specialist per road, compared with global rich-field head; two predeclared capacities.',
      features='Identical274 causal rich-field features, no new feature selection. Same8143 June-Nov windows, node/horizon sampling, IPW and80/20positive/auxiliary mass as prior speed experiment.',
      capacities=CAPACITY,seeds=SEEDS,configs=CONFIGS,
      October='June-Sep weights and end-Sep reference; one seed, fixed.25 select top2 of4 by IoU with FP/FN<=1.15R127 preferred. R127 in-sample on development dates.',
      November='Selected2 with3 seeds, no reselection. Final weights June-Nov with3 fixed seeds.',
      January='Only one global threshold per candidate, .05.. .75 step.025, maximizeIoU subject FP/FN<=1.05R127 and no road IoU loss>.03; else diagnostic maximum. No road-specific January gates or thresholds.',
      February='After hash freeze, same2 populations, no tuning. Preserve .817962444, .818656782 and best early_full onset. Retain every point-estimate improvement, report uncertainty.',
      labels='Reliable-observation proxy, hidden official S_queue unavailable; February reused development, not blind.'))
    shutil.copy2(old.OUT/'frozen_onset_hashes.json',OUT/'frozen_onset_hashes.json')
    shutil.copy2(speed.OUT/'overall_leaders.json',OUT/'incumbent.json')

def train(phase,capacities,seeds):
    names=read(speed.OUT/'feature_names.json');extra=names['extra'];base=len(names['base']);jobs=[]
    for cap in capacities:
        for seed in seeds:
            for panel in PANELS:
                file=OUT/f'{phase}_{cap}_{seed}'/f'{panel}.txt'
                if not file.exists():jobs.append((cap,seed,panel,file))
    if not jobs:return
    data=speed.training(phase)
    for panel in PANELS:
        paneljobs=[j for j in jobs if j[2]==panel]
        if not paneljobs:continue
        ids=data['X'][:,base+extra.index('panel_'+panel)]>.5;x=data['X'][ids];y=data['y'][ids];w=data['w'][ids];w=w/w.mean()
        for cap,seed,_,file in paneljobs:
            file.parent.mkdir(exist_ok=True);c=CAPACITY[cap];params=dict(objective='binary',learning_rate=.04,lambda_l2=20,max_bin=127,feature_fraction=.9,bagging_fraction=.9,bagging_freq=1,num_threads=8,seed=seed,deterministic=True,force_col_wise=True,verbosity=-1,**{k:v for k,v in c.items() if k!='rounds'})
            def cb(env):
                if (env.iteration+1)%200==0:log(stage='FIT',phase=phase,capacity=cap,seed=seed,panel=panel,round=env.iteration+1)
            model=lgb.train(params,lgb.Dataset(x,label=y,weight=w),num_boost_round=c['rounds'],callbacks=[cb]);model.save_model(str(file));log(stage='MODEL',phase=phase,capacity=cap,seed=seed,panel=panel,rows=len(x));del model;gc.collect()
        del x,y,w;gc.collect()
    del data;gc.collect()

def predict(cap,phase,seeds,pop,d,m):
    file=OUT/f'{pop}_{phase}_{cap}_{len(seeds)}seeds.npy'
    if file.exists():return np.load(file)
    result=np.empty(d['y'].shape,dtype='float32')
    for panel in PANELS:
        indices=np.concatenate([np.arange(w['start'],w['end']) for w in m['windows'] if w['panel']==panel]);x=d['X'][indices];acc=np.zeros((len(x),6))
        for seed in seeds:
            model=lgb.Booster(model_file=str(OUT/f'{phase}_{cap}_{seed}'/f'{panel}.txt'))
            for h in range(6):acc[:,h]+=model.predict(np.column_stack([x,np.full(len(x),(h+1)*5)]),num_threads=8)/len(seeds)
        result[indices]=acc
    assert np.isfinite(result).all();np.save(file,result);return result
def preds(pop,phase,configs,seeds):
    d,m=speed.query(pop);cache={};prob={}
    for c in configs:
        cap=c['capacity']
        if cap not in cache:cache[cap]=predict(cap,phase,seeds,pop,d,m)
        p=cache[cap];prob[c['name']]=p if c['mix']=='independent' else odds_mix(d['parent'],p)
        np.save(OUT/f"{pop}_{c['name']}_probability.npy",prob[c['name']])
    return prob,d,m
def screen():
    train('development',list(CAPACITY),SEEDS[:1]);p,d,m=preds('development','development',CONFIGS,SEEDS[:1]);octm=old.subset(m,'2030-10');base=old.score(d['parent'],d,octm);rows=[]
    for cfg in CONFIGS:
        s=old.score(p[cfg['name']],d,octm);c=s['cell_counts'];b=base['cell_counts'];ok=c['FP_cells']<=1.15*b['FP_cells'] and c['FN_cells']<=1.15*b['FN_cells'];rows.append(dict(config=cfg,eligible=bool(ok),October=s));log(stage='OCTOBER',name=cfg['name'],eligible=bool(ok),**old.compact(s))
    write(OUT/'October_search.json',rows);selected=sorted(rows,key=lambda x:(x['eligible'],x['October']['onset_IoU']),reverse=True)[:2];write(OUT/'selected_candidates.json',selected)
def confirm():
    configs=[r['config'] for r in read(OUT/'selected_candidates.json')];train('development',sorted({c['capacity'] for c in configs}),SEEDS);p,d,m=preds('development','development',configs,SEEDS);novm=old.subset(m,'2030-11');rows={}
    for cfg in configs:
        name=cfg['name'];s=old.score(p[name],d,novm);rows[name]=s;log(stage='NOVEMBER',name=name,**old.compact(s))
    write(OUT/'November_confirmation.json',dict(R127=old.score(d['parent'],d,novm),candidates=rows))
def calibrate():
    configs=[r['config'] for r in read(OUT/'selected_candidates.json')];train('final',sorted({c['capacity'] for c in configs}),SEEDS);p,d,m=preds('january','final',configs,SEEDS);base=old.score(d['parent'],d,m);rows={}
    for cfg in configs:
        name=cfg['name'];best=None;anybest=None;grid=[]
        for threshold in np.round(np.arange(.05,.7501,.025),3):
            s=old.score(p[name],d,m,float(threshold));c=s['cell_counts'];b=base['cell_counts'];ok=c['FP_cells']<=1.05*b['FP_cells'] and c['FN_cells']<=1.05*b['FN_cells'] and min(s['per_road'][r]-base['per_road'][r] for r in base['per_road'])>=-.03;grid.append(dict(threshold=float(threshold),eligible=bool(ok),**old.compact(s)));item=(float(threshold),s)
            if anybest is None or s['onset_IoU']>anybest[1]['onset_IoU']:anybest=item
            if ok and (best is None or s['onset_IoU']>best[1]['onset_IoU']):best=item
        t,s=best or anybest;rows[name]=dict(config=cfg,threshold=t,feasible=best is not None,January=s,grid=grid);log(stage='JANUARY',name=name,threshold=t,feasible=best is not None,**old.compact(s))
    write(OUT/'January_results.json',dict(R127=base,candidates=rows));check();files=list(OUT.glob('final_*/*.txt'))+[OUT/n for n in ['protocol.json','selected_candidates.json','January_results.json','frozen_onset_hashes.json']]+[Path(__file__)]
    write(OUT/'frozen_before_February.json',{str(p.relative_to(BASE)):sha(p) for p in files})
def evaluate():
    for p,h in read(OUT/'frozen_before_February.json').items():assert sha(BASE/p)==h
    cal=read(OUT/'January_results.json');configs=[r['config'] for r in cal['candidates'].values()];incumbent=read(speed.OUT/'February_results.json');onset=read(BASE/'outputs/early_onset_20261030/February_results.json');results={};rows=[];roadrows=[];hrows=[];retained=[]
    for pop in ['new_phase_population','old410']:
        prob,d,m=preds(pop,'final',configs,SEEDS);base=old.score(d['parent'],d,m);inc=incumbent[pop]['candidates']['binary_fields_odds50']['score'];fixed=onset[pop]['candidates']['early_full']['score']['onset_IoU'];res={}
        for name,r in cal['candidates'].items():
            s=old.score(prob[name],d,m,r['threshold']);res[name]=dict(score=s,threshold=r['threshold'],overall_IoU=(s['onset_IoU']+fixed)/2,fixed_onset_IoU=fixed,IoU_vs_R127=old.prior.REF.bootstrap(s,base),IoU_vs_incumbent=old.prior.REF.bootstrap(s,inc),exceeds_current_best=s['onset_IoU']>inc['onset_IoU']);log(stage='FEBRUARY',population=pop,name=name,overall=res[name]['overall_IoU'],**old.compact(s))
        results[pop]=dict(R127=base,incumbent=inc,candidates=res)
        for name,s,t in [('R127',base,.25),('incumbent_binary_fields_odds50',inc,.3)]+[(n,r['score'],r['threshold']) for n,r in res.items()]:
            rows.append(dict(population=pop,model=name,ongoing_IoU=s['onset_IoU'],overall_IoU=(s['onset_IoU']+fixed)/2,FP=s['cell_counts']['FP_cells'],FN=s['cell_counts']['FN_cells'],threshold=t))
            for panel,v in s['per_road'].items():roadrows.append(dict(population=pop,model=name,panel=panel,IoU=v))
        preds0={'R127':d['parent']>=.25,'incumbent':np.load(speed.OUT/f'{pop}_binary_fields_odds50_probability.npy')>=.3,**{name:prob[name]>=r['threshold'] for name,r in cal['candidates'].items()}}
        for name,p in preds0.items():
            for h in range(6):
                y=np.asarray(d['y'][:,h],bool);known=np.asarray(d['m'][:,h],bool);hrows.append(dict(population=pop,model=name,minutes=5*(h+1),FP=int((p[:,h]&~y&known).sum()),FN=int((~p[:,h]&y&known).sum())))
    write(OUT/'February_results.json',results)
    for c in configs:
        name=c['name']
        if any(results[p]['candidates'][name]['exceeds_current_best'] for p in results):retained.append(dict(candidate='ongoing_roads_20261031_'+name,config=c,threshold=cal['candidates'][name]['threshold'],populations={p:results[p]['candidates'][name] for p in results},retained=True))
    write(OUT/'retained_candidates.json',retained)
    if retained:
        f=BASE/'outputs/retained_ongoing_candidates.json';reg=read(f);reg['candidates']+=retained;write(f,reg)
    for name,data in [('comparison',rows),('per_road',roadrows),('per_horizon',hrows)]:pd.DataFrame(data).to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')
    check();write(OUT/'verification.json',dict(onset_unchanged=True,models_and_threshold_hashes_verified=True,February_tuning=False,same_causal_feature_cache=True))
    (OUT/'report.txt').write_text('按道路分别训练：十月选配置，十一月仅报告，一月只选全局门槛，二月冻结比较。\n最好onset固定，真值为可靠观测代理，二月非独立盲测。\n\n'+pd.DataFrame(rows).to_string(index=False)+'\n\n超过此前最好已保留：'+(', '.join(r['candidate'] for r in retained) or '无'),encoding='utf8')
    code=OUT/'code';shutil.copytree(speed.OUT/'code',code,dirs_exist_ok=True);shutil.copy2(__file__,code/Path(__file__).name);shutil.copytree(speed.OUT/'final_reference',OUT/'final_reference',dirs_exist_ok=True);shutil.copytree(speed.OUT/'basis',OUT/'basis',dirs_exist_ok=True)
    write(OUT/'manifest.json',{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'})
    with zipfile.ZipFile(OUT.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in OUT.rglob('*'):
            if p.is_file():z.write(p,Path(OUT.name)/p.relative_to(OUT))
    log(stage='COMPLETE',retained=[r['candidate'] for r in retained])
if __name__=='__main__':init();screen();confirm();calibrate();evaluate()
