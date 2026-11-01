"""Full-horizon BCE and window Soft-IoU continuation; frozen onset and January-only thresholds."""
import gc, hashlib, json, shutil, time, zipfile,sys,subprocess,concurrent.futures,argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import train_ongoing_speed as speed
from speed_distribution_core import odds_mix,PANELS
from ongoing_jaccard_core import objective,sigmoid
old=speed.old;BASE=old.BASE;OUT=BASE/'outputs/ongoing_jaccard_20261031'
SEEDS=old.SEEDS;START=time.time();read=old.read;write=old.write
CAPACITY={'control':dict(strength=0.),'j025':dict(strength=.25),'j075':dict(strength=.75)}
CONFIGS=[dict(name=f'{c}_{mix}',capacity=c,mix=mix) for c in CAPACITY for mix in ['independent','odds50']]
def log(**kw):print(json.dumps(dict(seconds=round(time.time()-START,1),**kw)),flush=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check():
    for p,h in read(OUT/'frozen_onset_hashes.json').items():assert sha(BASE/p)==h
def init():
    OUT.mkdir(exist_ok=True)
    write(OUT/'protocol.json',dict(date='2026-10-31',method='Full-known-horizon continuation: matched BCE control versus BCE+window Soft-IoU gradients. Start frozen800-tree inverse-density road models;100newtrees lr.02.',
      inputs='Identical274 causal features and selectednodes/IPW from8143windows. All known5-30min horizons replace sampled<=2horizons/node. Missing future cells never enter rows. No future input features.',
      weights='Same inverse-density positive-window factor clip(.1/(q+.01),.5,8),aux factor1,positive/aux mass80/20 perroad,mean1. q from known fitting targets only. Same weights/data across3heads.',
      objective='Weighted BCE plus lambda * sum_window(weight_mass*(1-softIoU)). Exact analytical gradient; positive Fisher-like curvature proxy because true SoftIoU Hessian may be negative. Auxiliary allnegative windows still trained by BCE. lambda0,.25,.75 fixed before development.',
      configs=CONFIGS,capacities=CAPACITY,seeds=SEEDS,rounds=100,fixed_control_ablation='Append control_odds50 to October top2;final3seeds;November confirmation only selectedSoftIoU2,no reselection.',
      October='June-Sep800-tree initialization and known-horizon fitting;one seed6configs(control/j025/j075 x independent/odds50),fixed.25 pick top2IoU with FP/FN<=1.15R127 preferred. R127 fit these dates,anchored development not fully OOS.',
      November='Selected2,three seeds,report only;no reselection. FinalJune-Nov800-tree initialization plus100trees.',
      January='Primary threshold only: maximizeJanuaryIoU on .05.. .75 step.025 without hard FP/FN constraints, as requested higherIoU priority. Report FP/FN<=1.05incumbent and perroadloss<=.03 feasibility and best feasible threshold separately; no February threshold selection.',
      February='All weights/config/Januarythresholds+objective source hash frozen before same2populations;no tuning;retain everypointleader;preserve082245,082186,bestearly_full onset.',
      caveat='Reliable-observation proxy,not official S_queue;February repeated development,not blind.'))
    shutil.copy2(old.OUT/'frozen_onset_hashes.json',OUT/'frozen_onset_hashes.json');shutil.copy2(BASE/'outputs/ongoing_macro_20261031/overall_leaders.json',OUT/'incumbent.json')

def train_local(phase,capacities,seeds,panel):
    for panel in [panel]:
        jobs=[(cap,seed,OUT/f'{phase}_{cap}_{seed}'/f'{panel}.txt') for cap in capacities for seed in seeds if not (OUT/f'{phase}_{cap}_{seed}'/f'{panel}.txt').exists()]
        if not jobs:continue
        source=old.RUN/'fit'/panel;meta=read(source/'complete.json');base=np.load(source/'X.npy',mmap_mode='r');extra=np.load(speed.RUN/'fit'/panel/'extra.npy',mmap_mode='r');yy=np.load(source/'y.npy',mmap_mode='r');mm=np.load(source/'m.npy',mmap_mode='r');month=np.load(source/'month.npy');use=month<203010 if phase=='development' else np.ones(len(month),bool);row,hs=np.nonzero(mm&use[:,None]);x=np.column_stack([base[row],extra[row],(hs+1)*5]).astype('float32');y=yy[row,hs].astype('float64');nodew=np.load(source/'w.npy');density=np.zeros(len(month));positive=np.zeros(len(month),bool);groups=np.zeros(len(month),dtype='int32')
        with np.load(BASE/'outputs/ongoing_analog_20261031/library'/f'{panel}.npz') as z:lib={k:z[k] for k in ['origin','future']}
        bytime={old.prior.old.normtime(t):v for t,v in zip(lib['origin'],lib['future'])}
        for i,win in enumerate(meta['windows']):
            v=bytime[old.prior.old.normtime(win['origin'])];known=np.isfinite(v);a,b=win['start'],win['end'];density[a:b]=((v<=0)&known).sum()/max(int(known.sum()),1);positive[a:b]=win['positive'];groups[a:b]=i
        q=density[row];pos=positive[row];group=groups[row];factor=np.where(pos,np.clip(.1/(q+.01),.5,8.),1.);w=nodew[row].astype('float64')*factor
        for label,mass in [(True,.8),(False,.2)]:
            mask=pos==label;w[mask]*=mass/w[mask].sum()
        w/=w.mean();assert np.isfinite(y).all() and len(y)==int((mm&use[:,None]).sum());log(stage='DATA',phase=phase,panel=panel,all_known_horizon_rows=len(row),positive_rows=int(y.sum()))
        for cap,seed,file in jobs:
            strength=CAPACITY[cap]['strength'];initial=BASE/'outputs/ongoing_macro_20261031'/f'{phase}_inverse_{seed}'/f'{panel}.txt';params=dict(objective='binary' if strength==0 else objective(w,group,strength),learning_rate=.02,num_leaves=63,max_depth=7,min_data_in_leaf=60,lambda_l2=20,max_bin=127,feature_fraction=.9,bagging_fraction=.9,bagging_freq=1,num_threads=8,seed=seed,deterministic=True,force_col_wise=True,verbosity=-1);file.parent.mkdir(exist_ok=True)
            model=lgb.train(params,lgb.Dataset(x,label=y,weight=w),num_boost_round=100,init_model=str(initial));assert model.num_trees()==900;model.save_model(str(file));log(stage='MODEL',phase=phase,head=cap,seed=seed,panel=panel,rows=len(x),initial_sha256=sha(initial));del model;gc.collect()
        del x,y,w;gc.collect()

def train(phase,capacities,seeds):
    write(OUT/'execution_notes.json',dict(concurrent_road_processes=2,threads_per_process=8,model_parameters_unchanged=True))
    panels=[p for p in PANELS if any(not (OUT/f'{phase}_{c}_{seed}'/f'{p}.txt').exists() for c in capacities for seed in seeds)]
    def run(panel):
        subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--fit-panel',panel,'--phase',phase,'--capacities',','.join(capacities),'--seeds',','.join(str(x) for x in seeds)],check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,panels))

def predict(cap,phase,seeds,pop,d,m):
    file=OUT/f'{pop}_{phase}_{cap}_{len(seeds)}seeds.npy'
    if file.exists():return np.load(file)
    result=np.empty(d['y'].shape,dtype='float32')
    for panel in PANELS:
        indices=np.concatenate([np.arange(w['start'],w['end']) for w in m['windows'] if w['panel']==panel]);x=d['X'][indices];acc=np.zeros((len(x),6))
        for seed in seeds:
            model=lgb.Booster(model_file=str(OUT/f'{phase}_{cap}_{seed}'/f'{panel}.txt'))
            for h in range(6):acc[:,h]+=sigmoid(model.predict(np.column_stack([x,np.full(len(x),(h+1)*5)]),raw_score=True,num_threads=8))/len(seeds)
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
    write(OUT/'October_search.json',rows);selected=sorted(rows,key=lambda x:(x['eligible'],x['October']['onset_IoU']),reverse=True)[:2];control=next(r for r in rows if r['config']['name']=='control_odds50');selected=selected+([dict(control,role='fixed matched full-horizon BCE ablation')] if all(r['config']['name']!='control_odds50' for r in selected) else []);write(OUT/'selected_candidates.json',selected)
def confirm():
    configs=[r['config'] for r in read(OUT/'selected_candidates.json') if r['config']['name']!='control_odds50'];train('development',sorted({c['capacity'] for c in configs}),SEEDS);p,d,m=preds('development','development',configs,SEEDS);novm=old.subset(m,'2030-11');rows={}
    for cfg in configs:
        name=cfg['name'];s=old.score(p[name],d,novm);rows[name]=s;log(stage='NOVEMBER',name=name,**old.compact(s))
    write(OUT/'November_confirmation.json',dict(R127=old.score(d['parent'],d,novm),candidates=rows))
def calibrate():
    configs=[r['config'] for r in read(OUT/'selected_candidates.json')];train('final',sorted({c['capacity'] for c in configs}),SEEDS);p,d,m=preds('january','final',configs,SEEDS);r127=old.score(d['parent'],d,m);base=read(BASE/'outputs/ongoing_macro_20261031/January_results.json')['candidates']['inverse_odds50']['January'];rows={}
    for cfg in configs:
        name=cfg['name'];best=None;anybest=None;grid=[]
        for threshold in np.round(np.arange(.05,.7501,.025),3):
            s=old.score(p[name],d,m,float(threshold));c=s['cell_counts'];b=base['cell_counts'];ok=c['FP_cells']<=1.05*b['FP_cells'] and c['FN_cells']<=1.05*b['FN_cells'] and min(s['per_road'][r]-base['per_road'][r] for r in base['per_road'])>=-.03;grid.append(dict(threshold=float(threshold),eligible=bool(ok),**old.compact(s)));item=(float(threshold),s)
            if anybest is None or s['onset_IoU']>anybest[1]['onset_IoU']:anybest=item
            if ok and (best is None or s['onset_IoU']>best[1]['onset_IoU']):best=item
        t,s=anybest;selected_ok=next(g['eligible'] for g in grid if g['threshold']==t);rows[name]=dict(config=cfg,threshold=t,feasible=bool(selected_ok),best_feasible_threshold=best[0] if best else None,threshold_objective='maximum January IoU',January=s,grid=grid);log(stage='JANUARY',name=name,threshold=t,feasible=best is not None,**old.compact(s))
    write(OUT/'January_results.json',dict(R127=r127,incumbent=base,candidates=rows));check();files=list(OUT.glob('final_*/*.txt'))+[OUT/n for n in ['protocol.json','selected_candidates.json','January_results.json','frozen_onset_hashes.json']]+[Path(__file__),BASE/'outputs/ongoing_jaccard_core.py']
    write(OUT/'frozen_before_February.json',{str(p.relative_to(BASE)):sha(p) for p in files})
def evaluate():
    for p,h in read(OUT/'frozen_before_February.json').items():assert sha(BASE/p)==h
    cal=read(OUT/'January_results.json');configs=[r['config'] for r in cal['candidates'].values()];incumbent=read(BASE/'outputs/ongoing_macro_20261031/February_results.json');onset=read(BASE/'outputs/early_onset_20261030/February_results.json');results={};rows=[];roadrows=[];hrows=[];retained=[]
    for pop in ['new_phase_population','old410']:
        prob,d,m=preds(pop,'final',configs,SEEDS);base=old.score(d['parent'],d,m);inc=incumbent[pop]['candidates']['inverse_odds50']['score'];fixed=onset[pop]['candidates']['early_full']['score']['onset_IoU'];res={}
        for name,r in cal['candidates'].items():
            s=old.score(prob[name],d,m,r['threshold']);res[name]=dict(score=s,threshold=r['threshold'],overall_IoU=(s['onset_IoU']+fixed)/2,fixed_onset_IoU=fixed,IoU_vs_R127=old.prior.REF.bootstrap(s,base),IoU_vs_incumbent=old.prior.REF.bootstrap(s,inc),exceeds_current_best=s['onset_IoU']>inc['onset_IoU']);log(stage='FEBRUARY',population=pop,name=name,overall=res[name]['overall_IoU'],**old.compact(s))
        results[pop]=dict(R127=base,incumbent=inc,candidates=res)
        for name,s,t in [('R127',base,.25),('incumbent_inverse_odds50',inc,.35)]+[(n,r['score'],r['threshold']) for n,r in res.items()]:
            rows.append(dict(population=pop,model=name,ongoing_IoU=s['onset_IoU'],overall_IoU=(s['onset_IoU']+fixed)/2,FP=s['cell_counts']['FP_cells'],FN=s['cell_counts']['FN_cells'],threshold=t))
            for panel,v in s['per_road'].items():roadrows.append(dict(population=pop,model=name,panel=panel,IoU=v))
        preds0={'R127':d['parent']>=.25,'incumbent':np.load(BASE/'outputs/ongoing_macro_20261031'/f'{pop}_inverse_odds50_probability.npy')>=.35,**{name:prob[name]>=r['threshold'] for name,r in cal['candidates'].items()}}
        for name,p in preds0.items():
            for h in range(6):
                y=np.asarray(d['y'][:,h],bool);known=np.asarray(d['m'][:,h],bool);hrows.append(dict(population=pop,model=name,minutes=5*(h+1),FP=int((p[:,h]&~y&known).sum()),FN=int((~p[:,h]&y&known).sum())))
    write(OUT/'February_results.json',results)
    for c in configs:
        name=c['name']
        if any(results[p]['candidates'][name]['exceeds_current_best'] for p in results):retained.append(dict(candidate='ongoing_jaccard_20261031_'+name,config=c,threshold=cal['candidates'][name]['threshold'],populations={p:results[p]['candidates'][name] for p in results},retained=True))
    write(OUT/'retained_candidates.json',retained)
    if retained:
        f=BASE/'outputs/retained_ongoing_candidates.json';reg=read(f);reg['candidates']+=retained;write(f,reg)
    for name,data in [('comparison',rows),('per_road',roadrows),('per_horizon',hrows)]:pd.DataFrame(data).to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')
    check();write(OUT/'verification.json',dict(onset_unchanged=True,models_and_threshold_hashes_verified=True,February_tuning=False,same_causal_feature_cache=True))
    (OUT/'report.txt').write_text('完整时距监督与Soft-IoU损失：十月选配置，十一月仅报告，一月只选全局门槛，二月冻结比较。\n最好onset固定，真值为可靠观测代理，二月非独立盲测。\n\n'+pd.DataFrame(rows).to_string(index=False)+'\n\n超过此前最好已保留：'+(', '.join(r['candidate'] for r in retained) or '无'),encoding='utf8')
    code=OUT/'code';shutil.copytree(speed.OUT/'code',code,dirs_exist_ok=True);shutil.copy2(__file__,code/Path(__file__).name);shutil.copy2(BASE/'outputs/ongoing_jaccard_core.py',code/'ongoing_jaccard_core.py');shutil.copy2(BASE/'work/ongoing_jaccard_tests.txt',OUT/'unit_tests.txt');shutil.copytree(speed.OUT/'final_reference',OUT/'final_reference',dirs_exist_ok=True);shutil.copytree(speed.OUT/'basis',OUT/'basis',dirs_exist_ok=True)
    write(OUT/'manifest.json',{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'})
    with zipfile.ZipFile(OUT.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in OUT.rglob('*'):
            if p.is_file():z.write(p,Path(OUT.name)/p.relative_to(OUT))
    log(stage='COMPLETE',retained=[r['candidate'] for r in retained])
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--fit-panel');ap.add_argument('--phase');ap.add_argument('--capacities');ap.add_argument('--seeds');args=ap.parse_args()
    if args.fit_panel:train_local(args.phase,args.capacities.split(','),[int(v) for v in args.seeds.split(',')],args.fit_panel)
    else:init();screen();confirm();calibrate();evaluate()
