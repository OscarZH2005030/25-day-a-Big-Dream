"""第二轮：扩大覆盖、历史因果特征、树/网络对照。二月是重复开发后评估。"""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import json,time,random,copy,traceback,hashlib,sys,gc
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
from task2q.data import PANELS,FEATURE_NAMES,make_sample,load_sample,released_sample,network
from task2q.cache import read_arrays,atomic_json
from task2q.baselines import tree_features,official_persistence,persistence
from task2q.metrics import aggregate_metrics,window_iou
from task2q.model import QueueForecaster,masked_queue_loss
from task2q.smoke import tensor_input
from task2q.submission import validate_queue
from causal_features import enhanced
ROOT=Path(__file__).resolve().parent
OLD=Path(r'C:\Users\18049\Documents\Codex\2026-10-24\trafficflowbench-186-global-road-adaptation\outputs\task2-queue-20261026')
CACHE=OLD.parents[1]/'work/task2-queue-cache'
DATA=Path(r'C:\Users\18049\Documents\Codex\2026-10-22\trafficflowbench-kaggle-rtx-5080-16gb-1\work\competition-data\kaggle_public')
SEED=20261027
def save(name,v):atomic_json(ROOT/name,v)
def progress(phase,**kw):
    v=dict(phase=phase,pid=os.getpid(),utc=pd.Timestamp.now(tz='UTC').isoformat(),**kw);save('progress.json',v);print(json.dumps(v,ensure_ascii=False),flush=True)
@lru_cache(maxsize=6)
def arrays(key):
    with np.load(CACHE/(key+'.npz'),allow_pickle=False) as f:return {k:f[k] for k in f.files}
def sample(e):return make_sample(arrays(e['day_key']),arrays(e['network_key']),e['origin_slot'],e['panel'])
def role_samples(role):
    p=OLD/'datasets'/role/'index.json';m=json.loads(p.read_text());return [load_sample(p,e,CACHE) for e in m['samples']]
def score(ss,ps,threshold=.5,spaced=False):
    rows=[];last={}
    for s,p in sorted(zip(ss,ps),key=lambda a:(a[0]['panel'],a[0]['origin'])):
        if not s['has_future_queue']:continue
        key=(s['panel'],s['condition']);t=pd.Timestamp(s['origin'])
        if spaced and key in last and (t-last[key]).total_seconds()<21600:continue
        last[key]=t
        th=threshold[s['condition']] if isinstance(threshold,list) else threshold
        rows.append(dict(panel=s['panel'],condition=s['condition'],window_id=s['origin'],iou=window_iou(s['y'],p>=th,s['m'])))
    return aggregate_metrics(rows)
@torch.inference_mode()
def neural(model,ss):
    model.eval();return [model(**tensor_input(s,'cuda'))['logits'].sigmoid()[0].cpu().numpy() for s in ss]
def tree(model,ss,fn):return [model.predict(fn(s,s['network']),num_threads=4).astype('float32').reshape(6,-1) for s in ss]
def expanded_index():
    p=ROOT/'expanded-fit.json'
    if p.exists():return json.loads(p.read_text())['samples']
    m=json.loads((OLD/'datasets/fit/index.json').read_text());by_panel={p:[] for p in PANELS};nets={e['panel']:e['network_key'] for e in m['samples']}
    verified=0
    for key,record in m['assets']['days'].items():
        found=read_arrays(CACHE,key);assert found and found[1]['sha256']==record['sha256'];v=found[0];verified+=1
        panel=next(p for p in PANELS if '/'+p+'/' in next(iter(record['metadata']['sources']),'') ) if False else None
        # 日缓存元数据仅含相对源名；按网络源哈希精确定位走廊。
        sources=record['metadata']['sources']
        for p,k in nets.items():
            nr=m['assets']['networks'][k]['metadata']['sources']
            if all(sources.get(a)==b for a,b in nr.items()):panel=p;break
        assert panel is not None
        net=arrays(nets[panel]);speed=v['speed'];q=(speed<=net['threshold'])&np.isfinite(speed)&v['eligible'];eligible=v['eligible']&np.isfinite(speed)&np.isfinite(v['quality'])&(v['quality']>=75)
        for slot in range(12,282,3):
            mask=eligible[slot+1:slot+7]
            if not mask.any():continue
            condition=int(np.any(q[slot-12:slot].sum(0)>=2));positive=bool((q[slot+1:slot+7]&mask).any())
            origin=pd.Timestamp(int(v['times_ns'][slot]),tz='UTC').isoformat()
            by_panel[panel].append(dict(panel=panel,day_key=key,network_key=nets[panel],origin_slot=slot,condition=condition,has_future_queue=positive,origin=origin))
        if verified%100==0:progress('expanding_coverage',days=verified)
    rng=np.random.default_rng(SEED);selected=[];counts={}
    for p,entries in by_panel.items():
        groups={}
        for e in entries:groups.setdefault((e['condition'],e['has_future_queue'],e['origin'][:7]),[]).append(e)
        for g in groups.values():rng.shuffle(g)
        chosen=[]
        while len(chosen)<1024:
            advanced=False
            for k in sorted(groups):
                if groups[k] and len(chosen)<1024:chosen.append(groups[k].pop());advanced=True
            if not advanced:break
        selected.extend(sorted(chosen,key=lambda e:e['origin']));counts[p]=dict(candidates=len(entries),selected=len(chosen),onset_positive=sum(e['condition']==0 and e['has_future_queue'] for e in chosen),ongoing_positive=sum(e['condition']==1 and e['has_future_queue'] for e in chosen),negative=sum(not e['has_future_queue'] for e in chosen))
    save('expanded-fit.json',dict(samples=selected,counts=counts,verified_days=verified,stride_minutes=15));return selected
def main():
    started=time.time();torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED);rng=np.random.default_rng(SEED)
    save('experiment.json',dict(seed=SEED,fit_windows_per_panel=1024,tree_cells_per_window=96,tree_configs=[dict(leaves=31,rounds=400),dict(leaves=63,rounds=600)],neural_updates=1600,neural_initialization='first-round best.pt; fit only, selected December',neural_iou_weight=.6,cpu_threads=4,cpu_temperature=None,official_S_queue=None,february_status='reused development-after-evaluation; not blind',selection='December model/round selection; January global condition thresholds and blend; February report only'))
    entries=expanded_index();tune=role_samples('tune')
    # 多日期/八方向未来扰动不变性；标签/未来质量也不得影响特征。
    checks=[]
    for p in PANELS:
        e=next(e for e in entries if e['panel']==p);v=copy.deepcopy(arrays(e['day_key']));a=sample(e);slot=e['origin_slot']
        for k in ['speed','flow','occupancy','quality','origin_speed','origin_flow','origin_occupancy','origin_quality']:v[k][slot+1:]=999
        v['eligible'][slot+1:]=False;b=make_sample(v,a['network'],slot,p)
        assert np.array_equal(a['x'],b['x']) and np.array_equal(enhanced(a,a['network']),enhanced(b,b['network'])) and a['condition']==b['condition']
        checks.append(dict(panel=p,passed=True))
    save('causality-check.json',dict(passed=True,checks=checks,future_channels_perturbed=True,Task1_offline_runtime_imported=False))
    # 重算首轮结果，检查交接成绩真实性。
    oldtree=lgb.Booster(model_file=str(OLD/'lightgbm.txt'));audit=role_samples('audit');old_probs=tree(oldtree,audit,tree_features)
    old_report=score(audit,old_probs,.25);expected=json.loads((OLD/'proxy-results.json').read_text())['reports']['frozen_blend_proxy']['proxy_IoU'];assert abs(old_report['proxy_IoU']-expected)<1e-12
    save('first-round-reproduction.json',dict(passed=True,expected=expected,recomputed=old_report,warning='February has already been inspected; no longer blind'))
    # 训练矩阵每窗等量抽样：80%正例窗口，20%自然负例总权重，并按panel-condition均衡。
    matrix_root=ROOT.parents[1]/'work/round2-training';matrix_root.mkdir(parents=True,exist_ok=True)
    matrix=matrix_root/'train-X.npy';yp=matrix_root/'train-y.npy';wp=matrix_root/'train-w.npy'
    if not (ROOT/'matrix.json').exists():
        d=enhanced(sample(entries[0]),sample(entries[0])['network']).shape[1];xx=np.lib.format.open_memmap(matrix,mode='w+',dtype='float32',shape=(len(entries)*96,d));yy=np.lib.format.open_memmap(yp,mode='w+',dtype='float32',shape=(len(entries)*96,));ww=np.lib.format.open_memmap(wp,mode='w+',dtype='float32',shape=(len(entries)*96,));cursor=0
        strata={}
        for e in entries:
            k=(e['panel'],e['condition'],e['has_future_queue']);strata[k]=strata.get(k,0)+1
        for i,e in enumerate(entries):
            s=sample(e);ids=np.flatnonzero(s['m'].ravel());ids=rng.choice(ids,min(96,len(ids)),replace=False);k=len(ids);xx[cursor:cursor+k]=enhanced(s,s['network'])[ids];yy[cursor:cursor+k]=s['y'].ravel()[ids];ww[cursor:cursor+k]=(.8 if e['has_future_queue'] else .2)/strata[(e['panel'],e['condition'],e['has_future_queue'])]/k;cursor+=k
            if i%512==0:progress('building_tree_matrix',windows=i,total=len(entries),features=d)
        xx.flush();yy.flush();ww.flush();save('matrix.json',dict(rows=cursor,features=d));del xx,yy,ww
    meta=json.loads((ROOT/'matrix.json').read_text());X=np.load(matrix,mmap_mode='r')[:meta['rows']];y=np.load(yp,mmap_mode='r')[:meta['rows']];w=np.load(wp,mmap_mode='r')[:meta['rows']];w=w/w.mean()
    candidates=[];tree_preds={};round_reports=[]
    for leaves,rounds in [(31,400),(63,600)]:
        path=ROOT/f'tree-{leaves}.txt'
        if not path.exists():
            progress('training_lightgbm',leaves=leaves,total_rounds=rounds,rows=meta['rows'])
            def callback(env):
                if (env.iteration+1)%100==0:env.model.save_model(str(ROOT/f'tree-{leaves}-checkpoint.txt'));progress('training_lightgbm',leaves=leaves,iteration=env.iteration+1,total_rounds=rounds)
            checkpoint=ROOT/f'tree-{leaves}-checkpoint.txt'
            initial=lgb.Booster(model_file=str(checkpoint)) if checkpoint.exists() else None
            completed=initial.current_iteration() if initial else 0
            model=lgb.train(dict(objective='binary',num_leaves=leaves,learning_rate=.035,min_data_in_leaf=80,feature_fraction=.85,lambda_l2=2,num_threads=4,verbosity=-1,seed=SEED,deterministic=True,force_col_wise=True),lgb.Dataset(X,label=y,weight=w),num_boost_round=rounds-completed,init_model=initial,callbacks=[callback]) if completed<rounds else initial
            model.save_model(str(path))
        model=lgb.Booster(model_file=str(path))
        for iteration in [200,400]+([600] if rounds==600 else []):
            ps=[model.predict(enhanced(s,s['network']),num_iteration=iteration,num_threads=4).astype('float32').reshape(6,-1) for s in tune]
            # 十二月允许选择训练轮数，阈值也作为开发诊断；最终阈值在一月冻结。
            for th in [.1,.2,.3,.4,.5]:round_reports.append(dict(leaves=leaves,iteration=iteration,threshold=th,proxy=score(tune,ps,th)['diagnostic_macro_IoU']))
        save('tree-development.json',round_reports)
    best=max(round_reports,key=lambda r:r['proxy']);model=lgb.Booster(model_file=str(ROOT/f"tree-{best['leaves']}.txt"));model.save_model(str(ROOT/'selected-tree.txt'),num_iteration=best['iteration']);tree_model=lgb.Booster(model_file=str(ROOT/'selected-tree.txt'));save('tree-selection.json',best)
    del X,y,w;gc.collect();progress('training_neural',updates=1600)
    # 与树共享扩展窗口池，平衡方向/条件/正负；训练以IoU损失加强onset。
    selected=[]
    for p in PANELS:
        group=[e for e in entries if e['panel']==p];selected.extend(group[::4])
    fit=[sample(e) for e in selected];groups={}
    for s in fit:groups.setdefault((s['panel'],s['condition'],s['has_future_queue']),[]).append(s)
    groups=list(groups.values());cfg=dict(width=128,layers=4,heads=4,dropout=.1);model=QueueForecaster(len(FEATURE_NAMES),**cfg).cuda();ck=torch.load(OLD/'best.pt',weights_only=False);model.load_state_dict(ck['model']);opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4);first=0;bestn=-1;trace=[]
    if (ROOT/'last.pt').exists():
        ck=torch.load(ROOT/'last.pt',weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);first=ck['step'];bestn=ck['best'];trace=ck['trace'];rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
    for step in range(first,1600):
        g=groups[step%len(groups)];s=g[int(rng.integers(len(g)))];model.train();opt.zero_grad(set_to_none=True);output=model(**tensor_input(s,'cuda'));loss=masked_queue_loss(output['logits'],torch.as_tensor(s['y'],device='cuda')[None],torch.as_tensor(s['m'],device='cuda')[None],iou_weight=.6)['total'];loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if (step+1)%100==0:progress('training_neural',step=step+1,total=1600,loss=float(loss.detach()))
        if (step+1)%400==0:
            ps=neural(model,tune);val=max(score(tune,ps,t)['diagnostic_macro_IoU'] for t in [.1,.2,.3,.4,.5]);trace.append(dict(step=step+1,tune_proxy=val));save('neural-development.json',trace)
            if val>bestn:bestn=val;torch.save(dict(model=model.state_dict(),config=cfg,step=step+1),ROOT/'best.pt')
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step+1,best=bestn,trace=trace,rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),ROOT/'last.pt')
    model.load_state_dict(torch.load(ROOT/'best.pt',weights_only=False)['model']);del fit;gc.collect()
    progress('calibration');cal=role_samples('calibrate');pn=neural(model,cal);pt=tree(tree_model,cal,enhanced);po=tree(oldtree,cal,tree_features);candidates=[]
    # 首轮也是候选，避免改进失败却自动替换；仅全局条件阈值，不按二月逐路调参。
    pools={'new_tree':pt,'old_tree':po,'new_neural':pn}
    for alpha in [.25,.5,.75]:pools[f'blend-{alpha}']=[alpha*a+(1-alpha)*b for a,b in zip(pn,pt)]
    for name,ps in pools.items():
        for t0 in [.1,.15,.2,.25,.3,.4,.5]:
            for t1 in [.15,.25,.35,.45,.5]:
                r=score(cal,ps,[t0,t1]);candidates.append(dict(model=name,thresholds=[t0,t1],proxy=r['diagnostic_macro_IoU']))
    winner=max(candidates,key=lambda r:r['proxy']);save('frozen-calibration.json',dict(selected=winner,candidates=candidates,selection_month='January',february='previously inspected first round; report only this round',hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'selected-tree.txt',ROOT/'best.pt']}))
    progress('evaluating');pn=neural(model,audit);pt=tree(tree_model,audit,enhanced)
    def chosen(n,t,o):
        name=winner['model']
        if name=='new_tree':return t
        if name=='old_tree':return o
        if name=='new_neural':return n
        alpha=float(name.split('-')[1]);return [alpha*a+(1-alpha)*b for a,b in zip(n,t)]
    final=chosen(pn,pt,old_probs);reports={'official_persistence':score(audit,[official_persistence(s,s['network']) for s in audit]),'carry_T':score(audit,[persistence(s,s['network'],True) for s in audit]),'first_round_frozen':old_report,'new_tree_fixed05':score(audit,pt),'new_neural_fixed05':score(audit,pn),'selected':score(audit,final,winner['thresholds']),'selected_360min_spacing_diagnostic':score(audit,final,winner['thresholds'],True)}
    save('proxy-results.json',dict(official_S_queue=None,reports=reports,selected=winner,seconds=time.time()-started,february_status='reused development-after-evaluation',cpu_temperature=None))
    # 官方历史与代理历史/condition对齐核验；真实condition优先使用released标签。
    checks=[]
    for panel in PANELS:
        index=pd.read_csv(DATA/'task2'/panel/'train/window_index.csv');row=index.iloc[0];released=released_sample(DATA,panel,'train',row.window_id)
        from task2q.data import day
        origin=pd.Timestamp(released['origin']);v=day(DATA,panel,origin.strftime('%Y-%m-%d'),CACHE);proxy=make_sample(v,released['network'],origin.hour*12+origin.minute//5,panel)
        checks.append(dict(panel=panel,window_id=row.window_id,history_max_abs_diff=float(abs(released['x'][:12]-proxy['x'][:12]).max()),condition_match=released['condition']==proxy['condition'],T_max_abs_diff=float(abs(released['x'][12]-proxy['x'][12]).max())))
    save('proxy-official-alignment.json',dict(checks=checks,proxy_uses_noisy_future=True,official_uses_latent_truth=True,proxy_origins_30min_first_round=True,official_minimum_spacing_minutes=360,proxy_primary_is_not_exact_official_distribution=True))
    progress('predicting_official');checks={}
    for split in ['validation','private']:
        frames=[]
        for panel in PANELS:
            directory=DATA/'task2'/panel/split;index=pd.read_csv(directory/'window_index.csv');template=pd.read_csv(directory/'sample_submission_queue.csv');parts=[]
            for row in index.itertuples():
                s=released_sample(DATA,panel,split,row.window_id);p=chosen(neural(model,[s]),tree(tree_model,[s],enhanced),tree(oldtree,[s],tree_features))[0];keys=template[template.window_id==row.window_id].copy();ii={l:i for i,l in enumerate(s['network']['link_ids'])};h=((pd.to_datetime(keys.timestamp,utc=True)-pd.Timestamp(s['origin'])).dt.total_seconds()/300).to_numpy(dtype=int)-1;keys['queue_pred']=(p[h,keys.link_id.map(ii).to_numpy()]>=winner['thresholds'][s['condition']]).astype(int);parts.append(keys)
            result=pd.concat(parts,ignore_index=True);checks[split+'/'+panel]=validate_queue(result,template);result['panel']=panel;frames.append(result)
        result=pd.concat(frames,ignore_index=True);assert len(result)==87000;result.to_csv(ROOT/f'queue-{split}.csv',index=False)
    save('final-verification.json',dict(passed=True,rows=174000,checks=checks,causal_checks_passed=True,first_round_reproduced=True,official_S_queue=None,selected=winner));progress('complete',proxy=reports['selected']['proxy_IoU'],seconds=time.time()-started)
if __name__=='__main__':
    # 操作系统独占锁可自动随进程退出释放，避免并发重复训练。
    import msvcrt
    lock=open(ROOT/'run.lock','a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise SystemExit('已有本轮训练进程，拒绝重复启动')
    try:main()
    except Exception:save('failure.json',dict(traceback=traceback.format_exc()));progress('failed');raise
