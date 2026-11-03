"""January missingness-aware selection of already trained, immutable models."""
import common as d
import augmentation_fit as a
from pseudo_data import generate
import json,msvcrt,traceback
import numpy as np,pandas as pd,lightgbm as lgb
from datetime import datetime,timezone

O=d.OUT/'missingness_selection'
CONDITIONS=['point10','block10']

def candidates(panel):
 rp,rm=a.load_reference(panel);result={'context':(rp['alpha'],rm)}
 for recipe in ['balanced_original','augmented_balanced']:
  models=[lgb.Booster(model_file=str(a.O/'models'/panel/recipe/name)) for name in ['speed.txt','flow_per_lane.txt']]
  for alpha in [.25,.5,1.]:result[recipe+'_'+str(alpha)]=(alpha,models)
 return result

def score_cases(panel,route,keys,dates,seed,stage,models):
 rows=[]
 for condition in CONDITIONS:
  items=[];outputs={name:[] for name in ['baseline']+list(models)}
  for date in dates:
   d.check_deadline();d.state(status='running',stage='missingness_'+stage,panel=panel,condition=condition,date=date,next='January-only selection; freeze all roads before fresh February auxiliary evaluation')
   folder=d.WORK/'missingness_selection'/stage/panel/condition/date
   callbacks={name:(lambda p,X,alpha=alpha,mm=mm:a.correct(p,X,mm,alpha)) for name,(alpha,mm) in models.items()}
   item,_,vv=generate(panel,date,condition,seed,folder,route,keys,callbacks);items.append(item)
   for name in outputs:outputs[name].append(vv[name])
  rows.extend(dict(panel=panel,condition=condition,variant=name,**d.night.metrics(items,pp)) for name,pp in outputs.items())
  print('MISSINGNESS',stage,panel,condition,flush=True)
 return rows

def main():
 routes=d.night.predictor.load_routes();plans={};templates={}
 for panel in routes:
  k=pd.read_csv(d.DATA/'task1'/panel/'train/sample_submission_state.csv',usecols=d.KEYS,dtype=str);templates[panel]=k;v=k.assign(date=k.timestamp.str[:10]);plans[panel]={}
  for name,lo,hi in [('january','2031-01-21','2031-01-31'),('february','2031-02-03','2031-02-27')]:
   vv=v[(v.date>=lo)&(v.date<=hi)];plans[panel][name]=sorted(vv.groupby('mask_regime').date.apply(lambda z:sorted(set(z))[len(set(z))//2]).tolist())
 protocol=dict(registered_utc=datetime.now(timezone.utc).isoformat(),hypothesis='Original-mask January selection may suppress models robust to extra missingness; select on matched January auxiliary conditions with original-mask safety gates.',plans=plans,january_seed=2026110221,february_seed=2026110222,training='None; immutable existing January-trained balanced-original and augmented-balanced models',selection='Maximize equal mean January S_state across original, point10, block10. Original S_state must be >= context minus 0.0001; each of three accumulation RMSEs <=1.01 times context. Context fallback, no February retuning.',source_sha256=d.sha(__file__),generator_sha256=d.sha(d.WORK/'pseudo_data.py'),model_sha256={str(p.relative_to(d.ROOT)):d.sha(p) for p in (a.O/'models').rglob('*.txt')},caveat='Fresh masks on exposed public train, not whole-system blind; not official Task3 or leaderboard improvement. No Kaggle uploads; Task4 baseline frozen.')
 if (O/'protocol.json').exists():
  old=json.loads((O/'protocol.json').read_text());protocol['registered_utc']=old['registered_utc'];assert old==protocol
 else:d.save(O/'protocol.json',protocol)
 policies={};devrows=[];decisions=[]
 for panel,route in routes.items():
  models=candidates(panel);rows=score_cases(panel,route,templates[panel],plans[panel]['january'],protocol['january_seed'],'january',models)
  dev=[a.original(panel,f'2031-01-{i:02d}') for i in range(21,32)];items=[it for it,X in dev]
  for name,(alpha,mm) in models.items():rows.append(dict(panel=panel,condition='original',variant=name,**d.night.metrics(items,[a.correct(it['frame'],X,mm,alpha) for it,X in dev])))
  df=pd.DataFrame(rows);ref=df[df.variant=='context'].set_index('condition');best='context';best_score=ref.S_state.mean()
  for name in models:
   rr=df[df.variant==name].set_index('condition').loc[ref.index];eligible=bool(rr.loc['original','S_state']>=ref.loc['original','S_state']-.0001 and (rr.station_target_deltaN_RMSE<=1.01*ref.station_target_deltaN_RMSE).all());objective=float(rr.S_state.mean());decisions.append(dict(panel=panel,variant=name,eligible=eligible,objective=objective))
   if eligible and objective>best_score:best=name;best_score=objective
  policies[panel]=dict(variant=best,January_objective=best_score);devrows.extend(rows);pd.DataFrame(devrows).to_csv(O/'january_by_panel.csv',index=False);pd.DataFrame(decisions).to_csv(O/'decisions.csv',index=False);d.save(O/'policy_partial.json',policies)
 d.save(O/'frozen_policy.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),policy=policies,protocol_sha256=d.sha(O/'protocol.json'),note='All routes frozen before new seed February evaluation; no retuning'))
 rows=[];prior=json.loads((a.O/'frozen_policy.json').read_text())['policy']
 for panel,route in routes.items():
  allmodels=candidates(panel);models={'context':allmodels['context'],'selected':allmodels[policies[panel]['variant']]};pp,mm=a.load_selected(panel,prior[panel]);models['previous_selection']=(pp['alpha'],mm)
  rows.extend(score_cases(panel,route,templates[panel],plans[panel]['february'],protocol['february_seed'],'february',models));pd.DataFrame(rows).to_csv(O/'february_by_panel.csv',index=False)
 pd.DataFrame(rows).groupby(['condition','variant'])[['S_state','speed_RMSE','flow_per_lane_RMSE','station_target_deltaN_RMSE']].mean().to_csv(O/'february_summary.csv')
 assert protocol['model_sha256']=={str(p.relative_to(d.ROOT)):d.sha(p) for p in (a.O/'models').rglob('*.txt')}
 d.state(status='stage_complete',stage='missingness_selection_complete',next='Inspect January safety gates and fresh auxiliary transfer without retuning; preserve failures and official baseline.')

if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True);lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 except Exception as e:d.state(status='failed_or_cutoff',stage='missingness_selection',error=repr(e),next='Repair execution without changing protocol or frozen choices');traceback.print_exc();raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
