"""Execute registered matched-severity training and January-only mixture selection."""
import common as d
import augmentation_fit as a
import pseudo_data as g
import severe_masks as stress
import numpy as np,pandas as pd,lightgbm as lgb,json,msvcrt,traceback
from datetime import datetime,timezone
O=d.OUT/'severe_augmentation';PARENT=d.OUT/'multimonth_candidate'

def weighted(groups):
 # Equal month/regime groups within source; original receives double source mass.
 w=a.weights(groups)*np.array([2. if s.startswith('original/') else 1. for s in groups]);return w*len(w)/w.sum()

def load_case(panel,phase,condition,date,records):
 folder=d.WORK/'severe_augmentation_data'/phase/panel/condition/date;r=records[(panel,phase,condition,date)];fp=folder/'features61.npz';yp=folder/'public_train_supervision.parquet';assert d.sha(fp)==r['features_sha256'] and d.sha(yp)==r['supervision_sha256'];z=np.load(fp);assert str(z['keys_sha256'])==d.sha(folder/'auxiliary_keys.csv');f=pd.read_parquet(yp);return dict(date=date,frame=f),z['X']

def train_arrays(cases):
 X=np.concatenate([xx for kind,it,xx in cases]);Y=np.concatenate([(it['frame'][['true_v','true_q_lane']].to_numpy()-it['frame'][['speed_kmh','q_lane']].to_numpy())/[25,600] for kind,it,xx in cases]);groups=np.concatenate([np.array([kind+'/'+it['date'][:7]+'/'+reg for reg in it['frame'].mask_regime]) for kind,it,xx in cases]);return X,Y,groups

def mixture(frame,X,parent,new,parent_alpha,gamma):
 before=a.correct(frame,X,parent,parent_alpha)
 if gamma==0:return before
 after=a.correct(frame,X,new,1.);return before+gamma*(after-before)

def main():
 protocol=json.loads((O/'protocol.json').read_text());d.save(O/'fit_source.json',dict(source_sha256=d.sha(__file__),protocol_sha256=d.sha(O/'protocol.json'),prepared_cases_sha256=d.sha(O/'prepared_cases.csv')));table=pd.read_csv(O/'prepared_cases.csv');records={(r['panel'],r['phase'],r['condition'],r['date']):r for r in table.to_dict('records')};parentpolicy=json.loads((PARENT/'policy.json').read_text())['policy'];policies={};devrows=[];newmodels={}
 for panel in d.night.predictor.load_routes():
  d.check_deadline();d.state(status='running',stage='severe_augmentation_january_fit',panel=panel,next='Fit registered50/25/25 loss and January-only blending; all policies frozen before reserved February seed')
  parent=[lgb.Booster(model_file=str(PARENT/'models'/panel/n)) for n in ['speed.txt','flow_per_lane.txt']];pa=parentpolicy[panel]['alpha'];train=[('original',*a.original(panel,date)) for date in protocol['original_training_dates']];dev=[('original',*a.original(panel,date)) for date in protocol['original_development_dates']]
  for condition in protocol['conditions']:
   train.extend((condition,*load_case(panel,'train',condition,date,records)) for date in protocol['plans'][panel]['train']);dev.extend((condition,*load_case(panel,'development',condition,date,records)) for date in protocol['plans'][panel]['development'])
  X,Y,groups=train_arrays(train);V,T,gv=train_arrays(dev);models=[]
  for channel,name in enumerate(['speed.txt','flow_per_lane.txt']):
   d.check_deadline();path=O/'models'/panel/name;path.parent.mkdir(parents=True,exist_ok=True)
   if path.exists():m=lgb.Booster(model_file=str(path))
   else:
    params={k:v for k,v in protocol['model'].items() if k not in ['max_trees','patience']};params.update(objective='regression',metric='rmse',seed=protocol['model_seed'],num_threads=4,verbosity=-1,deterministic=True,force_col_wise=True);m=lgb.train(params,lgb.Dataset(X,label=Y[:,channel],weight=weighted(groups)),num_boost_round=protocol['model']['max_trees'],valid_sets=[lgb.Dataset(V,label=T[:,channel],weight=weighted(gv))],callbacks=[lgb.early_stopping(protocol['model']['patience'],verbose=False)]);m.save_model(str(path))
   models.append(m)
  newmodels[panel]=models;ref={};best=dict(gamma=0.,objective=None);candidates={}
  for gamma in [0.,.25,.5,1.]:
   scores={}
   for condition in ['original']+protocol['conditions']:
    subset=[(it,xx) for kind,it,xx in dev if kind==condition];scores[condition]=d.night.metrics([it for it,xx in subset],[mixture(it['frame'],xx,parent,models,pa,gamma) for it,xx in subset]);devrows.append(dict(panel=panel,condition=condition,gamma=gamma,**scores[condition]))
   if gamma==0:ref=scores;best['objective']=float(np.mean([s['S_state'] for s in ref.values()]))
   eligible=bool(scores['original']['S_state']>=ref['original']['S_state']-.0001 and all(scores[c]['station_target_deltaN_RMSE']<=ref[c]['station_target_deltaN_RMSE']*1.01 for c in scores));objective=float(np.mean([s['S_state'] for s in scores.values()]));candidates[str(gamma)]=dict(eligible=eligible,objective=objective)
   if eligible and objective>best['objective']:best=dict(gamma=gamma,objective=objective)
  policies[panel]=dict(**best,parent_alpha=pa,iterations=[m.current_iteration() for m in models],January_choices=candidates);d.save(O/'policy_partial.json',policies);pd.DataFrame(devrows).to_csv(O/'january_by_panel.csv',index=False);print('SEVERE AUG FROZEN',panel,best,flush=True)
 d.save(O/'frozen_policy.json',dict(policy=policies,frozen_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=d.sha(O/'protocol.json'),fit_source_sha256=d.sha(__file__),model_sha256={str(p.relative_to(d.ROOT)):d.sha(p) for p in (O/'models').rglob('*.txt')}))
 if all(p['gamma']==0 for p in policies.values()):
  d.save(O/'evaluation_skipped.json',dict(reason='All January gates retained parent; unchanged evaluation skipped',new_score=None));d.state(status='stage_complete',stage='severe_augmentation_complete',next='Matched severity augmentation failed January safety gates; preserve all models and negative results.');return
 rows=[];daily=[]
 for panel in parentpolicy:
  d.check_deadline();d.state(status='running',stage='severe_augmentation_original_evaluation',panel=panel,next='Frozen mixture original February development evaluation, no retuning')
  parent=[lgb.Booster(model_file=str(PARENT/'models'/panel/n)) for n in ['speed.txt','flow_per_lane.txt']];p=policies[panel];items=[];vv={n:[] for n in ['parent','selected']}
  for i in range(1,29):
   it,X=a.original(panel,f'2031-02-{i:02}');items.append(it)
   for name,gamma in [('parent',0.),('selected',p['gamma'])]:value=mixture(it['frame'],X,parent,newmodels[panel],p['parent_alpha'],gamma);vv[name].append(value);daily.append(dict(panel=panel,date=it['date'],variant=name,**d.night.metrics([it],[value])))
  rows.extend(dict(panel=panel,variant=n,gamma=p['gamma'],**d.night.metrics(items,v)) for n,v in vv.items());pd.DataFrame(rows).to_csv(O/'february_original_by_panel.csv',index=False);pd.DataFrame(daily).to_csv(O/'february_original_by_date.csv',index=False)
 pd.DataFrame(rows).groupby('variant')[['S_state','speed_RMSE','flow_per_lane_RMSE','station_target_deltaN_RMSE']].mean().to_csv(O/'february_original_summary.csv');rows=[];oldselector=g.f.mask_selection;oldseed=stress.SEED;g.f.mask_selection=stress.selector;stress.SEED=protocol['evaluation_seed']
 try:
  for panel,route in d.night.predictor.load_routes().items():
   p=policies[panel];parent=[lgb.Booster(model_file=str(PARENT/'models'/panel/n)) for n in ['speed.txt','flow_per_lane.txt']];keys=pd.read_csv(d.DATA/'task1'/panel/'train/sample_submission_state.csv',usecols=d.KEYS,dtype=str)
   for condition in protocol['conditions']:
    items=[];vv={n:[] for n in ['baseline','parent','selected']}
    for date in protocol['plans'][panel]['evaluation']:
     d.check_deadline();d.state(status='running',stage='severe_augmentation_reserved_evaluation',panel=panel,condition=condition,date=date,next='Reserved new seed, fixed January choices, no road/model reselection')
     callbacks=dict(parent=lambda f,X:mixture(f,X,parent,newmodels[panel],p['parent_alpha'],0.),selected=lambda f,X:mixture(f,X,parent,newmodels[panel],p['parent_alpha'],p['gamma']));it,_,outputs=g.generate(panel,date,condition,protocol['evaluation_seed'],d.WORK/'severe_augmentation_evaluation'/panel/condition/date,route,keys,callbacks);items.append(it)
     for name in vv:vv[name].append(outputs[name])
    rows.extend(dict(panel=panel,condition=condition,variant=name,gamma=p['gamma'],**d.night.metrics(items,v)) for name,v in vv.items());pd.DataFrame(rows).to_csv(O/'february_reserved_by_panel.csv',index=False);print('SEVERE AUG EVAL',panel,condition,flush=True)
 finally:g.f.mask_selection=oldselector;stress.SEED=oldseed
 pd.DataFrame(rows).groupby(['condition','variant'])[['S_state','speed_RMSE','flow_per_lane_RMSE','station_target_deltaN_RMSE']].mean().to_csv(O/'february_reserved_summary.csv');d.state(status='stage_complete',stage='severe_augmentation_complete',next='Assess frozen original/severe transfer and all negative roads. No official claim, no Kaggle uploads; preserve immutable parent and context.')

if __name__=='__main__':
 lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 except Exception as e:d.state(status='failed_or_cutoff',stage='severe_augmentation_fit',error=repr(e),next='Repair registered training/evaluation execution without changing selection rules, targets or source models');traceback.print_exc();raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
