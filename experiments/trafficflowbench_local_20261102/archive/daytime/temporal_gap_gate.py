"""Global January gate on nearest observed temporal distance; frozen ensemble."""
import common as d
import augmentation_fit as a
import severe_seed_ensemble as ens
import severe_augmentation_fit as fit
import numpy as np,pandas as pd,lightgbm as lgb,json,msvcrt,traceback
from datetime import datetime,timezone
O=d.OUT/'temporal_gap_gate';GATES={'unconditional':None,**{f'floor{f}_ticks{t}':(f,t) for f in [0.,.5] for t in [3,6,12]}}

def gate(X,spec):
 if spec is None:return np.ones(len(X))
 floor,t=spec;left=np.expm1(np.minimum(X[:,8].astype(float),np.log1p(288)));right=np.expm1(np.minimum(X[:,9].astype(float),np.log1p(288)));gap=np.fmin(left,right);gap=np.where(np.isfinite(gap),np.rint(gap),288);return floor+(1-floor)*np.clip((gap-1)/(t-1),0,1)

def load(panel,pol):
 parent=[lgb.Booster(model_file=str(ens.PARENT/'models'/panel/n)) for n in ['speed.txt','flow_per_lane.txt']];teams=[[lgb.Booster(model_file=str(fit.O/'models'/panel/n)) for n in ['speed.txt','flow_per_lane.txt']]]
 if pol['gamma']>0:
  for seed in ens.SEEDS[1:]:teams.append([lgb.Booster(model_file=str(ens.O/'models'/panel/str(seed)/n)) for n in ['speed.txt','flow_per_lane.txt']])
 return parent,teams

def outputs(f,X,parent,teams,pol,spec):
 vv=ens.predictions(f,X,parent,teams,pol['parent_alpha'],pol['gamma']);base=vv['parent'];full=vv['ensemble'];return base,full,base+gate(X,spec)[:,None]*(full-base)

def aux(root,panel,conditions):
 result={}
 for condition in conditions:
  cases=[]
  for path in sorted((root/panel/condition).glob('*/public_train_supervision.parquet')):
   folder=path.parent;z=np.load(folder/'features61.npz');assert str(z['keys_sha256'])==d.sha(folder/'auxiliary_keys.csv');f=pd.read_parquet(path);X=z['X'];assert len(f)==len(X);cases.append((dict(date=folder.name,frame=f),X))
  assert cases,(panel,condition);result[condition]=cases
 return result

def main():
 X=np.zeros((3,61));X[:,8]=np.log1p([1,6,np.nan]);X[:,9]=np.log1p([1,9,np.nan]);assert np.allclose(gate(X,(0.,6)),[0,1,1]);assert np.allclose(gate(X,(.5,6)),[.5,1,1]);d.save(O/'gate_runtime_checks.json',dict(isolated_and_long_and_unknown_distances=True,feature_indices=dict(log_previous_speed_gap=8,log_next_speed_gap=9)))
 pols=json.loads((fit.O/'frozen_policy.json').read_text())['policy'];protocol=dict(registered_utc=datetime.now(timezone.utc).isoformat(),hypothesis='A fixed road-wide severe correction applies to mild isolated gaps too; attenuate it using public nearest-observation gap to reduce accumulation regressions.',gates=GATES,features='28 temporal features indices8/9 are log1p previous/next observed speed distance. Integer nearest distance; unavailable both sides=288. No labels used.',selection='One global gate minimizes equally weighted mean January accumulation RMSE across original/point10/block10/station15/block30, subject total mean S_state>=unconditional-0.00002, original mean state>=unconditional-0.0001, every road original state>=unconditional-0.0002, every condition mean accumulation<=unconditional, and every road/condition accumulation<=unconditional*1.02. Unconditional fallback.',January_dates='Original Jan21-31; every cached development case from missingness_selection/january and severe_augmentation_data/development',inherited_policy_sha256=d.sha(fit.O/'frozen_policy.json'),ensemble_policy_sha256=d.sha(ens.O/'frozen_policy.json'),source_sha256=d.sha(__file__),scope='Adaptively reused exposed public development, no blind/official claim. No February model/road selection. Task4 baseline fixed; no uploads.')
 d.save(O/'protocol.json',protocol);rows=[]
 for panel,pol in pols.items():
  d.check_deadline();d.state(status='running',stage='temporal_gap_gate_january',panel=panel,next='Select one global public gap gate on January only; freeze before February and keep negative gates')
  parent,teams=load(panel,pol);cases={'original':[a.original(panel,f'2031-01-{i:02}') for i in range(21,32)]};cases.update(aux(d.WORK/'missingness_selection/january',panel,['point10','block10']));cases.update(aux(d.WORK/'severe_augmentation_data/development',panel,['station15','block30']))
  for condition,itemsX in cases.items():
   items=[it for it,X in itemsX];values={n:[] for n in GATES}
   for it,X in itemsX:
    vv=ens.predictions(it['frame'],X,parent,teams,pol['parent_alpha'],pol['gamma']);before=vv['parent'];full=vv['ensemble']
    for name,spec in GATES.items():values[name].append(before+gate(X,spec)[:,None]*(full-before))
   rows.extend(dict(panel=panel,condition=condition,variant=n,**d.night.metrics(items,v)) for n,v in values.items());pd.DataFrame(rows).to_csv(O/'january_by_panel.csv',index=False)
  print('GAP GATE JAN',panel,flush=True)
 df=pd.DataFrame(rows);ref=df[df.variant=='unconditional'].set_index(['panel','condition']);best=ref.station_target_deltaN_RMSE.mean();chosen='unconditional';decisions=[]
 for name in GATES:
  rr=df[df.variant==name].set_index(['panel','condition']).loc[ref.index];orig=rr.xs('original',level='condition');reforig=ref.xs('original',level='condition');eligible=bool(rr.S_state.mean()>=ref.S_state.mean()-.00002 and orig.S_state.mean()>=reforig.S_state.mean()-.0001 and (orig.S_state>=reforig.S_state-.0002).all() and (rr.groupby(level='condition').station_target_deltaN_RMSE.mean()<=ref.groupby(level='condition').station_target_deltaN_RMSE.mean()+1e-12).all() and (rr.station_target_deltaN_RMSE<=ref.station_target_deltaN_RMSE*1.02+1e-12).all());value=float(rr.station_target_deltaN_RMSE.mean());decisions.append(dict(variant=name,eligible=eligible,objective=value,S_state=float(rr.S_state.mean())))
  if eligible and value<best:chosen=name;best=value
 pd.DataFrame(decisions).to_csv(O/'january_selection.csv',index=False);d.save(O/'frozen_policy.json',dict(gate=chosen,spec=GATES[chosen],frozen_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=d.sha(O/'protocol.json')))
 if chosen=='unconditional':
  d.save(O/'evaluation_skipped.json',dict(reason='No alternative gap gate passed January constraints; unchanged ensemble evaluation skipped',new_score=None));d.state(status='stage_complete',stage='temporal_gap_gate_complete',next='Negative January gate result retained; no duplicate February screening, keep frozen mixture and all failures.');return
 rows=[]
 for panel,pol in pols.items():
  d.check_deadline();d.state(status='running',stage='temporal_gap_gate_february',panel=panel,next='Frozen global gate transfer, no selection by February')
  parent,teams=load(panel,pol);cases={'original':[a.original(panel,f'2031-02-{i:02}') for i in range(1,29)]}
  for dataset,root in ens.DATASETS.items():
   for condition,items in aux(root,panel,['station15','block30'] if dataset=='severe82' else ['point10','block10']).items():cases[dataset+'/'+condition]=items
  for condition,itemsX in cases.items():
   values={n:[] for n in ['parent','unconditional','gated']}
   for it,X in itemsX:
    before,full,new=outputs(it['frame'],X,parent,teams,pol,GATES[chosen]);values['parent'].append(before);values['unconditional'].append(full);values['gated'].append(new)
   rows.extend(dict(panel=panel,condition=condition,variant=n,**d.night.metrics([it for it,X in itemsX],v)) for n,v in values.items());pd.DataFrame(rows).to_csv(O/'february_by_panel.csv',index=False)
 metrics=['S_state','speed_RMSE','flow_per_lane_RMSE','station_target_deltaN_RMSE'];pd.DataFrame(rows).groupby(['condition','variant'])[metrics].mean().to_csv(O/'february_summary.csv');d.state(status='stage_complete',stage='temporal_gap_gate_complete',next='Assess frozen gate transfer and all failures; do not pick roads or gates using February; no official improvement claim or upload.')

if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True);lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 except Exception as e:d.state(status='failed_or_cutoff',stage='temporal_gap_gate',error=repr(e),next='Repair gate execution retaining global January selection rules and all dates');traceback.print_exc();raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
