"""January-only point/temporal-error accumulation correction controls."""
import common as d
import augmentation_fit as a
import numpy as np,pandas as pd,lightgbm as lgb,json,msvcrt,traceback
from datetime import datetime,timezone
O=d.OUT/'accumulation_fit';RECIPES=[0.,1.];ALPHAS=[.25,.5,1.]

def features(p,X,context):
 N=context[:,1]/np.maximum(context[:,0],1)*p.length_km.to_numpy()
 return np.column_stack([X,context[:,0]/25,context[:,1]/p.lanes.to_numpy()/600,N/10,p.length_km,p.lanes]),N

def edges(items):
 frames=[it['frame'][['timestamp','station_id','link_id']].assign(date=it['date']) for it in items];p=pd.concat(frames,ignore_index=True);p['row']=np.arange(len(p));p=p.sort_values(['date','station_id','link_id','timestamp']);same=(p.date.to_numpy()[1:]==p.date.to_numpy()[:-1])&(p.station_id.to_numpy()[1:]==p.station_id.to_numpy()[:-1])&(p.link_id.to_numpy()[1:]==p.link_id.to_numpy()[:-1]);t=pd.to_datetime(p.timestamp,utc=True).astype('int64').to_numpy();good=same&(np.diff(t)==300000000000);r=p.row.to_numpy();return r[:-1][good],r[1:][good]

def gh(pred,y,edge,lam):
 e=pred-y;g=e.copy();h=np.ones(len(e));i,j=edge;delta=lam*(e[i]-e[j]);np.add.at(g,i,delta);np.add.at(g,j,-delta);np.add.at(h,i,lam);np.add.at(h,j,lam);return g,h

def loss(pred,y,edge,lam):
 e=pred-y;i,j=edge;return .5*(e@e+lam*np.sum((e[i]-e[j])**2))

def checks():
 y=np.array([.2,.5,-.1]);v=np.array([1.,-.3,.7]);ee=(np.array([0,1]),np.array([1,2]));g,h=gh(v,y,ee,1.);eps=1e-5
 for i in range(3):
  step=np.zeros(3);step[i]=eps;assert abs((loss(v+step,y,ee,1)-loss(v-step,y,ee,1))/(2*eps)-g[i])<1e-7;assert abs((loss(v+step,y,ee,1)-2*loss(v,y,ee,1)+loss(v-step,y,ee,1))/eps**2-h[i])<1e-4
 d.save(O/'runtime_checks.json',dict(custom_gradient_finite_difference=True,diagonal_hessian_finite_difference=True))

def prepared(panel,dates):
 rp,rm=a.load_reference(panel);items=[];Xs=[];Ys=[];contexts=[];Ns=[]
 for date in dates:
  it,X=a.original(panel,date);p=it['frame'];c=a.correct(p,X,rm,rp['alpha']);F,N=features(p,X,c);trueN=p.true_q.to_numpy()/np.maximum(p.true_v.to_numpy(),1)*p.length_km.to_numpy();items.append(it);Xs.append(F);Ys.append((trueN-N)/10);contexts.append(c);Ns.append(N)
 return items,Xs,Ys,contexts,Ns

def corrected(item,context,N,increment,alpha):
 p=item['frame'];out=context.copy();out[:,1]=np.maximum(N+10*alpha*increment,0)*np.maximum(out[:,0],1)/p.length_km.to_numpy();return out

def main():
 checks();protocol=dict(registered_utc=datetime.now(timezone.utc).isoformat(),hypothesis='Jointly target actual accumulation errors instead of independent flow residuals; temporal error penalty may improve adjacent accumulation change.',train_dates=[f'2031-01-{i:02d}' for i in range(5,21)],dev_dates=[f'2031-01-{i:02d}' for i in range(21,32)],eval_dates=[f'2031-02-{i:02d}' for i in range(1,29)],recipes=RECIPES,alphas=ALPHAS,target='(true_q/max(true_v,1)*length - context_q/max(context_v,1)*length)/10',objective='0.5*sum(error^2)+0.5*lambda*sum(adjacent_error_difference^2); exact gradient and diagonal Hessian, five-minute same station/link/date edges only',features='61 public masked features + context speed/25, context per-lane flow/600, context accumulation/10, link length, lanes',model=dict(num_leaves=31,min_data_in_leaf=1000,learning_rate=.03,lambda_l2=20,num_boost_round=1000,early_stopping=60,seed=2026110240),selection='One global recipe and alpha minimizes mean January development accumulation-change RMSE; mean state >=context-0.00002, each road state >=context-0.0002 and each road accumulation RMSE <=context*1.02; context fallback. No February tuning.',source_sha256=d.sha(__file__),context_model_sha256={str(p.relative_to(d.ROOT)):d.sha(p) for p in (d.ROOT/'outputs/overnight/context_candidate/models').rglob('*.txt')},caveat='Exposed public development data, not blind/official Task3; topology-fallback LWR never used. Task4 frozen; no Kaggle upload.')
 d.save(O/'protocol.json',protocol);rows=[]
 for panel in d.night.predictor.load_routes():
  d.check_deadline();d.state(status='running',stage='accumulation_january_fit',panel=panel,next='Fit point/temporal accumulation objectives, choose global policy on January, freeze before February')
  train=prepared(panel,protocol['train_dates']);dev=prepared(panel,protocol['dev_dates']);X=np.concatenate(train[1]);y=np.concatenate(train[2]);V=np.concatenate(dev[1]);z=np.concatenate(dev[2]);tr_edges=edges(train[0]);va_edges=edges(dev[0]);ref=d.night.metrics(dev[0],dev[3]);rows.append(dict(panel=panel,recipe='context',alpha=0.,**ref))
  for lam in RECIPES:
   def objective(pred,ds):return gh(pred,ds.get_label(),tr_edges,lam)
   def metric(pred,ds):return 'accumulation_and_change',2*loss(pred,ds.get_label(),va_edges,lam)/len(pred),False
   params=dict(objective=objective,metric='None',num_leaves=31,min_data_in_leaf=1000,learning_rate=.03,lambda_l2=20,num_threads=4,verbosity=-1,seed=2026110240,deterministic=True,force_col_wise=True)
   m=lgb.train(params,lgb.Dataset(X,label=y),num_boost_round=1000,valid_sets=[lgb.Dataset(V,label=z)],feval=metric,callbacks=[lgb.early_stopping(60,verbose=False)]);path=O/'models'/panel/('lambda'+str(lam)+'.txt');path.parent.mkdir(parents=True,exist_ok=True);m.save_model(str(path));preds=[m.predict(xx,num_threads=4) for xx in dev[1]]
   for alpha in ALPHAS:
    score=d.night.metrics(dev[0],[corrected(it,c,n,pr,alpha) for it,c,n,pr in zip(dev[0],dev[3],dev[4],preds)]);rows.append(dict(panel=panel,recipe=str(lam),alpha=alpha,iterations=m.current_iteration(),**score))
  pd.DataFrame(rows).to_csv(O/'january_by_panel.csv',index=False);print('ACC FIT',panel,flush=True)
 df=pd.DataFrame(rows);ref=df[df.recipe=='context'].set_index('panel');best=ref.station_target_deltaN_RMSE.mean();chosen=dict(recipe='context',alpha=0.);decisions=[]
 for (recipe,alpha),r in df[df.recipe!='context'].groupby(['recipe','alpha']):
  rr=r.set_index('panel').loc[ref.index];eligible=bool(rr.S_state.mean()>=ref.S_state.mean()-.00002 and (rr.S_state>=ref.S_state-.0002).all() and (rr.station_target_deltaN_RMSE<=ref.station_target_deltaN_RMSE*1.02).all());value=float(rr.station_target_deltaN_RMSE.mean());decisions.append(dict(recipe=recipe,alpha=alpha,eligible=eligible,objective=value,S_state=float(rr.S_state.mean())))
  if eligible and value<best:best=value;chosen=dict(recipe=recipe,alpha=alpha)
 pd.DataFrame(decisions).to_csv(O/'january_selection.csv',index=False);d.save(O/'frozen_policy.json',dict(**chosen,frozen_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=d.sha(O/'protocol.json'),model_sha256={str(p.relative_to(d.ROOT)):d.sha(p) for p in (O/'models').rglob('*.txt')}))
 if chosen['recipe']=='context':
  d.save(O/'evaluation_skipped.json',dict(reason='No new model passed January safety gates; skip unchanged February evaluation',new_score=None));d.state(status='stage_complete',stage='accumulation_fit_complete',next='Negative January result preserved; register next hypothesis without choosing by February.');return
 rows=[];daily=[]
 for panel in d.night.predictor.load_routes():
  d.check_deadline();d.state(status='running',stage='accumulation_february_evaluation',panel=panel,next='Fixed global policy development transfer; no retuning')
  items,Xs,_,contexts,Ns=prepared(panel,protocol['eval_dates']);m=lgb.Booster(model_file=str(O/'models'/panel/('lambda'+chosen['recipe']+'.txt')));new=[corrected(it,c,n,m.predict(X,num_threads=4),chosen['alpha']) for it,c,n,X in zip(items,contexts,Ns,Xs)]
  for name,values in [('context',contexts),('selected',new)]:
   rows.append(dict(panel=panel,variant=name,**d.night.metrics(items,values)))
   for it,val in zip(items,values):daily.append(dict(panel=panel,date=it['date'],variant=name,**d.night.metrics([it],[val])))
  pd.DataFrame(rows).to_csv(O/'february_by_panel.csv',index=False);pd.DataFrame(daily).to_csv(O/'february_by_date.csv',index=False)
 pd.DataFrame(rows).groupby('variant')[['S_state','speed_RMSE','flow_per_lane_RMSE','station_target_deltaN_RMSE']].mean().to_csv(O/'february_summary.csv');d.state(status='stage_complete',stage='accumulation_fit_complete',next='Inspect fixed policy and failures; if useful test fresh auxiliary masking with frozen policy, never claim official improvement.')

if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True);lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 except Exception as e:d.state(status='failed_or_cutoff',stage='accumulation_fit',error=repr(e),next='Repair execution without modifying registered January selection rules');traceback.print_exc();raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
