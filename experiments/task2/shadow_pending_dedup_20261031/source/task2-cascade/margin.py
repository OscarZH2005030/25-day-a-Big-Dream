"""Continuous speed supervision with a fixed physical queue boundary."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
import lightgbm as lgb
from core import r
from episodes import load,HERE
import run as cascade

def fit(months,name,backend):
    xs=[];ys=[];ws=[]
    for pop in months:
        meta,z=load(pop,'early');counts=Counter(e['panel'] for e in meta)
        for i,e in enumerate(meta):
            if e['condition']!=0:continue
            ri=z[f'{i}_ri'];hi=z[f'{i}_hi'];known=z[f'{i}_m'][hi,ri]
            if not known.any():continue
            slot=int(e['origin'][11:13])*12+int(e['origin'][14:16])//5;net=backend[3][e['panel']];raw=backend[1].load_day(e['panel'],e['origin'][:10]);speed=raw['speed'][slot+1:slot+7]
            assert np.array_equal((speed<=net['threshold'])&z[f'{i}_m'],z[f'{i}_y'].astype(bool)&z[f'{i}_m'])
            x=z[f'{i}_Z'][known];target=(speed[hi[known],ri[known]]-net['threshold'][ri[known]])/net['free_speed'][ri[known]]
            # Regress a change from the last observed speed margin, rather
            # than relearning each link's absolute free-flow speed.
            xs.append(x);ys.append(target+x[:,3]);ws.append(np.full(known.sum(),1/counts[e['panel']]/known.sum()))
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);w/=w.mean()
    params=dict(objective='regression',num_leaves=7,learning_rate=.04,min_data_in_leaf=30,lambda_l2=10,feature_fraction=.9,num_threads=4,verbosity=-1,seed=20261029,deterministic=True,force_col_wise=True,max_bin=31)
    model=lgb.train(params,lgb.Dataset(x,label=y,weight=w),num_boost_round=160);model.save_model(str(HERE/(name+'.txt')))
    r.write(HERE/(name+'.json'),dict(months=months,rows=len(y),features=x.shape[1],threshold=.5,class_weight_boost=False,target='normalized future speed minus queue cutoff, residualized by last observed margin',decision='sigmoid(-predicted_margin)>=0.5; fixed published physical queue cutoff; score is not calibrated probability'))
    return model

class Margin:
    def __init__(self,model):self.model=model
    def predict(self,x,**kw):
        margin=self.model.predict(x,**kw)-x[:,3]
        return 1/(1+np.exp(np.clip(margin,-50,50)))

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();backend=r.setup(args);probe=Margin(fit(['october'],'margin-probe',backend));meta,z=load('november','early');result=cascade.metrics(meta,z,backend,'verified',probe)[0]
    prior=r.read(args.output/'task2_joint_pattern_internal.json')['internal']['previous'];selected=result['onset_macro']>prior['onset_macro'] and result['FP']<=prior['FP'] and result['negative_false_any']<=prior['negative_false_any']
    r.write(args.output/'task2_margin_internal.json',dict(scores=result,selected=selected,threshold=.5,selection='November only; improve IoU without increasing FP cells or negative false-any vs previous verifier'))
    r.write(HERE/'margin-frozen-policy.json',dict(selected=selected,threshold=.5,later_month_selection=False));model=Margin(fit(['october','november'],'margin-full',backend));reports={}
    for pop in ['december','january','february_new','february_old']:
        meta,z=load(pop,'full');scores,rows=cascade.metrics(meta,z,backend,'verified',model);reports[pop]=dict(scores=scores,windows=rows)
        r.write(args.output/'task2_margin_external.json',dict(reports=reports,selected_on_november=selected,threshold=.5,all_evaluations_reused_development=True,official_truth_available=False));print(json.dumps(dict(population=pop,selected=selected,scores={k:scores[k] for k in ['onset_macro','TP','FP','FN','negative_false_any']})),flush=True)

if __name__=='__main__':main()
