"""Causal traffic context for OOF-approved advance actions; fixed .5 gates."""
import argparse,json
from pathlib import Path
import numpy as np
import lightgbm as lgb
from core import r
import arrival_unit as unit
import arrival_context as context
import advance_conditional as conditional
from arrival_unit_directional import models as load_unit

HERE=conditional.HERE/'traffic_context';HERE.mkdir(exist_ok=True)
# History/current observations and public spatial fields only. No future margin,
# road IDs, calendar-month IDs or post-hoc February exception enters this schema.
INDICES=list(range(20))+list(range(68,93))+list(range(351,454))

def features(x,prob):
    assert x.shape[1]==542
    return np.column_stack([conditional.reliability_features(x,prob),x[:,INDICES]])

def aligned_features(x):
    # Context block 160:256 is the pre-existing 96-column same-timestamp,
    # valid-channel speed/flow/occupancy and directed lead-lag evidence.
    assert x.shape[1]==542
    return np.column_stack([x[:,INDICES],x[:,160:256],x[:,-31:]])

def fit(months,backend,name):
    x,y,d,w,dates=unit.dataset(months,backend,name);take=d==0
    x,y,w,dates=x[take],y[take],w[take],dates[take]
    mapping={day:i%3 for i,day in enumerate(sorted(set(dates)))}
    fold=np.array([mapping[day] for day in dates]);prob=np.zeros((3,len(x)))
    for k in range(3):
        te=fold==k
        for bag in range(3):
            teacher=lgb.Booster(model_file=str(conditional.HERE/f'{name}-teacher-{k}-{bag}.txt'))
            prob[bag,te]=teacher.predict(x[te],num_threads=4)
    accepted=(prob>=.5).all(0);xx=features(x,prob)[accepted];yy=y[accepted];ww=w[accepted];ww/=ww.mean()
    params=dict(objective='binary',num_leaves=3,learning_rate=.04,min_data_in_leaf=30,lambda_l2=20,num_threads=4,verbosity=-1,seed=20261030,deterministic=True,force_col_wise=True,max_bin=31)
    tree=lgb.train(params,lgb.Dataset(xx,label=yy,weight=ww),num_boost_round=80)
    tree.save_model(str(HERE/f'{name}-tree.txt'))
    # A separate method addresses the tiny OOF-approved cohort (118 / 179
    # examples): train a traffic-evidence head on ALL known advance candidates.
    # It still filters only frozen-committee additions at inference.
    aligned_params=dict(params,num_leaves=7,lambda_l2=20)
    aligned=lgb.train(aligned_params,lgb.Dataset(aligned_features(x),label=y,weight=w/w.mean()),num_boost_round=160)
    aligned.save_model(str(HERE/f'{name}-aligned.txt'))
    import torch
    torch.set_num_threads(4)
    mean=xx.mean(0);scale=np.maximum(xx.std(0),.01)
    z=torch.tensor((xx-mean)/scale,dtype=torch.float64);yt=torch.tensor(yy,dtype=torch.float64);wt=torch.tensor(ww/ww.sum(),dtype=torch.float64)
    coef=torch.zeros(xx.shape[1],dtype=torch.float64,requires_grad=True);bias=torch.zeros((),dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([coef,bias],max_iter=100,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();loss=(torch.nn.functional.binary_cross_entropy_with_logits(z@coef+bias,yt,reduction='none')*wt).sum()+.1*(coef**2).mean();loss.backward();return loss
    opt.step(closure)
    logistic=dict(mean=mean.tolist(),scale=scale.tolist(),coef=coef.detach().numpy().tolist(),bias=float(bias.detach()))
    r.write(HERE/f'{name}-logistic.json',logistic)
    r.write(HERE/f'{name}-training.json',dict(months=months,features=xx.shape[1],indices=INDICES,OOF_accepted=len(yy),positive=int(yy.sum()),negative=int((yy==0).sum()),aligned_all_candidates=len(y),aligned_features=aligned_features(x).shape[1],aligned_rounds=160,aligned_params=aligned_params,threshold=.5,tree=params,rounds=80,logistic_regularization='.1 mean squared standardized coefficients',primary_teacher_date_folds=3,independent_blind_test=False,future_diagnostic_used_as_feature=False))
    return dict(tree=tree,logistic=logistic,aligned=aligned)

def evaluate(backend,pop,regime,models):
    primary=load_unit('probe' if regime=='early' else 'full')['committee_advance'][0]
    class Gate:
        def __init__(self,kind,model):self.kind,self.model=kind,model
        def predict(self,x,**kw):
            prob=conditional.prob_matrix(primary.models,x);f=features(x,prob)
            if self.kind=='aligned':accept=self.model.predict(aligned_features(x),num_threads=4)>=.5
            elif self.kind=='tree':accept=self.model.predict(f,num_threads=4)>=.5
            else:
                m=self.model;accept=((f-np.array(m['mean']))/np.array(m['scale']))@np.array(m['coef'])+m['bias']>=0
            return ((prob>=.5).all(0)&accept).astype(float)
    heads={k:{0:Gate(k,v)} for k,v in models.items()};heads['committee_advance']={0:primary}
    return unit.evaluate(backend,pop,regime,heads,output_dir=HERE)

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();backend=r.setup(args)
    probe=fit(['october'],backend,'probe');internal=evaluate(backend,'november','early',probe)
    control=internal['committee_advance']['scores'];eligible=[k for k in probe if internal[k]['scores']['TP']>=control['TP'] and internal[k]['scores']['FP']<control['FP'] and internal[k]['scores']['onset_macro']>=control['onset_macro']]
    selected=max(eligible,key=lambda k:internal[k]['scores']['onset_macro']) if eligible else 'committee_advance'
    r.write(args.output/'task2_advance_context_internal.json',dict(reports=internal,selection=selected,selection_rule='November: retain all added TP and reduce FP and preserve onset IoU; otherwise existing advance model',all_months_reused_development=True))
    print(json.dumps(dict(pop='november',selected=selected,scores={k:{s:v['scores'][s] for s in ['onset_macro','TP','FP']} for k,v in internal.items()})),flush=True)
    full=fit(['october','november'],backend,'full');reports={}
    for pop in ['december','january','february_new','february_old']:
        reports[pop]=evaluate(backend,pop,'full',full)
        r.write(args.output/'task2_advance_context_external.json',dict(reports=reports,selection=selected,official_truth_available=False,all_months_reused_development=True,formal_R127_unchanged=True))
        print(json.dumps(dict(pop=pop,scores={k:{s:v['scores'][s] for s in ['proxy_IoU','onset_macro','TP','FP','FN','lost_original_R127_TP']} for k,v in reports[pop].items()})),flush=True)

if __name__=='__main__':main()
