"""Low-dimensional reliability model on OOF-accepted advance candidates."""
import argparse,json
from pathlib import Path
import numpy as np
import lightgbm as lgb
from core import ROOT,r
import arrival_unit as unit
import advance_precision as precision
from arrival_unit_directional import models as load_unit

HERE=precision.HERE/'conditional';HERE.mkdir(exist_ok=True)

def prob_matrix(models,x):return np.array([m.predict(x,num_threads=4) for m in models])

def reliability_features(x,prob):
    # The appended unit features have a fixed 31-column schema, starting with
    # horizon one-hot; raw link/context predictors are never refit from labels.
    tail=x[:,-unit.UNIT_EXTRA_WIDTH:];assert tail.shape[1]==31
    horizon=tail[:,:6].argmax(1)/5
    return np.column_stack([prob.min(0),prob.mean(0),prob.std(0),prob.max(0)-prob.min(0),horizon,tail[:,7],tail[:,10],tail[:,11],tail[:,13],tail[:,14],tail[:,15],tail[:,16],tail[:,17],tail[:,18],tail[:,19],tail[:,20],tail[:,21],tail[:,22],tail[:,23]])

def fit(months,backend,name):
    import torch
    torch.set_num_threads(4);x,y,d,w,dates=unit.dataset(months,backend,name);take=d==0;x=x[take];y=y[take];w=w[take];dates=dates[take];w/=w.mean();mapping={day:i%3 for i,day in enumerate(sorted(set(dates)))};fold=np.array([mapping[day] for day in dates]);oof=np.zeros((3,len(x)));provenance=[]
    for k in range(3):
        tr=fold!=k;te=~tr;assert not set(dates[tr])&set(dates[te])
        for bag in range(3):
            params=dict(objective='binary',num_leaves=7,learning_rate=.04,min_data_in_leaf=30,lambda_l2=10,feature_fraction=.9,num_threads=4,verbosity=-1,seed=20261030+bag,deterministic=True,force_col_wise=True,max_bin=31);model=lgb.train(params,lgb.Dataset(x[tr],label=y[tr],weight=w[tr]),num_boost_round=160);oof[bag,te]=model.predict(x[te],num_threads=4);model.save_model(str(HERE/f'{name}-teacher-{k}-{bag}.txt'))
        provenance.append(dict(fold=k,train_dates=sorted(set(dates[tr])),heldout_dates=sorted(set(dates[te]))))
    accepted=(oof>=.5).all(0);features=reliability_features(x,oof);xx=features[accepted];yy=y[accepted];ww=w[accepted];ww/=ww.sum();mean=xx.mean(0);scale=np.maximum(xx.std(0),.01);z=(xx-mean)/scale;zt=torch.tensor(z,dtype=torch.float64);yt=torch.tensor(yy,dtype=torch.float64);wt=torch.tensor(ww,dtype=torch.float64);coef=torch.zeros(z.shape[1],dtype=torch.float64,requires_grad=True);bias=torch.zeros((),dtype=torch.float64,requires_grad=True);optimizer=torch.optim.LBFGS([coef,bias],max_iter=100,line_search_fn='strong_wolfe');state={}
    def closure():
        optimizer.zero_grad();logits=zt@coef+bias;loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,yt,reduction='none')*wt).sum()+.1*(coef**2).mean();loss.backward();state['loss']=float(loss.detach());return loss
    optimizer.step(closure);model=dict(mean=mean.tolist(),scale=scale.tolist(),coef=coef.detach().numpy().tolist(),bias=float(bias.detach()),features=z.shape[1],threshold=.5,accepted_OOF_samples=int(accepted.sum()),positive=int(yy.sum()),negative=int((~yy.astype(bool)).sum()),months=months,date_folds=provenance,regularization='0.1 mean squared coefficients, intercept unpenalized',no_class_weight_boost=True,loss=state['loss']);r.write(HERE/f'{name}-reliability.json',model);return model

def reliability_accept(model,x,prob):
    features=reliability_features(x,prob);z=(features-np.array(model['mean']))/np.array(model['scale']);logits=z@np.array(model['coef'])+model['bias'];return logits>=0

def evaluate(backend,pop,regime,model):
    # Reuse the verified advance-only evaluator and wrap the frozen primary
    # committee. The wrapper returns zero for a rejected additional action.
    primary=load_unit('probe' if regime=='early' else 'full')['committee_advance'][0]
    class Conditional:
        def predict(self,x,**kw):
            prob=prob_matrix(primary.models,x);keep=(prob>=.5).all(0)&reliability_accept(model,x,prob);return keep.astype(float)
    return unit.evaluate(backend,pop,regime,{'conditional_advance':{0:Conditional()},'committee_advance':{0:primary}},output_dir=HERE)

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();backend=r.setup(args);probe=fit(['october'],backend,'probe');internal=evaluate(backend,'november','early',probe);b=internal['margin']['scores'];control=internal['committee_advance']['scores'];s=internal['conditional_advance']['scores'];selected='conditional_advance' if s['onset_macro']>=b['onset_macro'] and s['FP']<control['FP'] and s['TP']>=b['TP'] else 'committee_advance';r.write(args.output/'task2_advance_conditional_internal.json',dict(reports=internal,selected_research_candidate=selected,formal_parent='R127 unchanged',later_month_selection=False,all_evaluations_reused_development=True));r.write(HERE/'frozen-policy.json',dict(selected_research_candidate=selected,thresholds=dict(parent=.5,ongoing=.25,primary=.5,reliability=.5),later_month_selection=False,formal_parent_unchanged=True));model=fit(['october','november'],backend,'full');reports={}
    for pop in ['december','january','february_new','february_old']:
        reports[pop]=evaluate(backend,pop,'full',model);r.write(args.output/'task2_advance_conditional_external.json',dict(reports=reports,frozen_research_selection=selected,formal_parent='R127 unchanged',all_evaluations_reused_development=True,official_truth_available=False));print(json.dumps(dict(population=pop,selected_research_candidate=selected,scores={k:{j:v['scores'][j] for j in ['onset_macro','TP','FP','lost_original_R127_TP']} for k,v in reports[pop].items()})),flush=True)

if __name__=='__main__':main()
