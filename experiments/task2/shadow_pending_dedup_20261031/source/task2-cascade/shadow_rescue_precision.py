"""Independent shadow precision of OOF-accepted frozen margin rescues.

Default policies and ancestors remain unchanged. Original unfiltered rescues
are used when computing unit features/decisions; only shadow rescue outputs
can be vetoed. No future truth is used for inference or per-cell protection.
"""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import lightgbm as lgb
from core import ROOT,r
from episodes import load
from margin import Margin
from arrival_safe import HERE as OOF_DIR
import arrival_wait as wait
import arrival_wait_combined as retained
import propagation_research as research
from arrival_unit_directional import models as primary_models
from origin_stress_inference import check_models

HERE=research.HERE/'shadow_rescue';HERE.mkdir(exist_ok=True)

def risk_features(legacy,predicted_margin,raw,h,view):
    legacy=np.asarray(legacy);assert legacy.shape==(218,) and 0<=h<=5
    f=np.r_[legacy,predicted_margin,predicted_margin+legacy[3]]
    if view=='observed':f=np.r_[f,research.horizon_features(raw,h)]
    assert np.isfinite(f).all();return f.astype('float32')

def shadow_prediction(parent,raw_rescue,fixed_unit,veto):
    # Labels are intentionally absent from this interface.
    assert np.shape(parent)==np.shape(raw_rescue)==np.shape(fixed_unit)==np.shape(veto)
    return np.asarray(parent,bool)|np.asarray(fixed_unit,bool)|(np.asarray(raw_rescue,bool)&~np.asarray(veto,bool))

def oof_models(name):
    path=OOF_DIR/f'{name}-oof-rescues.npz';assert path.exists(),path
    acceptance=dict(np.load(path));provenance=r.read(OOF_DIR/f'{name}-rescue-provenance.json');bydate={d:f['fold'] for f in provenance['folds'] for d in f['heldout_dates']};models={k:lgb.Booster(model_file=str(OOF_DIR/f'{name}-rescue-fold{k}.txt')) for k in range(3)}
    return acceptance,bydate,models

def training(months,backend,name):
    accepted,fold,models=oof_models(name);rows=[]
    for pop in months:
        meta,z=load(pop,'early');counts=Counter(e['panel'] for e in meta);requests=[];windows={}
        for i,e in enumerate(meta):
            if e['condition']!=0:continue
            ri,hi,x=z[f'{i}_ri'],z[f'{i}_hi'],z[f'{i}_Z'];score=models[fold[e['origin'][:10]]].predict(x,num_threads=4)-x[:,3] if len(x) else np.zeros(0);a=accepted[f'{pop}_{i}'];assert np.array_equal(a,score<=0)
            known=z[f'{i}_m'][hi,ri];selected=np.flatnonzero(a&known);windows[i]=(selected,score)
            requests.extend(dict(i=i,link=int(ri[j])) for j in selected)
        raw=research.raw_cache(backend,pop,'early',meta,requests)
        for i,(selected,score) in windows.items():
            if not len(selected):continue
            e=meta[i];slot=int(e['origin'][11:13])*12+int(e['origin'][14:16])//5;values=backend[1].load_day(e['panel'],e['origin'][:10]);net=backend[3][e['panel']]
            for j in selected:
                link,h=int(z[f'{i}_ri'][j]),int(z[f'{i}_hi'][j]);x=z[f'{i}_Z'][j];actual=(values['speed'][slot+h+1,link]-net['threshold'][link])/net['free_speed'][link];truth=bool(z[f'{i}_y'][h,link]);assert bool(actual<=0)==truth
                rows.append(dict(views={v:risk_features(x,score[j],raw[f'{i}_{link}'],h,v) for v in ['legacy','observed']},not_queued=int(not truth),correction=float(actual-score[j]),date=e['origin'][:10],w=1/counts[e['panel']]/len(selected),panel=e['panel']))
    return rows

def fit(months,backend,name):
    records=training(months,backend,name);dates=np.array([c['date'] for c in records]);mapping={d:i%3 for i,d in enumerate(sorted(set(dates)))};fold=np.array([mapping[d] for d in dates]);w=np.array([c['w'] for c in records]);w/=w.mean();out={};stats=[]
    for view in ['legacy','observed']:
        x=np.array([c['views'][view] for c in records])
        for objective in ['binary','correction']:
            y=np.array([c['not_queued'] if objective=='binary' else c['correction'] for c in records]);models=[]
            for k in [-1,0,1,2]:
                tr=np.ones(len(y),bool) if k==-1 else fold!=k;params=dict(objective='binary' if objective=='binary' else 'regression',num_leaves=7,learning_rate=.04,min_data_in_leaf=30,lambda_l2=20,feature_fraction=.9,num_threads=4,verbosity=-1,seed=20261030,deterministic=True,force_col_wise=True,max_bin=31);model=lgb.train(params,lgb.Dataset(x[tr],label=y[tr],weight=w[tr]),num_boost_round=160);model.save_model(str(HERE/f'{name}-{view}-{objective}-{k}.txt'));models.append(model);stats.append(dict(view=view,objective=objective,fold=k,rows=int(tr.sum()),features=x.shape[1],train_dates=sorted(set(dates[tr]))))
            out[view+'_'+objective]=models
    r.write(HERE/f'{name}-training.json',dict(months=months,OOF_rescue_known_cells=len(records),TP=sum(1-c['not_queued'] for c in records),FP=sum(c['not_queued'] for c in records),heads=stats,thresholds=dict(binary_clear=.5,corrected_queue_margin=0),class_weights=False,unknown_targets_excluded=True,ancestry='frozen OOF margin models exclude target global date; shared earlier feature ancestors, not nested blind CV',shadow_only=True,rounds=160));return out

def fixed_unit_masks(backend,pop,regime,name):
    meta,z,rescue,cases=wait.candidates(backend,pop,regime,name);unit={i:np.zeros_like(z[f'{i}_base'],bool) for i,e in enumerate(meta) if e['condition']==0}
    if cases:
        accept=primary_models(name)['committee_advance'][0].predict(np.array([c['x'] for c in cases]),num_threads=4)>=.5;veto=retained.load(name)['pair'].predict(np.array([c['f'] for c in cases]),num_threads=4)>=.5
        for c,a,v in zip(cases,accept,veto):
            if a and not v:unit[c['i']][c['h'],c['link']]=True
    return meta,z,rescue,unit

def evaluate(backend,pop,regime,name,models):
    meta,z,rescue,unit=fixed_unit_masks(backend,pop,regime,name);_,old=load(pop,regime);fixed=lgb.Booster(model_file=str(ROOT/'work/task2-cascade'/f'margin-{name}.txt'));requests=[];windows={}
    for i,e in enumerate(meta):
        if e['condition']!=0:continue
        x,ri,hi=old[f'{i}_Z'],old[f'{i}_ri'],old[f'{i}_hi'];margin=fixed.predict(x,num_threads=4)-x[:,3] if len(x) else np.zeros(0);selected=np.flatnonzero(margin<=0);assert int(rescue[i].sum())==len(selected);windows[i]=(selected,margin);requests.extend(dict(i=i,link=int(ri[j])) for j in selected)
    raw=research.raw_cache(backend,pop,regime,meta,requests);policies=['retained']+[key+'_'+kind for key in models for kind in ['single','consensus']];veto={p:{i:np.zeros_like(rescue[i]) for i in rescue} for p in policies};catalog=[]
    for i,(selected,margin) in windows.items():
        if not len(selected):continue
        e=meta[i];features={view:np.array([risk_features(old[f'{i}_Z'][j],margin[j],raw[f"{i}_{int(old[f'{i}_ri'][j])}"],int(old[f'{i}_hi'][j]),view) for j in selected]) for view in ['legacy','observed']};scores={};decision={}
        for key,bags in models.items():
            view,objective=key.split('_');p=np.array([m.predict(features[view],num_threads=4) for m in bags]);reject=(p>=.5) if objective=='binary' else ((p+margin[selected][None])>0);scores[key]=p;decision[key+'_single']=reject[0];decision[key+'_consensus']=reject[1:].all(0)
            for policy,mask in [(key+'_single',reject[0]),(key+'_consensus',reject[1:].all(0))]:veto[policy][i][old[f'{i}_hi'][selected[mask]],old[f'{i}_ri'][selected[mask]]]=True
        for row,j in enumerate(selected):
            link,h=int(old[f'{i}_ri'][j]),int(old[f'{i}_hi'][j]);m=z[f'{i}_m'][h,link];y=z[f'{i}_y'][h,link];catalog.append(dict(panel=e['panel'],origin=e['origin'],link_index=link,horizon_minutes=5*(h+1),truth=int(y) if m else None,predicted_margin=float(margin[j]),scores={k:v[:,row].tolist() for k,v in scores.items()},veto={p:bool(veto[p][i][h,link]) for p in policies}))
    reports={}
    for policy in policies:
        rows=[];neg={};TP=FP=FN=lost_rescue=lost_retained=removed_fp=unknown_removed=0;allTP=allFP=allFN=0;anchors={}
        for i,e in enumerate(meta):
            parent=z[f'{i}_base']>=(.5 if e['condition']==0 else .25);p=parent.copy();m=z[f'{i}_m'];y=z[f'{i}_y'].astype(bool)
            if e['condition']==0:
                baseline=parent|rescue[i]|unit[i];p=shadow_prediction(parent,rescue[i],unit[i],veto[policy][i]);assert not (parent&~p).any() and not (unit[i]&~p).any();lost_rescue+=int((rescue[i]&y&m&~p).sum());lost_retained+=int((baseline&y&m&~p).sum());removed_fp+=int((rescue[i]&~y&m&~p).sum());unknown_removed+=int((rescue[i]&~m&~p).sum());allTP+=int((p&y&m).sum());allFP+=int((p&~y&m).sum());allFN+=int((~p&y&m).sum());anchors[(e['panel'],e['origin'])]=p
                if e['has_future_queue']:TP+=int((p&y&m).sum());FP+=int((p&~y&m).sum());FN+=int((~p&y&m).sum())
                else:neg.setdefault(e['panel'],[]).append(bool((p&m).any()))
            if e['has_future_queue']:rows.append(dict(panel=e['panel'],condition=e['condition'],window_id=e['origin'],date=e['origin'][:10],iou=backend[4](y,p,m)))
        anchor_counts=None
        if pop in ['february_new','february_old']:
            audit=r.read(ROOT/'outputs/task2_blocked_advance_audit.json')['reports'][pop]['catalog'];targets=[c for c in audit if c['group']=='blocked_TP'];kept=sum(bool(anchors[c['panel'],c['origin']][c['horizon_minutes']//5-1,c['link_index']]) for c in targets);assert kept==len(targets);anchor_counts=dict(total=len(targets),kept=kept)
        s=backend[5](rows);s.update(onset_macro=float(np.mean([v for k,v in s['by_panel_condition'].items() if k.endswith('queue_onset')])),TP=TP,FP=FP,FN=FN,all_onset_TP=allTP,all_onset_FP=allFP,all_onset_FN=allFN,lost_rescue_TP=lost_rescue,lost_retained_TP=lost_retained,removed_rescue_FP=removed_fp,unknown_removed=unknown_removed,negative_false_any=float(np.mean([np.mean(v) for v in neg.values()])) if neg else None);reports[policy]=dict(scores=s,windows=rows,unit_TP_anchors=anchor_counts)
    # Reference itself must match the original retained study, not a modified
    # rescue vector recomputed after the shadow gate.
    expected=r.read(ROOT/'outputs/task2_arrival_wait_combined.json')['reports'][pop]['combined']['scores']
    for k in ['TP','FP','FN','onset_macro']:assert np.isclose(reports['retained']['scores'][k],expected[k]),(pop,k,reports['retained']['scores'][k],expected[k])
    r.write(HERE/f'cases-{regime}-{pop}.json',catalog);return reports

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();frozen=r.read(args.output/'task2_origin_stress_frozen.json');check_models(frozen);backend=r.setup(args);probe=fit(['october'],backend,'probe');internal=evaluate(backend,'november','early','probe',probe);b=internal['retained']['scores'];eligible=[k for k,v in internal.items() if k!='retained' and v['scores']['lost_retained_TP']==0 and v['scores']['all_onset_FP']<b['all_onset_FP'] and v['scores']['onset_macro']>=b['onset_macro'] and v['scores']['negative_false_any']<=b['negative_false_any']];selected=max(eligible,key=lambda k:internal[k]['scores']['onset_macro']) if eligible else 'retained';r.write(args.output/'task2_shadow_rescue_internal.json',dict(reports=internal,selected=selected,shadow_only=True,all_months_development=True));print(json.dumps(dict(pop='november',selected=selected,scores={k:{j:v['scores'][j] for j in ['onset_macro','TP','FP','lost_rescue_TP','removed_rescue_FP']} for k,v in internal.items()})),flush=True)
    full=fit(['october','november'],backend,'full');reports={}
    for pop in ['december','january','february_new','february_old']:
        reports[pop]=evaluate(backend,pop,'full','full',full);r.write(args.output/'task2_shadow_rescue_external.json',dict(reports=reports,selected=selected,shadow_only=True,all_months_development=True,official_truth_available=False,formal_R127_unchanged=True,default_predictions_unchanged=True));print(json.dumps(dict(pop=pop,scores={k:{j:v['scores'][j] for j in ['proxy_IoU','onset_macro','TP','FP','lost_rescue_TP','removed_rescue_FP']} for k,v in reports[pop].items()})),flush=True)
    check_models(frozen)

if __name__=='__main__':main()
