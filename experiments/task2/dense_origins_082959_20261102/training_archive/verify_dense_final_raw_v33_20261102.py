"""Independent fresh-public-raw acceptance for both fixed dense ongoing candidates.

Run only after v32 has completed and frozen all models/thresholds. No selection,
training, threshold changes, or production writes. Scope: 394 ongoing windows;
onset is retained by hash, not claimed newly recomputed here.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
import copy,json,sys,time,hashlib,traceback
from pathlib import Path
import numpy as np
import train_ongoing_horizon as t
from dense_ongoing_final_runtime_v32_20261102 import Portable
B=t.BASE;F=B/'outputs/dense_ongoing_final_v32_20261102';O=B/'outputs/dense_final_raw_v33_20261102';START=time.time()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):t.write(O/n,x)
def log(**x):print(json.dumps(dict(seconds=round(time.time()-START,1),**x)),flush=True)

def main():
    assert read(F/'verification.json')['passed']
    frozen=read(F/'frozen_before_February.json')
    for p,h in frozen.items():assert sha(B/p)==h,p
    selection=read(F/'January_selection.json')['candidates'];expected=read(F/'February_results.json')
    assert list(selection)==['dense_100','dense_50']
    O.mkdir(exist_ok=False)
    save('decision_before_verification.json',dict(candidates_in_fixed_October_order=['dense_100','dense_50'],primary='dense_100',scope='Fresh public raw observations, all184+210ongoingwindows; original onset unchanged by hash, not a new777window onset re-evaluation',no_training_selection_or_threshold_change=True,production_unchanged=True,Kaggle_upload=False,source_sha256=sha(Path(__file__)),runtime_sha256=sha(B/'outputs/dense_ongoing_final_runtime_v32_20261102.py')))
    # Parent guard restricts native loading; preload every candidate head first.
    runtime=Portable(F);runtime.preload(t.PANELS)
    task=B/'work/repo/experiments/task2';sys.path.insert(0,str(task/'best_overall_082639_jaccard_20261031/code'))
    from hybrid_ongoing import Hybrid,features,spatial_features
    shared=Hybrid(task/'best_overall_082639_jaccard_20261031')
    sys.path.insert(0,str(task/'r127_reproduction_20261027'))
    from repro_common import setup
    parent,sample,_,audit=setup(B/'work/r127/ConditionalPosition-R127',B/'work/local_transfer/data/kaggle_public',O/'raw-cache',True,extra_allowed=(B/'outputs',task))
    shared.attach_parent(parent)
    import ramp_features as rf
    def construct(s):
        panel=s['panel'];basis=shared.onset.bases[panel]['position']
        x,n,_=features(s,rf.ramp_history(s),rf.topology(panel)['counts'],basis,shared.context[panel]);ex,en=spatial_features(s,basis)
        assert n==shared.names['base'] and en==shared.names['extra']
        return np.column_stack([x,ex]).astype('float32'),np.asarray(parent.probability(s).T,dtype='float32')
    results={};checks=[]
    for pop in ['new_phase_population','old410']:
        cached,meta=t.speed.query(pop);raw={k:np.empty_like(cached[k]) for k in ['X','parent','y','m']};labels=np.empty(len(cached['X']),dtype='U16')
        assert len(meta['windows'])=={'new_phase_population':184,'old410':210}[pop]
        entries=read(task/'r127_reproduction_20261027/indices'/f'february-{pop}.json')['samples']
        lookup={(e['panel'],t.old.prior.old.normtime(e['origin'])):e for e in entries if e['condition']==1 and e['has_future_queue']};seen=set()
        for i,w in enumerate(meta['windows']):
            e=lookup[w['panel'],t.old.prior.old.normtime(w['origin'])];s=sample(e);x,p=construct(s);a,b=w['start'],w['end'];assert len(x)==b-a
            raw['X'][a:b]=x;raw['parent'][a:b]=p;raw['y'][a:b]=np.asarray(s['y']).T;raw['m'][a:b]=np.asarray(s['m']).T;labels[a:b]=w['panel']
            if w['panel'] not in seen:
                other=copy.deepcopy(s);other['y']=1-s['y'];other['m']=~s['m'];other['speed_y']=np.full_like(s['speed_y'],999);other['has_future_queue']=not s['has_future_queue'];xx,pp=construct(other)
                np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(p,pp)
                for alpha in [1.,.5]:np.testing.assert_array_equal(runtime.predict(s['panel'],x,p,alpha),runtime.predict(s['panel'],xx,pp,alpha))
                checks.append(dict(population=pop,panel=w['panel'],future_y_m_speed_y_mutation_input_and_prediction_invariant=True));seen.add(w['panel'])
            if (i+1)%32==0:log(stage='RAW_REBUILD',population=pop,windows=i+1)
        for k in raw:np.testing.assert_array_equal(raw[k],cached[k],err_msg=pop+'/'+k)
        save(f'{pop}_raw_parity.json',dict(passed=True,windows=len(meta['windows']),node_rows=len(labels),all_X_parent_y_m_exact=True))
        reports={};r127=t.old.score(raw['parent'],raw,meta,.25)
        for name,row in selection.items():
            prob=np.empty_like(raw['parent']);alpha=row['config']['alpha']
            for panel in t.PANELS:
                ids=np.flatnonzero(labels==panel);prob[ids]=runtime.predict(panel,raw['X'][ids],raw['parent'][ids],alpha)
            gold=np.load(F/f'{pop}_{name}_probability.npy',mmap_mode='r');error=float(np.max(abs(prob-gold)));assert error<=1e-7,(pop,name,error)
            score=t.old.score(prob,raw,meta,np.asarray(row['selected']['thresholds']));target=expected[pop]['candidates'][name]
            assert score['cell_counts']==target['score']['cell_counts']
            assert abs(score['onset_IoU']-target['score']['onset_IoU'])<1e-12
            overall=(score['onset_IoU']+target['fixed_onset_IoU'])/2;assert abs(overall-target['overall_IoU'])<1e-12
            np.save(O/f'{pop}_{name}_probability.npy',prob)
            reports[name]=dict(max_probability_error=error,thresholds=row['selected']['thresholds'],score=score,fixed_onset_IoU=target['fixed_onset_IoU'],overall_IoU=overall,vs_default=t.old.prior.REF.bootstrap(score,expected[pop]['default']),vs_R127=t.old.prior.REF.bootstrap(score,r127))
            log(stage='CANDIDATE_VERIFIED',population=pop,name=name,overall=overall,FP=score['cell_counts']['FP_cells'],FN=score['cell_counts']['FN_cells'])
        results[pop]=dict(raw_windows=len(meta['windows']),R127=r127,candidates=reports);save('results.json',results)
    for p,h in frozen.items():assert sha(B/p)==h,p
    t.check()
    save('verification.json',dict(passed=True,raw_ongoing_windows=394,all_raw_X_parent_y_m_exact=True,all_candidate_probabilities_IoU_FP_FN_reproduced=True,causality_checks=checks,raw_day_arrays_hash_checked=len(audit['generated_day_keys']),onset_hashes_unchanged=True,onset_not_newly_recomputed=True,no_default_updated=True,no_February_selection=True,official_score=None,Kaggle_upload=False))
    log(stage='COMPLETE')

if __name__=='__main__':
    try:main()
    except Exception as e:
        if O.exists():save('failure.json',dict(error=repr(e),traceback=traceback.format_exc(),old_source_permanently_readonly=True))
        raise
