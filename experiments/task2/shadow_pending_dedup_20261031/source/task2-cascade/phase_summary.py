"""Paired date-block uncertainty for ongoing and the whole Task2 proxy."""
import numpy as np
from core import ROOT,r

def date_ci(before,after,condition=None,reps=4000):
    if condition is not None:before=[e for e in before if e['condition']==condition];after=[e for e in after if e['condition']==condition]
    old={(e['panel'],e['condition'],e['window_id']):e for e in before};new={(e['panel'],e['condition'],e['window_id']):e for e in after};assert old.keys()==new.keys()
    dates=sorted({e['date'] for e in before});groups=sorted({(e['panel'],e['condition']) for e in before});di={d:i for i,d in enumerate(dates)};gi={g:i for i,g in enumerate(groups)};sums=np.zeros((len(groups),len(dates)));counts=sums.copy()
    for k,e in old.items():
        i=gi[(e['panel'],e['condition'])];j=di[e['date']];sums[i,j]+=new[k]['iou']-e['iou'];counts[i,j]+=1
    rng=np.random.default_rng(20261029);w=rng.multinomial(len(dates),np.full(len(dates),1/len(dates)),size=reps);num=w@sums.T;den=w@counts.T;ok=(den>0).all(1);v=(num[ok]/den[ok]).mean(1)
    return dict(delta=float((sums.sum(1)/counts.sum(1)).mean()),descriptive_95_interval=np.quantile(v,[.025,.975]).tolist(),dates=len(dates),valid_replicates=int(ok.sum()),method='paired global-date blocks; preserve same-date road/condition correlation; equal panel-condition means',not_independent_test=True)

def main():
    out=ROOT/'outputs';base=r.read(out/'task2_learned_cascade_external.json')['reports'];onset=r.read(out/'task2_margin_external.json')['reports'];ongoing=r.read(out/'task2_timed_boundary_external.json')['reports'];reports={}
    for pop,v in ongoing.items():
        old=base[pop]['windows']['base'];on=[e for e in onset[pop]['windows'] if e['condition']==0];reports[pop]={}
        for variant,record in v.items():
            rows=on+record['windows'];on_iou=onset[pop]['scores']['onset_macro'];ong=record['scores']['ongoing_macro'];overall=(on_iou+ong)/2
            reports[pop][variant]=dict(onset_macro=on_iou,ongoing_macro=ong,overall_proxy_IoU=overall,vs_R127_ongoing_uncertainty=date_ci(old,record['windows'],1),vs_R127_overall_uncertainty=date_ci(old,rows),vs_retained_onset_plus_R127_ongoing_uncertainty=date_ci(onset[pop]['windows'],rows))
    r.write(out/'task2_queue_phase_combined_uncertainty.json',dict(reports=reports,selection='all new policies failed internal replacement criteria; retain R127 ongoing',all_evaluations_reused_development=True,official_truth_available=False))
    for pop,v in reports.items():print(pop,v['timed_boundary'])

if __name__=='__main__':main()
