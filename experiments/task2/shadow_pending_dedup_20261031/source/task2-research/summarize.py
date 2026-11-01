"""Summarize a frozen bounded experiment, without selecting on February."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from research import HERE,read,write

def paired_ci(base,candidate):
    lookup={(r['panel'],r['window_id']):r['iou'] for r in base if r['condition']==0}
    roads=defaultdict(lambda:defaultdict(list))
    for r in candidate:
        if r['condition']==0:
            roads[r['panel']][r['date']].append(r['iou']-lookup[(r['panel'],r['window_id'])])
    rng=np.random.default_rng(20261028);values=[]
    for _ in range(2000):
        draws=[]
        for dates in roads.values():
            groups=list(dates.values());ids=rng.integers(len(groups),size=len(groups));draws.append(np.mean([v for i in ids for v in groups[i]]))
        values.append(np.mean(draws))
    return dict(mean_road_macro_delta=float(np.mean([np.mean([v for g in dates.values() for v in g]) for dates in roads.values()])),date_block_paired_95pct=list(np.quantile(values,[.025,.975])),bootstrap_replicates=2000,scope='descriptive reused-development uncertainty; model selection not accounted for')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'outputs');args=ap.parse_args()
    output=args.output;dec=read(output/'december-experiment.json');base=dec['candidates'][0]
    baseline_rows=read(HERE/'december-base-0.0-windows.json');comparisons=[]
    for kind in ['all64','interval22','bce','bce_rescue','bce_consensus']:
        best=max([c for c in dec['candidates'] if c['kind']==kind],key=lambda c:c['report']['onset_macro'])
        rows=read(HERE/f"december-{kind}-{best['alpha']}-windows.json")
        train_kind='bce' if kind.startswith('bce_') else kind
        comparisons.append(dict(kind=kind,alpha=best['alpha'],onset_macro=best['report']['onset_macro'],delta=best['report']['onset_macro']-base['report']['onset_macro'],negative_false_any=best['report'].get('negative_onset_false_any_macro'),uncertainty=paired_ci(baseline_rows,rows),training=read(HERE/(train_kind+'-training.json'))))
    audit=read(output/'raw-data-pattern-audit.json');a=audit['audit'];cp=sum(v['complete_positive'] for v in a.values());pp=sum(v['partial_positive'] for v in a.values());bad=sum(v['incompatible22_complete_positive'] for v in a.values())
    selected=read(HERE/'frozen-selection.json');decision='retain R127';january=None
    if (output/'january-frozen-comparison.json').exists():
        january=read(output/'january-frozen-comparison.json');b=january['baseline'];c=january['candidate'];negative_ok=c.get('negative_onset_false_any_macro',0)<=b.get('negative_onset_false_any_macro',0)+.03
        if selected['kind']!='base' and c['onset_macro']>=b['onset_macro']+.004 and negative_ok:decision='candidate passed December/January proxy gates; official gain unverified'
        january=dict(baseline_onset=b['onset_macro'],candidate_onset=c['onset_macro'],delta=c['onset_macro']-b['onset_macro'],negative_false_alarm_gate=negative_ok,uncertainty=paired_ci(january['baseline_windows'],january['windows']))
    diagnostic=read(output/'task2_rescue_january_diagnostic.json') if (output/'task2_rescue_january_diagnostic.json').exists() else None
    if diagnostic is not None:diagnostic={k:v for k,v in diagnostic.items() if k not in ['windows','baseline_windows']}
    result=dict(decision=decision,baseline_december_onset=base['report']['onset_macro'],comparisons=comparisons,selected=selected,january=january,
        data_findings=dict(training_windows=audit['windows'],sampled_link_views=audit['rows'],complete_positive_views=cp,incomplete_positive_views=pp,incomplete_positive_fraction=pp/(cp+pp),noncontiguous_complete_positive_views=bad,noncontiguous_fraction=bad/cp),
        official_S_queue=None,all_months_previously_used_for_development=True,
        january_rescue_diagnostic=diagnostic,
        negative_cohort_caveat='No observed future queue; partial future coverage means this is not certified latent empty-window truth. False positives count only observed eligible proxy-negative cells.')
    write(output/'task2_experiment_summary.json',result);print(json.dumps(dict(decision=decision,December=[dict(kind=c['kind'],delta=c['delta']) for c in comparisons],January_rescue_delta=None if diagnostic is None else diagnostic['onset_delta'])))
if __name__=='__main__':main()
