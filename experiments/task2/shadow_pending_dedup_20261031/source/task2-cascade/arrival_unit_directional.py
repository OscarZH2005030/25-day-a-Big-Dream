"""Direction-separated ablation; advance-only cannot remove any parent TP."""
import argparse,json
from pathlib import Path
import lightgbm as lgb
from core import r
import arrival_unit as unit
from arrival_committee import Consensus
from arrival_safe import Keep

def models(name):
    out={}
    for d in range(2):
        files=[unit.HERE/f'{name}-committee-{k}-{d}.txt' for k in range(3)]
        out[d]=Consensus([lgb.Booster(model_file=str(f)) for f in files]) if all(f.exists() for f in files) else Keep()
    return {'committee':out,'committee_advance':{0:out[0]},'committee_delay':{1:out[1]}}

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();backend=r.setup(args);internal=unit.evaluate(backend,'november','early',models('probe'));b=internal['base']['scores'];eligible=[k for k,v in internal.items() if k not in ['base','margin'] and v['scores']['onset_macro']>b['onset_macro'] and v['scores']['FP']<=b['FP'] and v['scores']['TP']>=b['TP'] and v['scores']['negative_false_any']<=b['negative_false_any']];selected=max(eligible,key=lambda k:internal[k]['scores']['onset_macro']) if eligible else 'base';r.write(args.output/'task2_arrival_unit_directional_internal.json',dict(reports=internal,selected=selected,later_month_selection=False,all_evaluations_reused_development=True));r.write(unit.HERE/'directional-frozen-policy.json',dict(selected=selected,thresholds=dict(parent=.5,ongoing=.25,gate=.5),later_month_selection=False));reports={};full=models('full')
    for pop in ['december','january','february_new','february_old']:
        reports[pop]=unit.evaluate(backend,pop,'full',full);r.write(args.output/'task2_arrival_unit_directional_external.json',dict(reports=reports,frozen_selection=selected,all_evaluations_reused_development=True,official_truth_available=False));print(json.dumps(dict(population=pop,selected=selected,scores={k:{j:v['scores'][j] for j in ['proxy_IoU','onset_macro','TP','FP','FN','lost_original_R127_TP']} for k,v in reports[pop].items()})),flush=True)

if __name__=='__main__':main()
