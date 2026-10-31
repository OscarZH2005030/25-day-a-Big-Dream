"""Recompute the fixed hybrid on original February windows from public raw observations."""
import argparse,copy,json,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'code'))
from hybrid import Hybrid,verify_manifest

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True,help='Extracted unchanged ConditionalPosition-R127 folder');ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--work',type=Path,required=True);ap.add_argument('--population',choices=['new_phase_population','old410','both'],default='both');ap.add_argument('--causality',action='store_true');a=ap.parse_args();a.work.mkdir(parents=True,exist_ok=True)
    count=verify_manifest(HERE);model=Hybrid(HERE)
    repro=HERE.parent/'r127_reproduction_20261027';sys.path.insert(0,str(repro));from repro_common import setup
    parent,sample,_,audit=setup(a.package,a.data_root,a.work/'raw-cache',True,extra_allowed=(HERE,a.work));model.attach_parent(parent);expected=read(HERE/'reports/February_results.json');r127=read(HERE/'reports/R127_baseline.json')['reports'];reports={};checks=[];start=time.time()
    for population in (['new_phase_population','old410'] if a.population=='both' else [a.population]):
        entries=[e for e in read(repro/'indices'/f'february-{population}.json')['samples'] if e['has_future_queue']];groups={};windows=[];counts={'FP':0,'FN':0};seen=set()
        for i,e in enumerate(entries):
            s=sample(e);p=model.probability(s);binary=model.decode(s,p);mask=np.asarray(s['m'],bool);y=np.asarray(s['y'],bool);union=int(((binary|y)&mask).sum());inter=int(((binary&y)&mask).sum());iou=inter/union if union else (1. if mask.any() else 0.);condition='queue_ongoing' if s['condition']==1 else 'queue_onset';key=f"{s['panel']}/{condition}";groups.setdefault(key,[]).append(iou);windows.append(dict(panel=s['panel'],origin=s['origin'],condition=condition,IoU=iou));counts['FP']+=int((binary&~y&mask).sum());counts['FN']+=int((~binary&y&mask).sum())
            if a.causality and key not in seen:
                other=copy.deepcopy(s);other['y']=1-s['y'];other['m']=~s['m'];other['has_future_queue']=not s['has_future_queue'];np.testing.assert_array_equal(p,model.probability(other));seen.add(key);checks.append(dict(population=population,group=key,query_target_mutation_does_not_change_prediction=True))
            if (i+1)%32==0:print(population,i+1,len(entries),flush=True)
        scores={k:float(np.mean(v)) for k,v in groups.items()};assert len(scores)==16
        for key,value in scores.items():
            panel,condition=key.split('/');golden=expected[population]['candidates']['early_full']['score']['per_road'][panel] if condition=='queue_onset' else r127[population]['by_panel_condition'][key];assert abs(value-golden)<1e-12,(key,value,golden)
        overall=float(np.mean(list(scores.values())));onset=float(np.mean([v for k,v in scores.items() if k.endswith('/queue_onset')]));ongoing=float(np.mean([v for k,v in scores.items() if k.endswith('/queue_ongoing')]));assert abs(overall-(onset+ongoing)/2)<1e-12
        reports[population]=dict(overall_proxy_IoU=overall,onset_proxy_IoU=onset,ongoing_proxy_IoU=ongoing,positive_windows=len(entries),by_panel_condition=scores,cell_counts=counts,window_scores=windows);(a.work/'metrics.json').write_text(json.dumps(reports,indent=2),encoding='utf8');print(population,overall,onset,ongoing,flush=True)
    (a.work/'verification.json').write_text(json.dumps(dict(passed=True,published_files_verified=count,all16_panel_condition_scores_match_to_1e12=True,causality_checks=checks,query_data_rebuilt_and_hash_verified=len(audit['generated_day_keys']),seconds=time.time()-start,official_S_queue=None),indent=2),encoding='utf8')
if __name__=='__main__':main()
