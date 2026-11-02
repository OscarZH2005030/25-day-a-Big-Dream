"""Two-parameter horizon threshold calibration, January only; frozen model."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import train_ongoing_horizon as t
BASE=t.BASE;OUT=BASE/'outputs/ongoing_smooth_threshold_20261101';SOURCE=t.OUT
def write(name,x):t.write(OUT/name,x)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=False)
    protocol=dict(model='Frozen ongoing_horizon_20261031/bce_a100,alpha1,all weights unchanged',onset='early_full unchanged',threshold_family='a+b*(h-17.5)/25 clipped[.05,.75];2parameters;h=5,10,...30',January_only=True,February_parameter_selection=False,intercept_grid=np.round(np.arange(.05,.7501,.025),3).tolist(),slope_grid=[0.,-.05,.05,-.1,.1,-.15,.15],selection='Highest January road-macro window IoU; exact ties prefer smaller absolute slope then lower intercept. February is report-only reused development, not blind.')
    write('protocol.json',protocol);t.check();d,m=t.speed.query('january');p=np.load(SOURCE/'january_bce_a100_probability.npy');best=None;grid=[]
    for slope in protocol['slope_grid']:
        for intercept in protocol['intercept_grid']:
            thresholds=np.clip(intercept+slope*(np.arange(5,31,5)-17.5)/25,.05,.75);s=t.old.score(p,d,m,thresholds);r=dict(intercept=intercept,slope=slope,thresholds=thresholds.tolist(),**t.old.compact(s));grid.append(r)
            key=(s['onset_IoU'],-abs(slope),-intercept)
            if best is None or key>best[0]:best=(key,r,s)
    selected=best[1];write('January_grid.json',grid);write('January_selection.json',dict(selected=selected,score=best[2]));print(json.dumps(dict(stage='JANUARY',selected=selected)),flush=True)
    files=[OUT/'protocol.json',OUT/'January_selection.json',Path(__file__)]+[SOURCE/n for n in ['January_results.json','frozen_onset_hashes.json']]+list(SOURCE.glob('final_*/*.txt'))+list((SOURCE/'base_models').glob('*/*.txt'))
    frozen={str(f.relative_to(BASE)):sha(f) for f in files};write('frozen_before_February.json',frozen)
    for f,h in frozen.items():assert sha(BASE/f)==h
    inc=t.read(SOURCE/'February_results.json');results={}
    for pop in ['new_phase_population','old410']:
        d,m=t.speed.query(pop);p=np.load(SOURCE/f'{pop}_bce_a100_probability.npy');s=t.old.score(p,d,m,np.array(selected['thresholds']));previous=inc[pop]['candidates']['bce_a100'];r=dict(score=s,thresholds=selected['thresholds'],fixed_onset_IoU=previous['fixed_onset_IoU'],overall_IoU=(s['onset_IoU']+previous['fixed_onset_IoU'])/2,IoU_vs_incumbent=t.old.prior.REF.bootstrap(s,previous['score']),exceeds_current_best=s['onset_IoU']>previous['score']['onset_IoU']);results[pop]=r;print(json.dumps(dict(stage='FEBRUARY',population=pop,overall_IoU=r['overall_IoU'],**t.old.compact(s))),flush=True)
    write('February_results.json',results)
    for f,h in frozen.items():assert sha(BASE/f)==h
    t.check();write('verification.json',dict(passed=True,onset_unchanged=True,all_weights_unchanged=True,January_only=True,February_parameter_selection=False,frozen_hashes_verified=True))
    write('manifest.json',{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'})
    if any(r['exceeds_current_best'] for r in results.values()):
        f=BASE/'outputs/retained_ongoing_candidates.json';reg=t.read(f);reg['candidates'].append(dict(candidate=OUT.name,config=selected,populations=results,retained=True));t.write(f,reg)
if __name__=='__main__':main()
