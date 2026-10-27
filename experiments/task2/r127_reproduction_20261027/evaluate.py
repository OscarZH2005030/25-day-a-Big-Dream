"""Recompute both frozen February proxy populations from public observations."""
import argparse,copy,statistics,time
from pathlib import Path
from repro_common import HERE,read,write,setup
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True);ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--work',type=Path,default=Path.home()/'tfb-r127-work/evaluation');ap.add_argument('--causality',action='store_true');args=ap.parse_args()
    model,sample,r2,audit=setup(args.package,args.data_root,args.work,True)
    import numpy as np
    from task2q.data import make_sample
    reports={};checks=[];expected=read(HERE/'expected-metrics.json')['reports'];start=time.time()
    for name in ['new_phase_population','old410']:
        entries=read(HERE/'indices'/('february-'+name+'.json'))['samples'];samples=[];prob=[];seen=set()
        for i,e in enumerate(entries):
            s=sample(e);p=model.probability(s);samples.append(s);prob.append(p);assert s['origin'].startswith('2031-02')
            group=(s['panel'],s['condition'])
            if args.causality and group not in seen:
                a=copy.deepcopy(r2.arrays(e['day_key']));slot=e['origin_slot']
                for key in ['speed','flow','occupancy','quality','origin_speed','origin_flow','origin_occupancy','origin_quality']:a[key][slot+1:]=999
                a['eligible'][slot+1:]=False;q=make_sample(a,s['network'],slot,s['panel']);assert np.array_equal(p,model.probability(q))
                q=copy.deepcopy(s);q['y']=1-s['y'];q['m']=~s['m'];q['has_future_queue']=not s['has_future_queue'];assert np.array_equal(p,model.probability(q))
                import ramp_features as rf
                original=rf.day;date=s['origin'][:10]
                def perturb(panel,split,dt):
                    values=original(panel,split,dt).copy()
                    if dt>=date:values[slot if dt==date else 0:]=999999
                    return values
                try:rf.day=perturb;assert np.array_equal(p,model.probability(s))
                finally:rf.day=original
                checks.append(dict(population=name,panel=group[0],condition=group[1],passed=True));seen.add(group)
            if (i+1)%50==0:print(f'{name}: {i+1}/{len(entries)}',flush=True)
        result=r2.score(samples,prob,[.5,.25]);golden=expected[name+'_selected'];assert result['n_windows']==golden['n_windows']
        assert set(result['by_panel_condition'])==set(golden['by_panel_condition'])
        for key,value in result['by_panel_condition'].items():assert abs(value-golden['by_panel_condition'][key])<1e-12,(name,key,value,golden['by_panel_condition'][key])
        assert abs(result['proxy_IoU']-golden['proxy_IoU'])<1e-12
        reports[name]=result;write(args.work/'recomputed-metrics.json',dict(reports=reports,official_S_queue=None,February_repeated_development_not_blind=True,populations_not_pooled=True));print(name,result['proxy_IoU'],flush=True)
    write(args.work/'verification.json',dict(passed=True,seconds=time.time()-start,causality_checks=checks,original_model_cache_IO_blocked=True,package_models_only=True,native_models=sorted(set(audit['native_models'])),raw_day_cache_rebuilt_and_content_verified=len(audit['generated_day_keys']),all_panel_condition_scores_exact_to_1e12=True,official_S_queue=None))
if __name__=='__main__':main()
