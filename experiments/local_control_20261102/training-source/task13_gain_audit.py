"""Paired block uncertainty and concentration audit; report only, no new selection."""
import json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/day1/task13-gain-audit';OUT.mkdir(exist_ok=True)
KEY=['panel','date','month','regime']
def score(f):
    z=f.groupby(['panel','regime'])[['n_targets','speed_sse','flow_sse','fd_numerator','fd_denominator','dN_L1_error','dN_L1_truth']].sum()
    s=.54*np.maximum(0,1-np.sqrt(z.speed_sse/z.n_targets)/25)+.46*np.maximum(0,1-np.sqrt(z.flow_sse/z.n_targets)/600)
    p=(np.maximum(0,1-z.fd_numerator/z.fd_denominator)+2*np.maximum(0,1-z.dN_L1_error/z.dN_L1_truth))/3
    return np.array([s.groupby('panel').mean().mean(),p.groupby('panel').mean().mean()])
def main():
    protocol=dict(method='Paired date-block bootstrap, whole road/date concentration audit; 500 resamples seed20261102. No model selection or per-road routing from this report.',caveat='Conditional on developed55 dates and fixed10 roads; not independent hidden-test uncertainty. Physics is truth-anchored analogue, not official.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    a=pd.read_csv(ROOT/'outputs/day1/task13-all8-fullfield/daily-joint-proxy.csv')
    b=pd.read_csv(ROOT/'outputs/round9/daily-joint-proxy.csv');b=b[b.method=='baseline'].drop(columns='method')
    s=pd.read_csv(ROOT/'outputs/round7/daily-metrics.csv');s=s[s.method=='baseline'][KEY+['n','speed_sse','flow_sse']].rename(columns={'n':'n_targets'})
    b=b.merge(s,on=KEY,validate='one_to_one');assert len(a)==len(b)==550
    a=a.sort_values(KEY).reset_index(drop=True);b=b.sort_values(KEY).reset_index(drop=True);assert a[KEY].equals(b[KEY])
    rows=[]
    for kind in ['panel','date']:
        for value in sorted(a[kind].unique()):
            x=score(a[a[kind]==value]);y=score(b[b[kind]==value]);rows.append(dict(block=kind,value=value,delta_S_state=x[0]-y[0],delta_physics_analogue=x[1]-y[1]))
    pd.DataFrame(rows).to_csv(OUT/'block-gains.csv',index=False)
    rng=np.random.default_rng(20261102);dates=sorted(a.date.unique());aa=[a[a.date==d] for d in dates];bb=[b[b.date==d] for d in dates];draws=[]
    for _ in range(500):
        idx=rng.integers(0,len(dates),len(dates));draws.append(score(pd.concat([aa[i] for i in idx]))-score(pd.concat([bb[i] for i in idx])))
    draws=np.array(draws);delta=score(a)-score(b)
    result=dict(protocol=protocol,delta_state=float(delta[0]),delta_physics_analogue=float(delta[1]),paired_date_bootstrap_95percent=np.quantile(draws,[.025,.975],axis=0).tolist(),positive_blocks={k:int(sum(r['delta_S_state']>0 for r in rows if r['block']==k)) for k in ['panel','date']},official_score=None)
    (OUT/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf8');shutil.copy2(Path(__file__),OUT/Path(__file__).name);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
