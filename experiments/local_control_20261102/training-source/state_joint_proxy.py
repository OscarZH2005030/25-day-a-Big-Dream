"""Truth-anchored conservation analogue, not the hidden official physics score."""
import json,os,sys,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import state_postprocess as prior
ROOT=prior.ROOT;DATA=prior.DATA;PRED=prior.PRED;KEYS=prior.KEYS
OUT=ROOT/'outputs/round9';OUT.mkdir(exist_ok=True)
STATE_SCORE_FILE=ROOT/'outputs/round7/summary.csv'
PROTOCOL_FILE=ROOT/'work/state-joint-protocol.json'
REF=ROOT/'work/repo/experiments/task4/robust_odme_20261027/outputs/official_reference/src'
sys.path.insert(0,str(REF))
from task3.score_task3 import network_parameters

def run():
    print('Joint state proxy PID',os.getpid(),flush=True)
    assert PROTOCOL_FILE.exists()
    params={};rows=[]
    for file in sorted(PRED.glob('*.parquet')):
        panel,date=file.stem.rsplit('_',1);pred=pd.read_parquet(file)
        for key in KEYS:pred[key]=pred[key].astype(str)
        folder=DATA/'corridors'/panel/'train';name='synthetic_mainline_'+date.replace('-','_')+'.parquet'
        masked=pd.read_parquet(next((folder/'mainline_states_masked').rglob(name)))
        for key in KEYS:masked[key]=masked[key].astype(str)
        interp=prior.interpolants(masked);target=pred.merge(interp,on=KEYS,validate='one_to_one')
        # All candidate inputs are finalized before opening this truth.
        truth=pd.read_parquet(next((folder/'mainline_states').rglob(name)))
        for key in KEYS:truth[key]=truth[key].astype(str)
        truth=truth[truth.is_score_eligible.astype(bool)]
        tf=truth.groupby(['timestamp','link_id'],as_index=False)[['speed_kmh','flow_vph']].mean().rename(columns={'speed_kmh':'tv','flow_vph':'tq'})
        if panel not in params:params[panel]=network_parameters(DATA/'corridors'/panel).set_index('link_id')
        par=params[panel]
        for method,kind,alpha in prior.CONFIGS:
            v=target.speed_kmh.to_numpy().copy();q=target.flow_vph.to_numpy().copy()
            if kind=='state':
                a=target.speed_interp.to_numpy();b=target.flow_interp.to_numpy();v=np.where(np.isfinite(a),(1-alpha)*v+alpha*a,v);q=np.where(np.isfinite(b),(1-alpha)*q+alpha*b,q)
            elif kind=='density':
                k=(1-alpha)*q/np.maximum(v,1)+alpha*target.density_interp.to_numpy()
                v=np.where(np.isfinite(k)&(k>1e-9),q/np.maximum(k,1e-9),v)
            update=target[KEYS].copy();update['pv']=v;update['pq']=q
            full=masked.merge(update,on=KEYS,how='left',validate='one_to_one')
            full['v']=full.pv.combine_first(full.speed_kmh);full['q']=full.pq.combine_first(full.flow_vph)
            full=full[full.is_score_eligible.astype(bool)].groupby(['timestamp','link_id'],as_index=False)[['v','q']].mean()
            full=full.merge(tf,on=['timestamp','link_id'],validate='one_to_one').join(par,on='link_id')
            full=full.dropna(subset=['v','q','tv','tq','lanes','capacity_vph','free_speed_kmh','critical_density','k_jam','length_km'])
            k=full.q/full.v.clip(lower=1);kt=full.tq/full.tv.clip(lower=1);lanes=full.lanes.clip(lower=1)
            qlane=full.q/lanes;klane=k/lanes;cap=full.capacity_vph/lanes;kc=full.critical_density/lanes;kj=full.k_jam/lanes
            qfd=np.where(klane<=kc,full.free_speed_kmh*klane,cap/np.maximum(kj-kc,1e-9)*np.maximum(kj-klane,0))
            full['N']=k*full.length_km;full['Nt']=kt*full.length_km
            full=full.sort_values(['link_id','timestamp']);group=full.groupby('link_id');dn=group.N.diff();dnt=group.Nt.diff()
            dt=group.timestamp.transform(lambda x:pd.to_datetime(x,utc=True).diff().dt.total_seconds());valid=dt.eq(300)&dn.notna()&dnt.notna()
            truth_target=target[KEYS].merge(truth[KEYS+['speed_kmh','flow_vph']],on=KEYS,validate='one_to_one')
            lane=target.link_id.map(par.lanes).to_numpy()
            rows.append(dict(panel=panel,date=date,month=date[:7],regime=str(pred.mask_regime.iloc[0]),method=method,
                n_targets=len(target),speed_sse=float(np.square(v-truth_target.speed_kmh.to_numpy()).sum()),
                flow_sse=float(np.square((q-truth_target.flow_vph.to_numpy())/lane).sum()),
                fd_numerator=float(np.abs(qlane-qfd).sum()),fd_denominator=float(np.abs(qlane).sum()),
                dN_L1_error=float(np.abs(dn[valid]-dnt[valid]).sum()),dN_L1_truth=float(np.abs(dnt[valid]).sum()),transitions=int(valid.sum())))
        if len(rows)%100==0:print('days',len(rows)//len(prior.CONFIGS),flush=True)
    daily=pd.DataFrame(rows);daily.to_csv(OUT/'daily-joint-proxy.csv',index=False)
    state=pd.read_csv(STATE_SCORE_FILE).rename(columns={'variant':'method'}) if STATE_SCORE_FILE is not None else None;report=[]
    for scope in ['2031-01','2031-02','all55']:
        f=daily if scope=='all55' else daily[daily.month==scope]
        for method,g in f.groupby('method'):
            x=g.groupby(['panel','regime'])[['fd_numerator','fd_denominator','dN_L1_error','dN_L1_truth']].sum()
            x['S_FD']=np.maximum(0,1-x.fd_numerator/x.fd_denominator)
            x['S_LWR_analogue']=np.maximum(0,1-x.dN_L1_error/x.dN_L1_truth)
            x['physics_analogue']=(x.S_FD+2*x.S_LWR_analogue)/3
            physics=float(x.groupby('panel').physics_analogue.mean().mean())
            if state is None:
                z=g.groupby(['panel','regime'])[['n_targets','speed_sse','flow_sse']].sum()
                ss=.54*np.maximum(0,1-np.sqrt(z.speed_sse/z.n_targets)/25)+.46*np.maximum(0,1-np.sqrt(z.flow_sse/z.n_targets)/600)
                s=float(ss.groupby('panel').mean().mean())
            else:s=float(state[(state.scope==scope)&(state.method==method)].S_state.iloc[0])
            report.append(dict(scope=scope,method=method,S_state=s,physics_analogue=physics,partial_total_analogue=.35*s+.15*physics))
    summary=pd.DataFrame(report);summary.to_csv(OUT/'summary.csv',index=False)
    chosen=summary[summary.scope=='2031-01'].sort_values('partial_total_analogue',ascending=False).iloc[0].method
    (OUT/'selection.json').write_text(json.dumps(dict(selected_by_January=chosen,official_score=None,
        warning='Truth-anchored conservation analogue over all eligible link transitions. Does not reproduce hidden boundary flows or official ramp/topology validity mask. Cannot certify official improvement.'),indent=2),encoding='utf8')
    print(summary.to_string(index=False),flush=True)
    (OUT/'complete.json').write_text(json.dumps(dict(complete=True,official_score=None)),encoding='utf8')

if __name__=='__main__':run()
