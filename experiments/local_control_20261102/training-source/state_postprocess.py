"""January-selected, February-report-only masked-input state postprocessing."""
import os,json,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent.parent
DATA=Path('C:/Users/18049/Documents/Codex/2026-10-28/https-github-com-oscarzh2005030-25-day/work/data/kaggle_public')
PRED=ROOT/'work/task1-holdout/evaluation/train'
OUT=ROOT/'outputs/round7';OUT.mkdir(exist_ok=True)
KEYS=['timestamp','station_id','link_id']
CONFIGS=[('baseline','baseline',0),('state_interp10','state',.1),('state_interp25','state',.25),
         ('density_interp10','density',.1),('density_interp25','density',.25)]

def interpolants(masked):
    # This function only receives the masked public observations, never truth.
    f=masked.sort_values(['station_id','link_id','timestamp']).copy()
    f['density_interp']=f.flow_vph/f.speed_kmh.clip(lower=1)
    for source,target in [('speed_kmh','speed_interp'),('flow_vph','flow_interp'),('density_interp','density_interp')]:
        f[target]=f.groupby(['station_id','link_id'])[source].transform(lambda x:x.interpolate(limit_direction='both'))
    return f[KEYS+['speed_interp','flow_interp','density_interp']]

def run():
    print('State postprocess PID',os.getpid(),flush=True)
    while not (PRED/'metrics.json').exists():time.sleep(5)
    source=ROOT/'work/state-postprocess-protocol.json';assert source.exists()
    rows=[]
    for file in sorted(PRED.glob('*.parquet')):
        panel,date=file.stem.rsplit('_',1)
        pred=pd.read_parquet(file);pred.timestamp=pred.timestamp.astype(str);pred.station_id=pred.station_id.astype(str);pred.link_id=pred.link_id.astype(str)
        folder=DATA/'corridors'/panel/'train'
        name='synthetic_mainline_'+date.replace('-','_')+'.parquet'
        masked=pd.read_parquet(next((folder/'mainline_states_masked').rglob(name)))
        for key in KEYS:masked[key]=masked[key].astype(str)
        features=interpolants(masked)
        # Truth is opened only after interpolation features are finalized.
        truth=pd.read_parquet(next((folder/'mainline_states').rglob(name)))
        for key in KEYS:truth[key]=truth[key].astype(str)
        target=pred.merge(features,on=KEYS,validate='one_to_one')
        true_target=pred[KEYS].merge(truth[KEYS+['speed_kmh','flow_vph']],on=KEYS,validate='one_to_one')
        assert len(target)==len(pred)==len(true_target)
        lengths=pd.read_csv(DATA/'corridors'/panel/'network/links.csv',dtype={'link_id':str}).drop_duplicates('link_id').set_index('link_id').length_km
        lanes=pd.read_csv(DATA/'corridors'/panel/'network/fd_parameters.csv',dtype={'link_id':str}).set_index('link_id').lanes
        true_v=true_target.speed_kmh.to_numpy();true_q=true_target.flow_vph.to_numpy();lane=pred.link_id.map(lanes).to_numpy()
        for name,kind,alpha in CONFIGS:
            v=target.speed_kmh.to_numpy().copy();q=target.flow_vph.to_numpy().copy()
            if kind=='state':
                iv=target.speed_interp.to_numpy();iq=target.flow_interp.to_numpy()
                v=np.where(np.isfinite(iv),(1-alpha)*v+alpha*iv,v)
                q=np.where(np.isfinite(iq),(1-alpha)*q+alpha*iq,q)
            elif kind=='density':
                k=q/np.maximum(v,1);ik=target.density_interp.to_numpy();mixed=(1-alpha)*k+alpha*ik
                v=np.where(np.isfinite(mixed)&(mixed>1e-9),q/np.maximum(mixed,1e-9),v)
            density=q/np.maximum(v,1);true_density=true_q/np.maximum(true_v,1)
            values=target[KEYS].copy();values['pv']=v;values['pq']=q;values['target']=True
            full=masked.merge(values,on=KEYS,how='left',validate='one_to_one')
            is_target=full.target.fillna(False).astype(bool)
            full['v']=np.where(is_target,full.pv,full.speed_kmh);full['q']=np.where(is_target,full.pq,full.flow_vph)
            full=full.merge(truth[KEYS+['speed_kmh','flow_vph','is_score_eligible']].rename(columns={'speed_kmh':'tv','flow_vph':'tq','is_score_eligible':'eligible_truth'}),on=KEYS,validate='one_to_one')
            full['Nerror']=(full.q/full.v.clip(lower=1)-full.tq/full.tv.clip(lower=1))*full.link_id.map(lengths)
            full=full[full.eligible_truth.astype(bool)].sort_values(['station_id','link_id','timestamp'])
            group=full.groupby(['station_id','link_id']);error=group.Nerror.diff();dt=group.timestamp.transform(lambda x:pd.to_datetime(x,utc=True).diff().dt.total_seconds())
            touches=is_target.reindex(full.index)|group.target.shift().fillna(False).astype(bool)
            valid=dt.eq(300)&touches&error.notna()
            rows.append(dict(panel=panel,date=date,month=date[:7],regime=str(pred.mask_regime.iloc[0]),method=name,n=len(pred),
                speed_sse=float(np.square(v-true_v).sum()),flow_sse=float(np.square((q-true_q)/lane).sum()),
                density_sse=float(np.square(density-true_density).sum()),dN_sse=float(np.square(error[valid]).sum()),dN_n=int(valid.sum())))
        if len(rows)%100==0:print('days',len(rows)//len(CONFIGS),flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'daily-metrics.csv',index=False)
    report=[]
    for scope in ['2031-01','2031-02','all55']:
        f=frame if scope=='all55' else frame[frame.month==scope]
        for method,g in f.groupby('method'):
            regs=g.groupby(['panel','regime'])[['n','speed_sse','flow_sse','density_sse','dN_sse','dN_n']].sum()
            regs['S_state']=.54*np.maximum(0,1-np.sqrt(regs.speed_sse/regs.n)/25)+.46*np.maximum(0,1-np.sqrt(regs.flow_sse/regs.n)/600)
            score=float(regs.groupby('panel').S_state.mean().mean())
            report.append(dict(scope=scope,method=method,S_state=score,density_RMSE=float(np.sqrt(g.density_sse.sum()/g.n.sum())),
                dN_error_RMSE=float(np.sqrt(g.dN_sse.sum()/g.dN_n.sum())),targets=int(g.n.sum())))
    summary=pd.DataFrame(report);summary.to_csv(OUT/'summary.csv',index=False)
    jan=summary[summary.scope=='2031-01'];baseline=jan[jan.method=='baseline'].iloc[0]
    eligible=jan[(jan.S_state>=baseline.S_state)&(jan.dN_error_RMSE<=baseline.dN_error_RMSE)]
    selected=eligible.sort_values('S_state',ascending=False).iloc[0].method
    (OUT/'selection.json').write_text(json.dumps(dict(selected=selected,selection='January state score with nonworsening dN error; February report only',
        protocol_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),caveat='Previously used model-selection days, not independent test; dN error diagnostic is not official S_physics.'),indent=2),encoding='utf8')
    print(summary.to_string(index=False),flush=True)
    (OUT/'complete.json').write_text(json.dumps(dict(complete=True,official_score=None)),encoding='utf8')

if __name__=='__main__':run()
