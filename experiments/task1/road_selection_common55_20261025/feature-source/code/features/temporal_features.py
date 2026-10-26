"""Offline Task 1 interpolation using only quality-filtered released inputs."""
import numpy as np
import pandas as pd

KEYS = ['timestamp', 'station_id', 'link_id']


def temporal_values(obs, targets):
    obs = obs.copy()
    for key in KEYS:
        obs[key] = obs[key].astype(str)
    grid = sorted(obs.timestamp.unique())
    instants = pd.to_datetime(grid, utc=True)
    if len(grid)>1 and not np.all(np.diff(instants.asi8)==300_000_000_000):
        raise ValueError('Expected a complete five-minute daily grid')
    local = np.full((len(targets),2),np.nan)
    for ch,column in enumerate(['speed_kmh','flow_vph']):
        valid = obs[column].where(obs.pct_observed>=75)
        wide = obs.assign(value=valid).pivot(index='timestamp',columns='link_id',values='value').reindex(grid)
        tick = np.arange(len(wide))[:,None]
        seen = pd.DataFrame(np.where(wide.notna(),tick,np.nan),index=wide.index,columns=wide.columns)
        near = (tick-seen.ffill().to_numpy()<=6)&(seen.bfill().to_numpy()-tick<=6)
        result = wide.interpolate(method='linear',axis=0,limit_area='inside').where(near)
        ti = pd.Index(wide.index).get_indexer(targets.timestamp.astype(str))
        ci = pd.Index(wide.columns).get_indexer(targets.link_id.astype(str))
        if (ti<0).any() or (ci<0).any(): raise ValueError('Target outside observation grid')
        local[:,ch] = result.to_numpy()[ti,ci]
    return local


def replace_temporal(obs, targets, base):
    local = temporal_values(obs,targets)
    return np.where(np.isfinite(local),local,base),np.isfinite(local).sum(axis=0)
