"""ASM-inspired directional smoothing; an independent implementation.

Reference: Treiber & Helbing (2002), Reconstructing the Spatio-Temporal Traffic
Dynamics from Stationary Detector Data. This adaptation uses the released graph,
fixed finite support, per-channel masks and link-relative switching thresholds.
"""
from __future__ import annotations

import heapq
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

DEFAULTS=dict(c_free=80.,c_congested=-15.,sigma_km=1.,tau_minutes=10.,radius_km=5.,
              time_radius_minutes=120,threshold_ratio=.6,transition_ratio=.1)
FIELDS=['free','congested','coverage_free','coverage_congested','weight','mixed']


def validate_wave(cfg):
    if not all(np.isfinite(float(cfg[k])) for k in DEFAULTS):raise ValueError('Wave parameters must be finite')
    if cfg['c_free']<=0 or cfg['c_congested']>=0:raise ValueError('Free wave must be positive, congested wave negative')
    if min(cfg[k] for k in ['sigma_km','tau_minutes','radius_km','transition_ratio'])<=0:raise ValueError('Bandwidths must be positive')
    if not 0<cfg['threshold_ratio']<1:raise ValueError('Invalid speed threshold')
    if cfg['time_radius_minutes']<5 or cfg['time_radius_minutes']%5:raise ValueError('Temporal radius must be a positive multiple of five')


def signed_distances(nodes,net,topology,radius_km):
    """dx[receiver,source], positive downstream; traverse actual mainline edges."""
    links=nodes.link_id.astype(str).tolist();positions={link:i for i,link in enumerate(links)}
    if len(positions)!=len(links):raise ValueError('Duplicate node links')
    id_col='mainline_link_id' if 'mainline_link_id' in topology else 'link_id'
    allowed=set(topology[id_col].astype(str));lengths=net.length_km.to_dict()
    if set(links)-allowed:raise ValueError('Sensor link absent from mainline topology')
    if any(not np.isfinite(lengths.get(link,np.nan)) or lengths[link]<=0 for link in allowed):
        raise ValueError('Every mainline link needs positive finite length')
    distance=np.full((len(links),len(links)),np.nan,dtype=np.float64);np.fill_diagonal(distance,0)
    for sign,column in [(-1,'incoming_link_ids'),(1,'outgoing_link_ids')]:
        graph={}
        for _,row in topology.iterrows():
            raw=row.get(column)
            graph[str(row[id_col])]=[] if pd.isna(raw) else [part for part in str(raw).split(';') if part in allowed]
        for origin in links:
            queue=[(0.,origin)];best={}
            while queue:
                dist,link=heapq.heappop(queue)
                if link in best or dist>radius_km:continue
                best[link]=dist
                if link in positions:
                    i,j=positions[origin],positions[link];old=distance[i,j]
                    if not np.isfinite(old) or dist<abs(old):distance[i,j]=sign*dist
                for other in graph.get(link,[]):
                    if other not in best:
                        step=(float(lengths[link])+float(lengths[other]))/2
                        if dist+step<=radius_km:heapq.heappush(queue,(dist+step,other))
    return distance


def kernels(distances,cfg):
    validate_wave(cfg)
    offsets=np.arange(-cfg['time_radius_minutes'],cfg['time_radius_minutes']+1,5,dtype=np.float64)
    dx=distances[:,:,None];result=[]
    for c in [cfg['c_free'],cfg['c_congested']]:
        weights=np.exp(-np.abs(dx)/cfg['sigma_km']-np.abs(offsets-60*dx/c)/cfg['tau_minutes'])
        weights=np.where(np.isfinite(weights),weights,0)
        total=weights.sum((1,2),keepdims=True)
        if (total<=0).any():raise ValueError('Kernel must include self support')
        result.append((weights/total).astype(np.float32))
    return np.stack(result),offsets


class WaveEngine:
    def __init__(self,weights,free_speed,cfg,device):
        validate_wave(cfg)
        self.device=torch.device(device);self.cfg=cfg
        self.weights=torch.as_tensor(weights,dtype=torch.float32,device=self.device)
        self.free_speed=torch.as_tensor(free_speed,dtype=torch.float32,device=self.device)
        if self.weights.ndim!=4 or self.weights.shape[0]!=2 or self.weights.shape[1]!=self.weights.shape[2]:
            raise ValueError('Expected two square station kernels')
        if self.weights.shape[-1]!=2*cfg['time_radius_minutes']//5+1:raise ValueError('Wrong kernel support')
        if self.free_speed.shape!=(self.weights.shape[1],) or not torch.isfinite(self.free_speed).all() or (self.free_speed<=0).any():
            raise ValueError('Invalid free speeds')

    @torch.inference_mode()
    def reconstruct(self,observed,available):
        values=torch.as_tensor(observed,dtype=torch.float32,device=self.device)
        mask=torch.as_tensor(available,dtype=torch.bool,device=self.device)
        if values.ndim!=3 or values.shape[-1]!=2 or values.shape!=mask.shape or values.shape[1]!=len(self.free_speed):
            raise ValueError('Expected [time,station,2] observations and channel masks')
        if not torch.isfinite(values[mask]).all():raise ValueError('Non-finite visible observation')
        values=torch.where(mask,values,0)
        # Five batch members share the same station/time convolution: two
        # numerators, two denominators, and the potential in-day support mass.
        source=torch.cat([values.permute(2,1,0),mask.float().permute(2,1,0),
            torch.ones((1,values.shape[1],values.shape[0]),device=self.device)],dim=0).contiguous()
        estimates=[];coverages=[]
        for weights in self.weights:
            filtered=F.conv1d(source,weights,padding=weights.shape[-1]//2)
            numerator=filtered[:2].permute(2,1,0);denominator=filtered[2:4].permute(2,1,0)
            potential=filtered[4].T[:,:,None]
            estimates.append(torch.where(denominator>1e-8,numerator/denominator.clamp_min(1e-8),float('nan')))
            coverages.append((denominator/potential.clamp_min(1e-8)).clamp(0,1))
        free,congested=estimates
        vf,vc=free[:,:,0],congested[:,:,0]
        minimum=torch.where(torch.isfinite(vf)&torch.isfinite(vc),torch.minimum(vf,vc),
                            torch.where(torch.isfinite(vf),vf,vc))
        cut=self.cfg['threshold_ratio']*self.free_speed
        width=self.cfg['transition_ratio']*self.free_speed
        weight=.5*(1+torch.tanh((cut-minimum)/width))
        weight=torch.where(torch.isfinite(weight),weight,.5)
        mixture=(1-weight[:,:,None])*free+weight[:,:,None]*congested
        mixture=torch.where(torch.isfinite(free)&torch.isfinite(congested),mixture,
                            torch.where(torch.isfinite(free),free,congested))
        arrays=[free,congested,*coverages,weight,mixture]
        return {name:value.cpu().numpy() for name,value in zip(FIELDS,arrays,strict=True)}


def row_features(rows,cols,reconstruction,base):
    """Join only observable wave fields and existing historical/input features."""
    history=base[['speed_history','flow_lane_history']].to_numpy(dtype=float)
    initial=base[['speed_initial','flow_lane_initial']].to_numpy(dtype=float)
    raw=reconstruction['mixed'][rows,cols].astype(float)
    filled=np.where(np.isfinite(raw),raw,history)
    if not np.isfinite(filled).all():raise ValueError('Historical fallback must be finite')
    fields={}
    for ch,name in enumerate(['speed','flow_lane']):
        fields[f'wave_{name}']=filled[:,ch]
        fields[f'wave_{name}_free']=reconstruction['free'][rows,cols,ch]
        fields[f'wave_{name}_congested']=reconstruction['congested'][rows,cols,ch]
        fields[f'wave_{name}_coverage_free']=reconstruction['coverage_free'][rows,cols,ch]
        fields[f'wave_{name}_coverage_congested']=reconstruction['coverage_congested'][rows,cols,ch]
        fields[f'wave_{name}_fallback']=(~np.isfinite(raw[:,ch])).astype(float)
        fields[f'wave_{name}_candidate_difference']=fields[f'wave_{name}_free']-fields[f'wave_{name}_congested']
        fields[f'wave_{name}_minus_initial']=filled[:,ch]-initial[:,ch]
        fields[f'wave_{name}_minus_history']=filled[:,ch]-history[:,ch]
    fields['wave_congestion_weight']=reconstruction['weight'][rows,cols]
    features=pd.DataFrame(fields).astype(np.float32)
    # Standalone output remains float64 through the units conversion.
    return features,np.maximum(filled,0)
