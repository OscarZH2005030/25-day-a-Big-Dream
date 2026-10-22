from __future__ import annotations

import importlib.metadata
from pathlib import Path
import numpy as np
import torch

from ..cache import FeatureCache,snapshot
from ..data import read_csv,sha256
from .core import DEFAULTS,WaveEngine,kernels,signed_distances


def kernel_cache(cache,root,panel,nodes,net,cfg):
    folder=root/'corridors'/panel/'network'
    identity=dict(kind='traffic_wave_kernel',schema='traffic_wave_v1',panel=panel,
        nodes=nodes[['station_id','link_id']].to_dict('records'),wave_parameters={key:cfg[key] for key in DEFAULTS},
        inputs=snapshot([folder/file for file in ['links.csv','fd_parameters.csv','lwr_mainline_topology.csv']]),
        implementation=snapshot([Path(__file__).with_name('core.py')]),numpy=importlib.metadata.version('numpy'))
    def build(destination):
        distance=signed_distances(nodes,net,read_csv(folder/'lwr_mainline_topology.csv'),cfg['radius_km'])
        weights,offsets=kernels(distance,cfg)
        np.savez_compressed(destination/'kernels.npz',weights=weights,distance=distance,offsets=offsets)
        return dict(nodes=len(nodes),free_speed=nodes.link_id.map(net.free_speed_kmh).tolist(),
                    nonzero_station_pairs=int(np.isfinite(distance).sum()),time_taps=len(offsets))
    return cache.get_or_build(identity,build)


def make_engine(folder,manifest,cfg,device):
    with np.load(folder/'kernels.npz',allow_pickle=False) as arrays:weights=arrays['weights']
    return WaveEngine(weights,manifest['metadata']['free_speed'],cfg,device)


def backend_identity(device):
    device=torch.device(device)
    return dict(torch=torch.__version__,device_type=device.type,
        device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
        dtype='float32',cudnn_tf32=False,deterministic=True)


def day_cache(cache,date,grid_folder,grid_manifest,kernel_manifest,engine):
    source=grid_folder/(date+'.npz')
    if date not in grid_manifest['metadata']['dates']:raise ValueError('Day is outside the source grid scope')
    identity=dict(kind='traffic_wave_day',schema='traffic_wave_v1',panel=grid_manifest['identity']['panel'],date=date,
        source_grid_key=grid_manifest['key'],source_archive_sha256=sha256(source),kernel_key=kernel_manifest['key'],
        backend=backend_identity(engine.device),implementation=snapshot([Path(__file__),Path(__file__).with_name('core.py')]),
        input_channels=['visible_speed_kmh','visible_flow_per_lane','availability_masks'],truth_used=False)
    def build(destination):
        # Never access y/supervise, even when present in the shared training archive.
        with np.load(source,allow_pickle=False) as archive:
            values,mask=archive['x'],archive['mask']
        result=engine.reconstruct(values,mask)
        np.savez_compressed(destination/'wave.npz',**result)
        return dict(date=date,shape=list(values.shape),truth_used=False)
    folder,manifest,hit=cache.get_or_build(identity,build)
    with np.load(folder/'wave.npz',allow_pickle=False) as arrays:result={name:arrays[name] for name in arrays.files}
    return result,dict(date=date,key=manifest['key'],hit=hit,source_grid_key=grid_manifest['key'])
