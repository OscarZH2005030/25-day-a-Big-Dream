from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import importlib.metadata
import json
import sys
import numpy as np
import pandas as pd
from ..cache import snapshot, cache_key
from ..data import KEY, TARGETS, save_json, sha256
from ..wave_neural.cache import NeuralCache

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
COMMIT = 'ad266f9fc2735178309fdab14d0a016cefffaf49'
METHODS = ['T0_original', 'T1_balanced', 'T2_reliable', 'T3_delayed']
PILOT = ['D7_I405_N', 'D7_I405_S']
SCALE = np.array([25., 600.])
LAGS = [-6, -3, -1, 0, 1, 3, 6]
FROZEN = dict(window=24, patch_nodes=64, width=64, heads=4, layers=3, dropout=.1,
              batch_size=2, seed=2026, threads=4, epochs=60, steps_per_epoch=1024,
              patience=8, learning_rate=.0003, weight_decay=.01, clip=1.)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def inside(path):
    path = Path(path).resolve()
    if not path.is_relative_to(PROJECT):
        raise ValueError('Path must stay inside competition project: '+str(path))
    return path


def versions():
    return {p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','pyarrow','lightgbm']}


def implementation():
    return snapshot(list(HERE.glob('*.py'))+list((HERE/'upstream').glob('*.py'))+
                    [PROJECT/'task1'/p for p in ['cache.py','data.py','metrics.py','wave_neural/cache.py']])


@lru_cache(None)
def upstream():
    expected = read(HERE/'upstream_sha256.json')
    for name, digest in expected.items():
        if sha256(HERE/'upstream'/name)!=digest:
            raise ValueError('Original reference source changed: '+name)
    aliases = ['training_io', 'task1_baseline']
    previous = {name:sys.modules.get(name) for name in aliases}
    loaded = {}
    try:
        for name in aliases+['full_train_worker']:
            spec = importlib.util.spec_from_file_location('_priority_upstream_'+name, HERE/'upstream'/(name+'.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded[name] = module
            if name in aliases:
                sys.modules[name] = module
    finally:
        for name in aliases:
            if previous[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous[name]
    return SimpleNamespace(worker=loaded['full_train_worker'], baseline=loaded['task1_baseline'], lock=loaded['training_io'].RunLock)


def frozen_plan():
    plan = read(HERE/'reference_split.json')
    columns = ['history_dates','training_dates','internal_validation_dates','holdout_dates']
    dates = [plan[k] for k in columns]
    if [len(x) for x in dates]!=[28,168,22,55] or sum(dates,[])!=pd.date_range('2030-06-01','2031-02-28').strftime('%Y-%m-%d').tolist():
        raise ValueError('Reference chronological split changed')
    return {k:plan[k] for k in columns+['panels']}


def device_for(name):
    import torch
    if name=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    return torch.device(name)


def metric_rows(y, prediction, regimes, method, panel, family, require_all=True):
    if y.shape!=prediction.shape or y.ndim!=2 or y.shape[1]!=2 or len(regimes)!=len(y):
        raise ValueError('Metric shapes differ')
    if not np.isfinite(y).all() or not np.isfinite(prediction).all() or (prediction<0).any():
        raise ValueError('Invalid metric input')
    rows=[]
    for regime in ['R1','R2','R3']:
        keep = np.asarray(regimes)==regime
        if not keep.any():
            if require_all:
                raise ValueError('Missing regime: '+regime)
            continue
        v,q = np.sqrt(((prediction[keep]-y[keep])**2).mean(0))
        rows.append(dict(method=method,panel=panel,family=family,regime=regime,n=int(keep.sum()),
                         rmse_speed_kmh=float(v),rmse_flow_per_lane=float(q),
                         S_state=float(.54*max(0,1-v/25)+.46*max(0,1-q/600))))
    return rows


def aligned(expected, prediction):
    if expected.duplicated(KEY).any() or prediction.duplicated(KEY).any():
        raise ValueError('Duplicate keys')
    check=expected[KEY].merge(prediction[KEY],on=KEY,how='outer',indicator=True,validate='one_to_one')
    if not check['_merge'].eq('both').all():
        raise ValueError('Prediction target coverage differs')
    result=expected[KEY].merge(prediction[KEY+TARGETS],on=KEY,validate='one_to_one',sort=False)
    if not np.isfinite(result[TARGETS]).all().all() or result[TARGETS].lt(0).any().any():
        raise ValueError('Nonfinite or negative predictions')
    return result


def route(expected, neural, lightgbm, isolated):
    neural, lightgbm = aligned(expected,neural), aligned(expected,lightgbm)
    mask=np.asarray(isolated,dtype=bool)
    if len(mask)!=len(expected):
        raise ValueError('Isolation scope mismatch')
    result=neural.copy()
    result.loc[mask,TARGETS]=lightgbm.loc[mask,TARGETS].to_numpy()
    if not np.array_equal(result.loc[mask,TARGETS].to_numpy(),lightgbm.loc[mask,TARGETS].to_numpy()):
        raise ValueError('Isolated fallback changed')
    if not np.array_equal(result.loc[~mask,TARGETS].to_numpy(),neural.loc[~mask,TARGETS].to_numpy()):
        raise ValueError('Nonisolated neural output changed')
    return result
