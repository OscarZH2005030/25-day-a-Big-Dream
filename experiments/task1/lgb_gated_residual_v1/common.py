"""Frozen data/provenance, blocked cross-fitting and atomic I/O helpers."""
from pathlib import Path
import contextlib
import hashlib
import json
import os
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = Path(__file__).resolve().parents[1]/'full_dataset_training_v2'
REFERENCE = SOURCE/'runs/all_corridors_time80_20'
FINAL_LGB = HERE.parent/'lightgbm_time80_20/runs/all_corridors_time80_20'
KEYS = ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']
CHANNELS = ['speed_kmh', 'flow_per_lane']
SCALE = np.array([25., 600.])
METHOD = 'lgb_gated_residual'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def save_array(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as stream:
        np.save(stream, value, allow_pickle=False)
    tmp.replace(path)


class RunLock:
    def __init__(self, path): self.path = Path(path); self.handle = None
    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('a+b')
        self.handle.seek(0, 2)
        if self.handle.tell() == 0: self.handle.write(b'0'); self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.handle.close(); self.handle = None
            raise RuntimeError('Another writer owns '+str(self.path)) from error
        return self
    def __exit__(self, *unused):
        if self.handle is not None:
            self.handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close(); self.handle = None


@contextlib.contextmanager
def console_log(path):
    class Tee:
        def __init__(self, terminal, file): self.terminal, self.file = terminal, file
        def write(self, value): self.terminal.write(value); self.file.write(value); self.file.flush()
        def flush(self): self.terminal.flush(); self.file.flush()
    stdout, stderr = sys.stdout, sys.stderr
    with Path(path).open('a', encoding='utf-8') as stream:
        sys.stdout, sys.stderr = Tee(stdout, stream), Tee(stderr, stream)
        try: yield
        finally: sys.stdout, sys.stderr = stdout, stderr


def make_folds(dates, folds=4, inner_days=14):
    """Outer OOF dates never participate in either selection or refitting."""
    dates = list(dates)
    if dates != sorted(set(dates)) or folds < 2: raise ValueError('Invalid ordered dates/folds')
    result = []
    for index, chunk in enumerate(np.array_split(np.array(dates), folds)):
        held = chunk.tolist(); train = [d for d in dates if d not in held]
        if not held or len(train) <= inner_days or inner_days < 1: raise ValueError('Fold too small')
        fit, valid = train[:-inner_days], train[-inner_days:]
        assert set(held).isdisjoint(train) and set(fit).isdisjoint(valid)
        assert set(fit+valid) == set(train)
        result.append(dict(fold=index, predict_dates=held, refit_dates=train,
                           inner_fit_dates=fit, inner_valid_dates=valid))
    assert sorted(d for f in result for d in f['predict_dates']) == dates
    return result


def score(y, pred, regimes, method=METHOD):
    if not np.isfinite(y).all() or not np.isfinite(pred).all(): raise ValueError('Nonfinite score input')
    rows = []
    for regime in ['R1', 'R2', 'R3']:
        keep = np.asarray(regimes) == regime
        if not keep.any(): raise ValueError('No targets for '+regime)
        error = np.sqrt(np.mean((pred[keep]-y[keep])**2, axis=0))
        s = np.maximum(0., 1-error/SCALE)
        rows.append(dict(method=method, regime=regime, n=int(keep.sum()),
            rmse_speed_kmh=float(error[0]), rmse_flow_per_lane=float(error[1]),
            S_speed=float(s[0]), S_flow=float(s[1]), S_state=float(s@np.array([.54, .46]))))
    rows.append(dict(method=method, regime='macro_mean', n=len(y), S_state=float(np.mean([r['S_state'] for r in rows]))))
    return rows


def panel_metadata(reference, final_lgb, plan, panel):
    """Read-only checks; no estimator fitting, tensor imports or GPU initialization."""
    folder = reference/panel; meta = read_json(folder/'cache/ready.json')
    signature = meta['signature']
    for key, value in dict(panel=panel, train_start=plan['training_dates'][0],
        internal_val_start=plan['internal_validation_dates'][0], val_start=plan['holdout_dates'][0],
        val_end='2031-03-01').items():
        if signature[key] != value: raise ValueError('Cache boundary mismatch: '+key)
    if signature['code_sha256'] != [sha(reference.parents[1]/n) for n in ['full_train_worker.py', 'task1_baseline.py']]:
        raise ValueError('Original cache source changed')
    for item in signature['data_files']:
        stat = Path(item['path']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (item['bytes'], item['mtime_ns']):
            raise ValueError('Original data changed: '+item['path'])
    expected = plan['training_dates']+plan['internal_validation_dates']+plan['holdout_dates']
    if meta['dates'] != expected or len(meta['names']) != 51: raise ValueError('Cache date/feature mismatch')
    if not (folder/'COMPLETE.json').exists() or not (final_lgb/panel/'COMPLETE.json').exists():
        raise ValueError('Existing reference runs must be complete')
    other = read_json(final_lgb/'plan.json')
    if Path(other['reference_run']).resolve() != reference.resolve() or other['plan_sha256'] != sha(reference/'plan.json'):
        raise ValueError('LightGBM reference identity mismatch')
    for key in ['history_dates', 'training_dates', 'internal_validation_dates', 'holdout_dates']:
        if other[key] != plan[key]: raise ValueError('LightGBM date split mismatch')
    for date in expected:
        for name in ['features.npy', 'base.npy', 'label.npy', 'visible.npy', 'observed.npy', 'target.npy', 'keys.parquet', 'ii.npy', 'jj.npy']:
            if not (folder/'cache'/date/name).is_file(): raise ValueError('Missing cached file '+date+'/'+name)
    if len(plan['history_dates']) != 28 or plan['history_dates'][-1] >= plan['training_dates'][0]:
        raise ValueError('Unexpected history period')
    mean, scale = np.array(meta['mean']), np.array(meta['scale'])
    if mean.shape != (51,) or scale.shape != (51,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Invalid original feature normalization')
    graph = original_graph(reference, plan, panel)
    first = folder/'cache'/plan['training_dates'][0]
    ii, jj = np.load(first/'ii.npy'), np.load(first/'jj.npy')
    keys = pd.read_parquet(first/'keys.parquet')
    target = np.load(first/'target.npy', mmap_mode='r').astype(bool)
    if len(keys) != len(ii) or int(target.sum()) != len(ii) or not target[jj, ii].all():
        raise ValueError('Cached template mask/key mismatch')
    np.testing.assert_array_equal(keys.link_id.to_numpy(), np.array(graph['links'])[ii])
    np.testing.assert_array_equal(keys.station_id.to_numpy(), np.array(graph['stations'])[ii])
    visible = np.load(first/'visible.npy', mmap_mode='r')
    if visible[jj, ii].any(): raise ValueError('Supervised template targets are visible')
    raw = np.load(first/'features.npy', mmap_mode='r')[:24, :min(64, len(graph['links']))]
    interp = np.load(first/'base.npy', mmap_mode='r')[:24, :min(64, len(graph['links']))]
    smoke = neural_features(raw, interp, interp, mean, scale)
    if smoke.shape[-1] != 106: raise ValueError('Wrong residual input dimension')
    return dict(cache_ready_sha256=sha(folder/'cache/ready.json'), history_sha256=sha(folder/'history.npz'),
        topology_sha256=sha(Path(plan['release_root'])/'corridors'/panel/'network/lwr_mainline_topology.csv'),
        final_lgb_sha256=[sha(final_lgb/panel/(c+'.txt')) for c in CHANNELS], names=meta['names'],
        graph_sha256=hashlib.sha256(graph['graph'].tobytes()+graph['direction'].tobytes()+graph['isolated'].tobytes()).hexdigest(),
        graph_nodes=len(graph['links']), graph_isolated_nodes=int(graph['isolated'].sum()),
        real_cached_slice_dimensions=list(smoke.shape))


def original_graph(reference, plan, panel):
    """Exactly the original global graph, including self and symmetric 1/2-hop edges."""
    with np.load(reference/panel/'history.npz', allow_pickle=False) as p:
        links, stations = p['links'].tolist(), p['stations'].tolist()
    topo = pd.read_csv(Path(plan['release_root'])/'corridors'/panel/'network/lwr_mainline_topology.csv').set_index('mainline_link_id')
    adj = np.eye(len(links), dtype=bool)
    for i, link in enumerate(links):
        ids = str(topo.loc[link, 'incoming_link_ids']).split(';')+str(topo.loc[link, 'outgoing_link_ids']).split(';')
        for j, other in enumerate(links):
            if other in ids: adj[i, j] = True
    adj = adj | adj.T
    isolated = adj.sum(axis=1) == 1
    graph = adj | ((adj.astype('int16')@adj.astype('int16')) > 0)
    assert np.array_equal(isolated, graph.sum(axis=1) == 1)
    order = np.array([topo.loc[link, 'order_index'] for link in links])
    return dict(graph=graph, direction=np.sign(order[None, :]-order[:, None]).astype('int64'),
        isolated=isolated, order=np.argsort(order), links=links, stations=stations)


def neural_features(raw, base, interpolation, mean, scale):
    finite = np.isfinite(raw)
    normalized = np.where(finite, (raw-mean)/scale, 0)
    result = np.concatenate([normalized, finite, base, base-interpolation], axis=-1).astype('float32')
    if not np.isfinite(result).all(): raise ValueError('Nonfinite neural input')
    return result


def baseline_grid(raw, interpolation, visible, observed, models, threads):
    """Predict all context positions using only masked features; never takes labels."""
    flat = raw.reshape(-1, raw.shape[-1])
    correction = np.column_stack([model.predict(flat, num_threads=threads) for model in models]).reshape(interpolation.shape)
    physical = np.maximum(interpolation.astype('float64')*SCALE+correction, 0)
    return np.where(visible, observed, physical/SCALE).astype('float32')
