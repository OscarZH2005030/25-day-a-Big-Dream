"""Read disk slices; training labels come only from compact OOF residual targets."""
from collections import OrderedDict
import numpy as np
import torch
from common import neural_features, original_graph, read_json


def graph_tensors(graph, nodes):
    nodes = np.array(nodes)
    sub = graph['graph'][np.ix_(nodes, nodes)]
    width = int(sub.sum(axis=1).max())
    neighbors = np.zeros((len(nodes), width), dtype='int64')
    valid = np.zeros_like(neighbors, dtype=bool)
    direction = np.zeros_like(neighbors)
    for i, row in enumerate(sub):
        ids = np.flatnonzero(row)
        neighbors[i, :len(ids)] = ids
        valid[i, :len(ids)] = True
        direction[i, :len(ids)] = graph['direction'][nodes[i], nodes[ids]]+1
    return dict(neighbors=torch.from_numpy(neighbors), neighbor_valid=torch.from_numpy(valid),
        direction=torch.from_numpy(direction), isolated=torch.from_numpy(graph['isolated'][nodes].copy()))


class DayStore:
    def __init__(self, args, plan, panel):
        self.args, self.plan, self.panel = args, plan, panel
        self.source = args.reference_run/panel/'cache'
        self.baselines = args.output/panel/'baseline'
        self.meta = read_json(self.source/'ready.json')
        self.mean, self.scale = np.array(self.meta['mean']), np.array(self.meta['scale'])
        self.graph = original_graph(args.reference_run, plan, panel)
        self.n = len(self.graph['links'])
        self.input_names = self.meta['names']+[n+'__finite' for n in self.meta['names']]+[
            'lgb_speed_scaled', 'lgb_flow_per_lane_scaled',
            'lgb_minus_interpolation_speed_scaled', 'lgb_minus_interpolation_flow_per_lane_scaled']
        self._residual = OrderedDict()

    def block(self, date, start, nodes, supervised=False):
        source, baseline = self.source/date, self.baselines/date
        selection = (slice(start, start+self.args.window), nodes)
        raw = np.load(source/'features.npy', mmap_mode='r')[selection]
        interp = np.load(source/'base.npy', mmap_mode='r')[selection]
        base = np.load(baseline/'base.npy', mmap_mode='r')[selection]
        visible = np.load(source/'visible.npy', mmap_mode='r')[selection].astype(bool)
        result = dict(features=neural_features(raw, base, interp, self.mean, self.scale), base=base, visible=visible)
        if supervised:
            if date not in self.plan['training_dates']: raise ValueError('Residual learning requested outside supervised days')
            result['target'] = np.load(source/'target.npy', mmap_mode='r')[selection].astype(bool)
            if date not in self._residual:
                ii, jj = np.load(source/'ii.npy'), np.load(source/'jj.npy')
                compact = np.load(baseline/'residual_targets.npy')
                if compact.shape != (len(ii), 2): raise ValueError('Residual/key shape mismatch')
                full_mask = np.load(source/'target.npy', mmap_mode='r').astype(bool)
                if int(full_mask.sum()) != len(ii) or not full_mask[jj, ii].all(): raise ValueError('Template mask/key mismatch')
                if not np.isfinite(compact).all(): raise ValueError('Nonfinite OOF residual labels')
                r = np.zeros((288, self.n, 2), dtype='float32'); r[jj, ii] = compact
                self._residual[date] = r
                if len(self._residual) > 8: self._residual.popitem(last=False)
            self._residual.move_to_end(date)
            result['residual'] = self._residual[date][selection]
        return {k:torch.from_numpy(np.ascontiguousarray(v)) for k, v in result.items()}

    def sample(self, rng):
        n = min(self.n, self.args.patch_nodes)
        for _ in range(128):
            start = int(rng.integers(self.n-n+1)); nodes = self.graph['order'][start:start+n]
            parts = []
            for _ in range(self.args.batch_size):
                date = self.plan['training_dates'][int(rng.integers(len(self.plan['training_dates'])))]
                tick = int(rng.integers(289-self.args.window))
                parts.append(self.block(date, tick, nodes, supervised=True))
            batch = {k:torch.stack([p[k] for p in parts]) for k in parts[0]}
            if batch['target'].any(): return batch, graph_tensors(self.graph, nodes)
        raise RuntimeError('No template targets found after 128 block draws')
