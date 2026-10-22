"""Read-only access to the finished pilot; inference writes only new caches."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import gc
import numpy as np
import pandas as pd
import torch

from task1.cache import FeatureCache, snapshot
from task1.data import KEY, TARGETS
from task1.transformer_priority.common import (PILOT, FROZEN, COMMIT, aligned, frozen_plan,
    read, save_json, sha256, upstream, versions)
from task1.transformer_priority.data import CachedData, graph_from
from task1.transformer_priority.model import model_for
from task1.transformer_priority.train import checkpoint_read, predict
from task1.transformer_priority.lightgbm import read_booster
from task1.transformer_priority.runner import context
from task1.wave_neural.cache import NeuralCache


def verify(items):
    for item in items:
        path = Path(item['path'])
        if not path.is_file() or sha256(path) != item['sha256']:
            raise ValueError('Protected file changed/missing: '+str(path))


class Source:
    def __init__(self, source, project, cache, output, smoke, device):
        self.source, self.project = Path(source), Path(project)
        self.cache = NeuralCache(cache)
        self.output, self.smoke = Path(output), smoke
        self.device = torch.device(device)
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; choose --device cpu')
        torch.set_num_threads(4)
        self.files, self.usage = {}, []
        self.cfg = self.protected('run_config.json', anchor=True)
        complete = self.protected('COMPLETE.json', anchor=True)
        manifest_path = self.source/'artifact_manifest.json'
        if sha256(manifest_path) != complete['artifact_manifest_sha256']:
            raise ValueError('Source completion manifest changed')
        self.protected('artifact_manifest.json', anchor=True)
        self.manifest = {str(Path(x['path']).resolve()): x for x in read(manifest_path)['artifacts']}
        self.protected_path('run_config.json')
        if (complete['experiment'] != 'transformer_priorities' or complete['smoke'] or
                complete['panels'] != PILOT or not complete['integrity_passed'] or not complete['all_regimes_present']):
            raise ValueError('Source must be the completed formal two-direction pilot')
        if (self.cfg['commit'] != COMMIT or self.cfg['panels'] != PILOT or
                self.cfg['split'] != frozen_plan() or self.cfg['training'] != dict(FROZEN, eval_batch=2)):
            raise ValueError('Source split/training recipe differs')
        verify(self.cfg['implementation'])
        self.add(self.cfg['implementation'])
        self.plan = self.cfg['split']
        self.records = self.protected('cache_usage.json')
        self.root = Path(self.cfg['release_root'])
        self.source_metrics = self.protected_path('metrics_by_panel_regime.csv')
        self.ref = pd.read_csv(project/'task1/transformer_priority/reference_routing_counts.csv').set_index('panel')
        self.data = {}
        self.roles = {}

    def add(self, items):
        self.files.update({str(Path(x['path']).resolve()): x for x in items})

    def protected_path(self, relative):
        path = (self.source/relative).resolve()
        if not path.is_relative_to(self.source.resolve()):
            raise ValueError('Source path escaped')
        expected = self.manifest.get(str(path))
        if expected is None:
            raise ValueError('File not protected by source completion: '+str(path))
        verify([expected])
        self.add([expected])
        return path

    def protected(self, relative, anchor=False):
        path = self.source/relative if anchor else self.protected_path(relative)
        if anchor:
            self.add(snapshot([path]))
        return read(path)

    def cache_manifest(self, folder):
        folder = Path(folder)
        if not folder.resolve().is_relative_to(self.project.resolve()):
            raise ValueError('Original cache escaped project')
        manifest = read(folder/'manifest.json')
        FeatureCache(folder.parent)._validate(folder, manifest['identity'])
        self.add(snapshot([folder/'manifest.json']+[folder/x['name'] for x in manifest['artifacts']]))
        return manifest

    def prepare(self, panel):
        provenance = self.protected(str(Path(panel)/'feature_cache.json'))
        if provenance['roles'] != dict(training=self.plan['training_dates'], internal=self.plan['internal_validation_dates'], holdout=self.plan['holdout_dates']):
            raise ValueError('Source feature date roles differ')
        original_cache = Path(self.cfg['cache_dir'])
        template_folder = original_cache/provenance['template_key']
        template_manifest = read(template_folder/'manifest.json')
        # Validate the complete template cache, whose daily keys anchor coverage.
        self.cache_manifest(template_folder)
        verify(template_manifest['identity']['source'])
        self.add(template_manifest['identity']['source'])
        metadata = {r['date']: r for r in template_manifest['metadata']['dates']}
        roles = {}
        for role, field in [('internal', 'internal_validation_dates'), ('holdout', 'holdout_dates')]:
            dates = self.plan[field]
            roles[role] = sorted({next(d for d in dates if regime in metadata[d]['regimes']) for regime in ['R1','R2','R3']}) if self.smoke else dates
        self.roles[panel] = roles
        args = SimpleNamespace(release_root=str(self.root), panel=panel, output=str(self.output/'metadata'/panel), seed=2026)
        exp = upstream().baseline.Experiment(args)
        records = {d: provenance['days'][d] for dates in roles.values() for d in dates}
        for date, record in records.items():
            manifest = self.cache_manifest(record['folder'])
            identity = manifest['identity']
            if manifest['key'] != record['key'] or identity['panel'] != panel or identity['date'] != date:
                raise ValueError('Feature identity mismatch')
            for name in ['masked', 'network', 'implementation']:
                verify(identity[name]); self.add(identity[name])
            keys = pd.read_parquet(Path(record['folder'])/'keys.parquet')
            expected = pd.read_parquet(template_folder/(date+'.parquet'))
            if not keys.equals(expected) or len(keys) != metadata[date]['rows']:
                raise ValueError('Cached daily target keys differ from source template')
            ii = np.load(Path(record['folder'])/'ii.npy')
            jj = np.load(Path(record['folder'])/'jj.npy')
            timestamps = pd.to_datetime(keys.timestamp, utc=True)
            if (not np.array_equal(ii, keys.station_id.map(exp.si).to_numpy()) or
                    not np.array_equal(jj, (timestamps.dt.hour*12+timestamps.dt.minute//5).to_numpy()) or
                    not np.array_equal(keys.link_id.to_numpy(), np.asarray(exp.links)[ii])):
                raise ValueError('Cached node/time indices differ from template')
        norm_folder = original_cache/provenance['normalization_key']
        norm_manifest = self.cache_manifest(norm_folder)
        if norm_manifest['identity']['dates'] != self.plan['training_dates']:
            raise ValueError('Normalization uses wrong dates')
        history = self.cache_manifest(original_cache/provenance['history_key'])
        if history['identity']['dates'] != self.plan['history_dates']:
            raise ValueError('Historical profile uses wrong dates')
        verify(history['identity']['inputs']); self.add(history['identity']['inputs'])
        data = CachedData(exp, self.cache, records, dict(training=[], **roles), graph_from(exp),
                          np.load(norm_folder/'mean.npy'), np.load(norm_folder/'scale.npy'), provenance)
        ref = self.ref.loc[panel]
        if data.n != ref.nodes or data.graph['isolated'].sum() != ref.isolated_nodes:
            raise ValueError('Original graph/isolation changed')
        for role, dates in roles.items():
            ctx = context(data, dates)
            if not np.isfinite(ctx.lanes).all() or (ctx.lanes <= 0).any():
                raise ValueError('Invalid lane count')
            if not self.smoke and role == 'holdout' and (len(ctx) != ref.n or ctx.isolated.sum() != ref.fallback_n):
                raise ValueError('Formal holdout target/isolation counts differ')
        self.data[panel] = data
        # Preflight also proves required models are complete without loading them.
        for method in ['T0_original','T1_balanced']:
            record = self.records[panel]['models'][method]['model']
            folder = Path(record['folder'])
            completion = read(folder/'COMPLETE.json')
            if completion['identity'] != record['key'] or not completion['all_training_finished']:
                raise ValueError('Required model is incomplete')
            verify(completion['artifacts']); self.add(completion['artifacts'])
            if read(folder/'best.json') != record['best']:
                raise ValueError('Required model best checkpoint changed')
        self.cache_manifest(self.records[panel]['lightgbm']['folder'])
        return data

    def saved_holdout(self, panel, data, dates):
        expected = data.expected(dates)
        result = {}
        for alias, name in [('T0','T0_original'), ('T1','T1_balanced'), ('LightGBM','LightGBM')]:
            frame = pd.read_parquet(self.protected_path(str(Path(panel)/(name+'_predictions.parquet'))))
            if not self.smoke:
                aligned(expected, frame)
            frame = frame.loc[frame.timestamp.str[:10].isin(dates)].reset_index(drop=True)
            result[alias] = aligned(expected, frame)[TARGETS].to_numpy(np.float64)
            self.usage.append(dict(panel=panel, role='holdout', method=alias, source='saved_formal_prediction', rows=len(frame), hit=True))
        iso = context(data, dates).isolated.to_numpy(bool)
        for name in ['T0','T1']:
            if not np.array_equal(result[name][iso], result['LightGBM'][iso]):
                raise ValueError('Source isolated fallback differs')
        return result

    def internal_predictions(self, panel, data, dates):
        result = {}
        for alias, name in [('T0','T0_original'), ('T1','T1_balanced'), ('LightGBM','LightGBM')]:
            record = self.records[panel]['lightgbm'] if alias == 'LightGBM' else self.records[panel]['models'][name]['model']
            folder = Path(record['folder'])
            loaded = None
            if alias == 'LightGBM':
                manifest = self.cache_manifest(folder)
                if manifest['key'] != record['key']:
                    raise ValueError('Tree model identity mismatch')
                paths = [folder/(n+'.txt') for n in ['speed_kmh', 'flow_per_lane']]
                model_identity = snapshot(paths)
            else:
                completion = read(folder/'COMPLETE.json')
                if completion['identity'] != record['key'] or not completion['all_training_finished']:
                    raise ValueError('Neural model incomplete')
                verify(completion['artifacts']); self.add(completion['artifacts'])
                pointer = read(folder/'best.json')
                if pointer != record['best']:
                    raise ValueError('Best checkpoint differs from source run')
                identity = read(folder/'identity.json')
                if identity['config'] != self.cfg['training'] or identity['method'] != name or identity['panel'] != panel:
                    raise ValueError('Neural model identity differs')
                model_identity = [pointer]
            frames = []
            for date in dates:
                identity = dict(kind='fusion_internal_prediction_v1', panel=panel, method=alias, date=date,
                    model_key=record['key'], weights=model_identity, input=data.records[date]['key'],
                    normalization=data.provenance['normalization_key'], eval_batch=self.cfg['training']['eval_batch'],
                    device=str(self.device) if alias != 'LightGBM' else 'cpu', versions=versions(),
                    inference_adapter_sha256=sha256(Path(__file__)), implementation=self.cfg['implementation'])
                def build(out, date=date):
                    nonlocal loaded
                    if loaded is None:
                        if alias == 'LightGBM':
                            loaded = [read_booster(p) for p in paths]
                        else:
                            loaded, _ = model_for(self.cfg['training'], name)
                            checkpoint, _ = checkpoint_read(folder, 'best.json', record['key'])
                            loaded.load_state_dict(checkpoint['state_dict'])
                            loaded = loaded.to(self.device)
                    if alias == 'LightGBM':
                        x, base = data.target_arrays(date)
                        values = np.maximum(base+np.column_stack([m.predict(x, num_threads=4) for m in loaded]), 0)
                    else:
                        values, _ = predict(loaded, data, [date], self.cfg['training'], self.device, False)
                    keys = data.expected([date])
                    keys['speed_kmh'] = values[:, 0]
                    keys['flow_vph'] = values[:, 1]*data.exp.lanes[data.array(date, 'ii')]
                    keys.to_parquet(out/'predictions.parquet', index=False)
                    return dict(rows=len(keys), training_started=False, date=date)
                cache_folder, manifest, hit = self.cache.get_or_build(identity, build)
                frame = aligned(data.expected([date]), pd.read_parquet(cache_folder/'predictions.parquet'))
                frames.append(frame)
                self.usage.append(dict(panel=panel, role='internal', method=alias, date=date, hit=hit,
                                       source='daily_inference_cache', key=manifest['key'], rows=len(frame)))
            result[alias] = pd.concat(frames, ignore_index=True)[TARGETS].to_numpy(np.float64)
            del loaded
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        return result

    def bundle(self, panel, role, selection_path=None):
        data = self.data[panel]
        dates = self.roles[panel][role]
        if role == 'holdout':
            if selection_path is None or not Path(selection_path).is_file():
                raise ValueError('Freeze internal selection before accessing holdout labels')
            frozen = read(selection_path)
            if frozen['selection_dates'] != self.internal_dates():
                raise ValueError('Frozen selection dates differ')
            verify(frozen['internal_artifacts'])
            predictions = self.saved_holdout(panel, data, dates)
            data.allow_holdout = True
        else:
            predictions = self.internal_predictions(panel, data, dates)
        ctx = context(data, dates)
        truth = data.truth(dates, role)
        for entry in data.label_records.values():
            verify(entry['source']); self.add(entry['source'])
        return dict(context=ctx, truth=truth, **predictions)

    def internal_dates(self):
        return sorted(set(d for roles in self.roles.values() for d in roles['internal']))

    def finish(self):
        verify(list(self.files.values()))
        save_json(self.output/'source_manifest.json', list(self.files.values()))
        save_json(self.output/'cache_usage.json', dict(training_started=False, records=self.usage))
