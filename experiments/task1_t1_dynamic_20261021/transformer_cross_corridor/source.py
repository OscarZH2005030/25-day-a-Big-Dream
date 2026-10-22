from pathlib import Path
import numpy as np
import pandas as pd
from task1.cache import FeatureCache, snapshot
from task1.data import save_json
from task1.transformer_priority import data as dataset
from task1.transformer_priority.common import read
from transformer_training.source import ReadThroughCache
from transformer_training.sampling import Sampler
from transformer_training.augment import features_from_observations
from transformer_fusion.source import verify


class Prepared:
    def __init__(self, args, split):
        self.args, self.split = args, split
        self.data = {}
        self.samplers = {}
        self.files = {}
        self.cache_records = {}
        self.preflight = {}
        self.parity = {}
        self.reference = pd.read_csv(args.project / 'task1/transformer_priority/reference_routing_counts.csv').set_index('panel')

    def add(self, items):
        self.files.update({x['path']: x for x in items})

    def cache_manifest(self, folder):
        folder = Path(folder)
        manifest = read(folder / 'manifest.json')
        FeatureCache(folder.parent)._validate(folder, manifest['identity'])
        self.add(snapshot([folder / 'manifest.json'] + [folder / x['name'] for x in manifest['artifacts']]))
        return manifest

    def prepare(self, panel):
        data = dataset.prepare(self.args.release_root, panel, self.args.cache_dir, self.split, self.args.smoke)
        data.cache = ReadThroughCache(self.args.cache_dir, [self.args.project / 'cache/transformer_priority',
                                                          self.args.project / 'cache/transformer_training'])
        for key in ['template_key', 'history_key', 'normalization_key']:
            self.cache_manifest(self.args.cache_dir / data.provenance[key])
        self.add(data.provenance['inputs'])
        for record in data.records.values():
            self.cache_manifest(record['folder'])
        metadata = read(self.args.cache_dir / data.provenance['template_key'] / 'manifest.json')['metadata']['dates']
        mapping = {}
        for item in metadata:
            if item['date'] in data.roles['training']:
                if len(item['regimes']) != 1:
                    raise ValueError('Multiple training regimes on one day')
                mapping[item['date']] = next(iter(item['regimes']))
        sampler = Sampler(data, 'A_dynamic', mapping)
        first = data.roles['training'][0]
        rebuilt = features_from_observations(data.exp, first, sampler.observations.get(first), np.arange(data.n))
        for name, values in zip(['features', 'base', 'visible', 'reliability'], rebuilt[:4]):
            np.testing.assert_array_equal(values, data.array(first, name))
        self.parity[panel] = dict(date=first, all_51_features_exact=True)
        ref = self.reference.loc[panel]
        if data.n != ref.nodes or data.graph['isolated'].sum() != ref.isolated_nodes:
            raise ValueError('Original graph/isolated node definition changed')
        if not self.args.smoke:
            holdout_rows = sum(data.records[d]['rows'] for d in data.roles['holdout'])
            isolated_rows = sum(int(data.graph['isolated'][data.array(d, 'ii')].sum()) for d in data.roles['holdout'])
            if holdout_rows != ref.n or isolated_rows != ref.fallback_n:
                raise ValueError('Formal target/fallback counts differ')
        if self.args.action == 'prepare':
            for date in data.roles['training']:
                sampler.observations.ensure(date)
        self.data[panel] = data
        self.samplers[panel] = sampler
        self.preflight[panel] = dict(dates=data.roles, nodes=data.n, isolated_nodes=int(data.graph['isolated'].sum()),
            rows={r:sum(data.records[d]['rows'] for d in ds) for r,ds in data.roles.items()},
            training_regimes=pd.Series(mapping).value_counts().to_dict(),
            holdout_labels_read=data.allow_holdout)
        self.cache_records[panel] = data.provenance
        return data, mapping

    def finish(self, out):
        labels = {}
        observations = {}
        for panel, data in self.data.items():
            labels[panel] = data.label_records
            for record in data.label_records.values():
                self.add(record['source'])
                self.cache_manifest(record['folder'])
            observations[panel] = self.samplers[panel].observations.records
            for record in observations[panel].values():
                self.add(record['source'])
                self.cache_manifest(record['folder'])
        verify(list(self.files.values()))
        save_json(out / 'source_manifest.json', list(self.files.values()))
        save_json(out / 'feature_cache_usage.json', self.cache_records)
        save_json(out / 'label_cache_usage.json', labels)
        save_json(out / 'observation_cache_usage.json', observations)
