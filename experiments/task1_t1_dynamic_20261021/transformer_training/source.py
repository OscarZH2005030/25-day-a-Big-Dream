from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from task1.cache import FeatureCache,cache_key,snapshot
from task1.transformer_priority.common import read,sha256
from task1.wave_neural.cache import NeuralCache
from transformer_fusion.source import Source,verify


class ReadThroughCache(NeuralCache):
    def __init__(self,root,original_roots):
        super().__init__(root)
        self.original_roots=list(map(Path,original_roots))

    def get_or_build(self,identity,builder):
        key=cache_key(identity)
        for root in self.original_roots:
            folder=root/key
            if folder.exists():
                manifest=FeatureCache(root)._validate(folder,identity)
                return folder,manifest,True
        return super().get_or_build(identity,builder)


class TrainingSource(Source):
    def __init__(self,*args,fusion_run,**kwargs):
        super().__init__(*args,**kwargs)
        self.fusion_run=Path(fusion_run)
        fusion_cfg=read(self.fusion_run/'run_config.json')
        self.cache=ReadThroughCache(self.cache.root,[fusion_cfg['cache_dir'],self.cfg['cache_dir']])
        self.date_regime={}

    def prepare(self,panel):
        data=super().prepare(panel)
        provenance=data.provenance
        original_cache=Path(self.cfg['cache_dir'])
        template_folder=original_cache/provenance['template_key']
        metadata={r['date']:r for r in read(template_folder/'manifest.json')['metadata']['dates']}
        dates=self.plan['training_dates']
        if self.smoke:
            dates=sorted({d for regime in ['R1','R2','R3'] for d in
                          [x for x in dates if regime in metadata[x]['regimes']][:2]})
        mapping={}
        for index,date in enumerate(dates):
            regimes=list(metadata[date]['regimes'])
            if len(regimes)!=1 or regimes[0] not in ['R1','R2','R3']:
                raise ValueError('Training date must have exactly one regime')
            mapping[date]=regimes[0]
            record=provenance['days'][date]
            manifest=self.cache_manifest(record['folder'])
            identity=manifest['identity']
            if identity['panel']!=panel or identity['date']!=date or manifest['key']!=record['key']:
                raise ValueError('Training feature identity mismatch')
            for name in ['masked','network','implementation']:
                verify(identity[name]);self.add(identity[name])
            keys=pd.read_parquet(Path(record['folder'])/'keys.parquet')
            expected=pd.read_parquet(template_folder/(date+'.parquet'))
            if not keys.equals(expected): raise ValueError('Training template alignment differs')
            data.records[date]=record
            if (index+1)%16==0:
                print('[training cache]',panel,f'{index+1}/{len(dates)}',flush=True)
        if set(mapping.values())!={'R1','R2','R3'}: raise ValueError('Training regime missing')
        data.roles['training']=dates
        self.roles[panel]['training']=dates
        self.date_regime[panel]=mapping
        history_folder=original_cache/provenance['history_key']
        with np.load(history_folder/'history.npz',allow_pickle=False) as z:
            if list(z['stations'])!=data.exp.stations or list(z['links'])!=data.exp.links:
                raise ValueError('History node order differs')
            data.exp.history=z['mean'].copy()
        return data

    def finish(self):
        for data in self.data.values():
            for entry in data.label_records.values():
                verify(entry['source']);self.add(entry['source'])
        super().finish()
