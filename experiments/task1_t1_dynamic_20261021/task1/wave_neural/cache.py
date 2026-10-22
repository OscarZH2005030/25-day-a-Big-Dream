"""Atomic new-cache publishing with bounded Windows file-lock retries."""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from ..cache import FeatureCache,cache_key
from ..data import save_json,sha256

PUBLISH_DELAYS=(.05,.1,.2,.4,.8,1.6,2.,2.)


def replace_with_retry(source,destination,on_existing=None):
    for attempt in range(len(PUBLISH_DELAYS)+1):
        try:
            os.replace(source,destination)
            return True
        except OSError as error:
            if on_existing is not None and Path(destination).exists():
                on_existing()
                return False
            if getattr(error,'winerror',None) not in (5,32) or attempt==len(PUBLISH_DELAYS):
                raise
            print(f'[cache] Windows publish lock; retry {attempt+1}/{len(PUBLISH_DELAYS)}',flush=True)
            time.sleep(PUBLISH_DELAYS[attempt])
    raise RuntimeError('Unreachable atomic publish state')


def save_checkpoint(path,payload):
    import torch
    path=Path(path);temporary=path.with_name(path.name+'.tmp_'+uuid.uuid4().hex)
    torch.save(payload,temporary)
    replace_with_retry(temporary,path)


def save_history(path,table):
    path=Path(path);temporary=path.with_name(path.name+'.tmp_'+uuid.uuid4().hex)
    table.to_csv(temporary,index=False)
    replace_with_retry(temporary,path)


class NeuralCache(FeatureCache):
    def get_or_build(self,identity,builder):
        key=cache_key(identity);folder=self.root/key
        if folder.exists():
            manifest=self._validate(folder,identity)
            print(f'[cache] HIT {identity["kind"]} {identity["panel"]} {key[:12]}',flush=True)
            return folder,manifest,True
        self.root.mkdir(parents=True,exist_ok=True)
        temporary=self.root/('.tmp_'+uuid.uuid4().hex);temporary.mkdir()
        print(f'[cache] MISS {identity["kind"]} {identity["panel"]} {key[:12]}',flush=True)
        metadata=builder(temporary)
        artifacts=[dict(name=str(p.relative_to(temporary)),sha256=sha256(p)) for p in sorted(temporary.rglob('*')) if p.is_file()]
        manifest=dict(key=key,identity=identity,metadata=metadata,artifacts=artifacts)
        save_json(temporary/'manifest.json',manifest)
        published=replace_with_retry(temporary,folder,on_existing=lambda:self._validate(folder,identity))
        return folder,self._validate(folder,identity),not published
