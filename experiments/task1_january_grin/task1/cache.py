"""Content-addressed feature caches. Incomplete or modified entries are never hits."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import uuid
from pathlib import Path

from .data import save_json, sha256


class CacheIntegrityError(RuntimeError):
    pass


def cache_key(identity):
    payload = json.dumps(identity, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def snapshot(paths):
    """Hash full contents, not just sizes/mtimes (including raw-data changes)."""
    return [{'path': str(p.resolve()), 'bytes': p.stat().st_size, 'sha256': sha256(p)}
            for p in sorted(set(map(Path, paths)), key=lambda p: str(p.resolve()))]


def feature_implementation():
    folder = Path(__file__).parent
    files = ['data.py', 'features.py', 'pipeline.py', 'cache.py', 'cache_builder.py']
    return dict(schema='task1_feature_superset_v1', sources=snapshot(folder / f for f in files),
                packages={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'pyarrow']})


class FeatureCache:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _validate(self, folder, identity):
        manifest_path = folder / 'manifest.json'
        if not manifest_path.is_file():
            raise CacheIntegrityError(f'Incomplete cache: {folder}. Choose a new --cache-dir.')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('identity') != identity or manifest.get('key') != cache_key(identity):
            raise CacheIntegrityError(f'Cache identity mismatch: {folder}')
        for artifact in manifest['artifacts']:
            path = folder / artifact['name']
            if not path.resolve().is_relative_to(folder.resolve()) or not path.is_file() or sha256(path) != artifact['sha256']:
                raise CacheIntegrityError(f'Cache artifact changed/missing: {path}. Choose a new --cache-dir.')
        return manifest

    def get_or_build(self, identity, builder):
        key = cache_key(identity)
        folder = self.root / key
        if folder.exists():
            manifest = self._validate(folder, identity)
            print(f'[cache] HIT {identity["kind"]} {identity["panel"]} {key[:12]}', flush=True)
            return folder, manifest, True
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.root / ('.tmp_' + uuid.uuid4().hex)
        temporary.mkdir()
        print(f'[cache] MISS {identity["kind"]} {identity["panel"]} {key[:12]}', flush=True)
        metadata = builder(temporary)
        artifacts = [{'name': str(p.relative_to(temporary)), 'sha256': sha256(p)}
                     for p in sorted(temporary.rglob('*')) if p.is_file()]
        manifest = dict(key=key, identity=identity, metadata=metadata, artifacts=artifacts)
        save_json(temporary / 'manifest.json', manifest)
        try:
            os.replace(temporary, folder)
        except OSError:
            if not folder.exists():
                raise
            # Another process may have published the same complete entry.
            return folder, self._validate(folder, identity), True
        return folder, manifest, False
