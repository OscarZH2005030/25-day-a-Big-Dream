"""Verify the isolated, pinned Task 2 code environment before any GPU smoke."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def verify_sources():
    manifest = json.loads((ROOT/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        p = (ROOT/entry['path']).resolve()
        if not p.is_relative_to(ROOT) or hashlib.sha256(p.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('Released Task 2 source changed: '+entry['path'])
    return len(manifest['files'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=parser.parse_args()
    if sys.version_info[:2] != (3,14):
        raise RuntimeError('Use a separate Python 3.14 .venv-task2 environment; reference is 3.14.8')
    for name in ['requirements-lock.txt','requirements-torch-deps.txt']:
        for line in (ROOT/name).read_text(encoding='utf-8').splitlines():
            if not line.strip() or line.startswith('#'):continue
            package,expected=line.split('==',1)
            if version(package)!=expected:
                raise RuntimeError(f'{package}: expected {expected}, got {version(package)}')
    if version('torch')!='2.10.0+cu128':
        raise RuntimeError('Expected PyTorch 2.10.0+cu128; run setup_task2.ps1')
    import torch
    torch.set_num_threads(4)
    count=verify_sources()
    from task2q.protocol import split_manifest
    expected=split_manifest()
    if json.loads((ROOT/'FROZEN_SPLIT.json').read_text(encoding='utf-8'))!=expected:
        raise ValueError('Frozen split file and protocol do not agree')
    result=dict(passed=True,python=sys.version.split()[0],torch=torch.__version__,source_files=count,
                protocol_sha256=expected['protocol_sha256'],formal_training_available=False)
    if args.device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; inspect NVIDIA driver before smoke')
        properties=torch.cuda.get_device_properties(0)
        x=torch.ones((8,8),device='cuda')
        assert float((x@x).sum())==512
        result.update(gpu=properties.name,vram_gib=round(properties.total_memory/2**30,2))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
