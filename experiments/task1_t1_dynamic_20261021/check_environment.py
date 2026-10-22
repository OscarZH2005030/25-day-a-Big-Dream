import argparse
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify_sources():
    from task1.data import sha256
    manifest = json.loads((ROOT / 'source_export_manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path = (ROOT / item['path']).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != item['sha256']:
            raise ValueError('Frozen training source changed: ' + item['path'])
    for name in ['task1', 'task1.transformer_priority.common', 'transformer_training',
                 'transformer_cross_corridor', 'transformer_remaining', 'transformer_fusion']:
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(ROOT):
            raise ValueError('Imported another checkout: ' + name)
    from task1.transformer_priority.common import upstream
    upstream()  # Includes the vendored original source SHA checks.
    return len(manifest['files'])


def main(argv=None):
    parser = argparse.ArgumentParser(description='Check portable sources, locked dependencies, references and GPU')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--release-root', type=Path)
    args = parser.parse_args(argv)
    if sys.version_info[:2] != (3, 14):
        raise RuntimeError('Python 3.14 is required; the reference used 3.14.8')
    for name in ['requirements-lock.txt', 'requirements-torch-deps.txt']:
        for line in (ROOT / name).read_text(encoding='utf-8-sig').splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            package, expected = line.split('==', 1)
            actual = importlib.metadata.version(package)
            if actual != expected:
                raise RuntimeError(f'{package}: expected {expected}, found {actual}')
    if importlib.metadata.version('torch') != '2.10.0+cu128':
        raise RuntimeError('Install the pinned PyTorch 2.10.0+cu128 environment')
    from task1.transformer_priority.common import versions
    from team_portable import verify_reference_archive
    sources = verify_reference_archive(ROOT)
    for source in sources:
        cfg = json.loads((Path(source['path']) / 'run_config.json').read_text(encoding='utf-8'))
        if cfg['versions'] != versions():
            raise RuntimeError('Runtime versions differ from the fixed four-direction reference')
    count = verify_sources()
    import torch
    device_name = 'CPU'
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; check the NVIDIA driver and pinned CUDA wheel')
        device_name = torch.cuda.get_device_name(0)
        x = torch.ones((8, 8), device='cuda')
        assert float((x @ x).sum()) == 512.0
    if args.release_root:
        from transformer_remaining.common import PANELS
        release = args.release_root.resolve()
        for panel in PANELS:
            for path in [release / 'task1' / panel / 'train/sample_submission_state.csv',
                         release / 'corridors' / panel / 'network/lwr_mainline_topology.csv']:
                if not path.is_file():
                    raise FileNotFoundError('Missing official input: ' + str(path))
    print(json.dumps(dict(passed=True, python=sys.version.split()[0], versions=versions(),
        device=device_name, exact_training_source_files=count, fixed_reference_directions=4), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
