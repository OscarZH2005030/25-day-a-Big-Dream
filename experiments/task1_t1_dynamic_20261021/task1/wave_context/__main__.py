from __future__ import annotations
import argparse
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description='I5: frozen best, long-context neural, same-anchor wave-feature neural')
    p.add_argument('action', choices=['run','prepare','log'])
    p.add_argument('--output', type=Path); p.add_argument('--run', type=Path)
    p.add_argument('--release-root', type=Path, default=PROJECT/'2026-ieee-big-data-traffic-flow-bench/kaggle_public')
    p.add_argument('--cache-dir', type=Path, default=PROJECT/'cache/wave_context')
    p.add_argument('--baseline-run', type=Path, default=PROJECT/'runs/target_dev_20261016_183242')
    p.add_argument('--grin-run', type=Path, default=PROJECT/'runs/grin_dev_20261017_015925')
    p.add_argument('--hybrid-run', type=Path, default=PROJECT/'runs/grin_hybrid_i5_20261017_142725')
    p.add_argument('--wave-run', type=Path, default=PROJECT/'runs/traffic_wave_dev_20261017_174736_955')
    p.add_argument('--input-run', type=Path, default=PROJECT/'runs/wave_neural_dev_20261017_194339_329')
    p.add_argument('--device', choices=['cuda','cpu','auto'], default='cuda')
    p.add_argument('--batch-size', type=int, default=2); p.add_argument('--threads', type=int, default=4)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--no-log', action='store_true')
    a = p.parse_args()
    if Path.cwd().resolve()!=PROJECT:
        p.error(f'Run from {PROJECT}')
    for name in ['output','run','cache_dir','release_root','baseline_run','grin_run','hybrid_run','wave_run','input_run']:
        value = getattr(a, name)
        if value is not None:
            value = value.resolve()
            if not value.is_relative_to(PROJECT):
                p.error('--'+name.replace('_','-')+' must stay inside project')
            setattr(a, name, value)
    if a.action=='log':
        if a.run is None:
            p.error('log requires --run')
        from .log import append_log
        print(append_log(a.run, PROJECT/'实验日志.txt')); return
    if a.output is None:
        p.error('--output is required')
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Choose a new empty output directory')
    from ..data import save_json
    from .runner import run
    try:
        run(a)
    except Exception as error:
        if not (a.output/'COMPLETE.json').exists():
            save_json(a.output/'FAILED.json', dict(reason=type(error).__name__, message=str(error),
                suggestion='Inspect failure, then use a new output folder. For CUDA OOM explicitly select batch-size 1; no silent fallback.'))
        raise


if __name__=='__main__':
    main()
