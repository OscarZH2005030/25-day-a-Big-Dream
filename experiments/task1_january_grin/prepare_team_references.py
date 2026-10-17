"""Create local score references from a newly completed paired baseline run.

Only copies verified summary metadata; never fits a model or copies predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

PROJECT = Path(__file__).resolve().parent
PANELS = {'D12_I5_N', 'D7_I405_N'}
DATES = dict(fit_start='2030-06-01', fit_end='2030-12-31',
             eval_start='2031-01-01', eval_end='2031-01-31')


def prepare(source: Path, output: Path):
    source, output = source.resolve(), output.resolve()
    if not source.is_relative_to(PROJECT) or not output.is_relative_to(PROJECT):
        raise ValueError('Source and output must stay inside this experiment directory')
    complete = json.loads((source / 'COMPLETE.json').read_text(encoding='utf-8'))
    cfg = json.loads((source / 'run_config.json').read_text(encoding='utf-8'))
    if complete.get('experiment') != 'corridor_validation' or not all(
            complete.get(k) for k in ['pairing_passed', 'integrity_passed']):
        raise ValueError('Use a completed, paired and integrity-checked corridor baseline')
    if not PANELS.issubset(set(complete['panels'])) or any(cfg[k] != v for k, v in DATES.items()):
        raise ValueError('Both I5 and D7 with the fixed June-December / January dates are required')
    if any(cfg.get(k) is not None for k in ['max_fit_days', 'max_eval_days']):
        raise ValueError('Smoke results are not formal references')
    if cfg['seed'] != 20261016 or cfg['sample_per_day'] != 1000:
        raise ValueError('Sampling protocol differs')
    metric = source / 'metrics_by_panel_regime.csv'
    with metric.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    for panel in PANELS:
        control = [r for r in rows if r['panel'] == panel and r['method'] == 'Fusion_gap60mean']
        if len(control) != 3 or {r['regime'] for r in control} != {'R1', 'R2', 'R3'}:
            raise ValueError('Missing or duplicate frozen control regimes')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a new empty reference directory')
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metric, output / metric.name)
    config = dict(DATES, sources=[str(source)], retrained=False,
                  note='Locally rebuilt score reference from the newly completed baseline; not an old prediction archive')
    marker = dict(experiment='cross_corridor_review', panels=sorted(PANELS), retrained=False,
                  source=str(source), source_metrics_sha256=hashlib.sha256(metric.read_bytes()).hexdigest())
    for name, value in [('run_config.json', config), ('COMPLETE.json', marker)]:
        (output / name).write_text(json.dumps(value, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=PROJECT / 'runs/team_reference')
    args = parser.parse_args()
    if Path.cwd().resolve() != PROJECT:
        parser.error('Run from this experiment directory')
    print(json.dumps(prepare(args.baseline_run, args.output), ensure_ascii=True))


if __name__ == '__main__':
    main()
