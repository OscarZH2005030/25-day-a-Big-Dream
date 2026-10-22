"""Recover and run the fixed ten-panel baseline without changing cache identities.

Run through run_task1_all_baseline.ps1. Modes: check (read-only), recover
(publish completed temporary caches only), run (recover and run all panels).
This entry point does not edit task1/cache.py or monkey-patch os.replace.
"""
from __future__ import annotations

import argparse
import base64
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from task1 import corridor_validation as c
from task1.cache import CacheIntegrityError, FeatureCache, cache_key, feature_implementation

PROJECT = Path(__file__).resolve().parent
RELEASE = PROJECT / '2026-ieee-big-data-traffic-flow-bench' / 'kaggle_public'
KINDS = {'train', 'evaluation', 'direct_labels', 'multihop_model_pair', 'multihop_predictions'}
TEMP_NAME = re.compile(r'\.tmp_[0-9a-f]{32}\Z')
KEY_NAME = re.compile(r'[0-9a-f]{64}\Z')


class CachePublishError(RuntimeError):
    pass


def powershell_rename(source: Path, destination: Path):
    """Paths are child environment values, never interpolated into shell code."""
    if os.name != 'nt':
        raise OSError('PowerShell fallback is only used on Windows')
    code = """$ErrorActionPreference = 'Stop'
Rename-Item -LiteralPath $env:TFB_CACHE_SOURCE -NewName $env:TFB_CACHE_DEST_NAME -Force -ErrorAction Stop
"""
    encoded = base64.b64encode(code.encode('utf-16-le')).decode('ascii')
    env = dict(os.environ, TFB_CACHE_SOURCE=str(source), TFB_CACHE_DEST_NAME=destination.name)
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-EncodedCommand', encoded],
        env=env, capture_output=True, timeout=20, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        message = result.stderr.decode(errors='replace')[-1200:]
        raise OSError(f'PowerShell rename failed ({result.returncode}): {message}')


class RecoverableCache(FeatureCache):
    def __init__(self, root, *, retries=4, native_replace=None, rename_fallback=None, sleep=None):
        super().__init__(Path(root).resolve())
        self.retries = retries
        self.native_replace = native_replace or os.replace
        self.rename_fallback = rename_fallback or powershell_rename
        self.sleep = sleep or time.sleep

    def _paths(self, temporary, identity):
        temporary = Path(temporary).resolve()
        key = cache_key(identity)
        if temporary.parent != self.root or not TEMP_NAME.fullmatch(temporary.name) or not KEY_NAME.fullmatch(key):
            raise ValueError('Refusing a cache publication outside the cache root')
        return temporary, self.root / key

    def publish(self, temporary, identity):
        temporary, destination = self._paths(temporary, identity)
        if destination.exists():
            return destination, self._validate(destination, identity), True
        self._validate(temporary, identity)
        last_error = None
        for attempt in range(self.retries):
            if destination.exists():
                return destination, self._validate(destination, identity), True
            try:
                self.native_replace(temporary, destination)
                return destination, self._validate(destination, identity), False
            except PermissionError as error:
                last_error = error
            except FileNotFoundError:
                if destination.exists():
                    return destination, self._validate(destination, identity), True
                raise
            if destination.exists():
                return destination, self._validate(destination, identity), True
            try:
                self.rename_fallback(temporary, destination)
                return destination, self._validate(destination, identity), False
            except (OSError, subprocess.TimeoutExpired) as error:
                last_error = error
                if destination.exists():
                    return destination, self._validate(destination, identity), True
            if attempt + 1 < self.retries:
                print(f'[cache publish] retry {attempt + 1}/{self.retries}: {identity["kind"]} {identity["panel"]}', flush=True)
                self.sleep(min(2 ** attempt, 4))
        raise CachePublishError(f'Cache is complete and preserved at {temporary}. '
            'Close programs holding this folder, pause its sync if applicable, and rerun this entry point; '
            'it will recover this cache before building again.') from last_error

    def recover_completed(self, *, publish, panels):
        records = []
        if not self.root.exists():
            return records
        for temporary in sorted(self.root.glob('.tmp_*')):
            manifest_path = temporary / 'manifest.json'
            if not temporary.is_dir() or not TEMP_NAME.fullmatch(temporary.name) or not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                identity = manifest['identity']
                if identity.get('kind') not in KINDS or identity.get('panel') not in panels:
                    continue
                self._paths(temporary, identity)
                self._validate(temporary, identity)
            except (OSError, ValueError, KeyError, TypeError, CacheIntegrityError) as error:
                records.append(dict(temporary=str(temporary), status='invalid_kept', error=str(error)))
                print(f'[recovery] invalid temporary cache kept: {temporary.name}: {error}', flush=True)
                continue
            record = dict(temporary=str(temporary), key=manifest['key'], kind=identity['kind'], panel=identity['panel'])
            if publish:
                destination, _, existed = self.publish(temporary, identity)
                record.update(status='destination_valid_temporary_kept' if existed else 'recovered', destination=str(destination))
            else:
                record['status'] = 'verified_ready'
            records.append(record)
            print(f'[recovery] {record["status"]}: {identity["kind"]} {identity["panel"]} {manifest["key"][:12]}', flush=True)
        return records

    def get_or_build(self, identity, builder):
        key = cache_key(identity)
        destination = self.root / key
        if destination.exists():
            print(f'[cache] HIT {identity["kind"]} {identity["panel"]} {key[:12]}', flush=True)
            return destination, self._validate(destination, identity), True
        self.root.mkdir(parents=True, exist_ok=True)
        # Recover an exact matching completed cache instead of recomputing it.
        for temporary in sorted(self.root.glob('.tmp_*')):
            path = temporary / 'manifest.json'
            if not TEMP_NAME.fullmatch(temporary.name) or not path.is_file():
                continue
            try:
                manifest = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            if manifest.get('identity') == identity:
                folder, result, _ = self.publish(temporary, identity)
                print(f'[cache] RECOVERED {identity["kind"]} {identity["panel"]} {key[:12]}', flush=True)
                return folder, result, True
        temporary = self.root / ('.tmp_' + uuid.uuid4().hex)
        temporary.mkdir()
        print(f'[cache] MISS {identity["kind"]} {identity["panel"]} {key[:12]}', flush=True)
        metadata = builder(temporary)
        artifacts = [dict(name=str(p.relative_to(temporary)), sha256=c.sha256(p))
                     for p in sorted(temporary.rglob('*')) if p.is_file()]
        c.save_json(temporary / 'manifest.json', dict(key=key, identity=identity, metadata=metadata, artifacts=artifacts))
        return self.publish(temporary, identity)


def comparison(detail):
    table = c.fusion_comparison(detail)
    score = float(table.loc[table.method.eq('Fusion_mean50'), 'S_state'].iloc[0])
    table['delta_S_state_vs_mean50'] = table.S_state - score
    return table


def append_log(output, log_path):
    complete = json.loads((output / 'COMPLETE.json').read_text(encoding='utf-8'))
    overall = pd.read_csv(output / 'comparison_overall.csv')
    panels = pd.read_csv(output / 'comparison_by_panel.csv')
    if complete['experiment'] != 'all_task1_baseline' or len(set(complete['panels'])) != 10:
        raise ValueError('Only a completed ten-panel baseline can be logged')
    if set(overall.method) != set(c.METHODS) or not overall.panel_count.eq(10).all() or not overall.family_count.eq(5).all() or not overall.all_regimes_present.all():
        raise ValueError('Incomplete overall baseline results')
    if len(panels) != 40 or set(panels.panel) != set(complete['panels']) or panels.duplicated(['panel', 'method']).any():
        raise ValueError('Incomplete per-panel baseline results')
    for _, group in panels.groupby('panel'):
        if set(group.method) != set(c.METHODS) or not group.all_regimes_present.all():
            raise ValueError('Incomplete methods or regimes for a panel')
    cfg = json.loads((output / 'run_config.json').read_text(encoding='utf-8'))
    marker = f'实验编号：{output.name}；结果目录：{output.resolve()}'
    existing = log_path.read_text(encoding='utf-8-sig') if log_path.exists() else ''
    if marker in existing:
        status = dict(status='already_logged', log_path=str(log_path))
    else:
        def line(row):
            return (f'{row.method}：S_state={row.S_state:.6f}；速度 RMSE={row.rmse_speed_kmh:.6f}；'
                f'每车道流量 RMSE={row.rmse_flow_per_lane:.6f}；相对残差 ΔS={row.delta_S_state_vs_residual:+.6f}；'
                f'相对直接 ΔS={row.delta_S_state_vs_direct:+.6f}；相对等权 ΔS={row.delta_S_state_vs_mean50:+.6f}。')
        lines = ['', '【Task1 全部道路 baseline｜已完成】', marker,
            '范围：5 条走廊、10 个方向；每个方向独立训练多跳残差及直接模型。',
            f'训练：{cfg["fit_start"]} 至 {cfg["fit_end"]}；评测：{cfg["eval_start"]} 至 {cfg["eval_end"]}；二月留出。',
            f'每天抽样 {cfg["sample_per_day"]}；轮数={cfg["rounds"]}；seed={cfg["seed"]}；线程={cfg["threads"]}。',
            'Fusion_mean50：各 50%；Fusion_gap60mean：可见速度观测间隔 >60 分钟或缺一侧用直接，其余等权，两个通道共用规则。',
            '按 regime→方向→走廊家族等权聚合；分数越大、RMSE 越小越好。', '全部道路汇总：']
        lines += [line(row) for row in overall.itertuples(index=False)]
        for panel in complete['panels']:
            lines.append(f'道路方向：{panel}')
            lines += [line(row) for row in panels.loc[panels.panel.eq(panel)].itertuples(index=False)]
        lines += ['完整结果见 comparison_overall.csv、comparison_by_panel.csv、metrics_by_panel_regime.csv。', '']
        with log_path.open('a', encoding='utf-8') as stream:
            stream.write('\n'.join(lines) + '\n')
        status = dict(status='appended', log_path=str(log_path))
    c.save_json(output / 'log_update.json', status)
    return status


def run_baseline(cache, output, recovery, threads):
    families = c.panel_families(RELEASE)
    panels = list(families)
    if len(panels) != 10 or len(set(families.values())) != 5:
        raise ValueError('Expected the ten-panel / five-family competition scope')
    output.mkdir(parents=True, exist_ok=False)
    cfg = dict(command='all-baseline-resume', panels=panels, release_root=str(RELEASE),
        output=str(output), cache_dir=str(cache.root), fit_start='2030-06-01', fit_end='2030-12-31',
        eval_start='2031-01-01', eval_end='2031-01-31', max_fit_days=None, max_eval_days=None,
        sample_per_day=1000, rounds=350, threads=threads, max_gap_minutes=60, seed=20261016)
    c.save_json(output / 'run_config.json', dict(cfg, fusion_rules=c.VALIDATION_RULES))
    c.save_json(output / 'cache_recovery.json', recovery)
    c.save_json(output / 'environment.json', dict(python=platform.python_version(),
        packages={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'pyarrow', 'lightgbm']},
        source_sha256={p.name: c.sha256(p) for p in (PROJECT / 'task1').glob('*.py')},
        runner_sha256=c.sha256(Path(__file__)), feature_implementation=feature_implementation()))
    rows, usage, tables, pairing = [], [], [], []
    required = set(pd.date_range(cfg['fit_start'], cfg['eval_end']).strftime('%Y-%m-%d'))
    for i, panel in enumerate(panels, 1):
        print(f'\n[{i}/10] {panel}', flush=True)
        for masked in [True, False]:
            if not required <= set(c.day_files(RELEASE, panel, 'train', masked)):
                raise ValueError(f'Missing required daily data: {panel}')
        td, tm, th = c.training_cache(cache, RELEASE, panel, cfg)
        ed, em, eh = c.evaluation_cache(cache, RELEASE, panel, cfg, td, tm)
        ld, lm, lh = c.direct_label_cache(cache, RELEASE, panel, td, tm)
        md, mm, mh = c.paired_model_cache(cache, panel, td, tm, ld, lm, cfg)
        pairing.append(dict(panel=panel, **c.verify_pair(md, tm, lm, cfg)))
        pred_dir, pm, ph = c.paired_prediction_cache(cache, panel, md, mm, ed, em, cfg)
        diag = pd.read_parquet(pred_dir / 'diagnostics.parquet')
        dates = sorted(diag.timestamp.str[:10].unique())
        if len(diag) != em['metadata']['target_rows'] or dates != em['metadata']['eval_dates'] or len(dates) != 31 or not diag.panel.eq(panel).all():
            raise ValueError(f'Cached prediction scope differs: {panel}')
        predictions = c.fuse_predictions(diag[c.KEY],
            pd.read_parquet(pred_dir / f'{c.RESIDUAL}_predictions.parquet'),
            pd.read_parquet(pred_dir / f'{c.DIRECT}_predictions.parquet'),
            diag.gap_minutes, rules=c.VALIDATION_RULES)
        truth, net = c.truth_rows(RELEASE, panel, diag[c.KEY]), c.network(RELEASE, panel)
        destination = output / 'evaluation' / panel
        destination.mkdir(parents=True)
        local = []
        for method, pred in predictions.items():
            if method in c.VALIDATION_RULES:
                pred.to_parquet(destination / f'{method}_predictions.parquet', index=False)
            detail, cells = c.score(pred, truth, net, families[panel], method)
            local.extend(detail)
            cells = cells.merge(diag, on=c.KEY, validate='one_to_one')
            for name, grouping in [('daily', ['date', 'mask_regime']), ('gap', ['gap']), ('congestion', ['congested'])]:
                cells.groupby(grouping, as_index=False).agg(n=('speed_sq_error', 'size'),
                    speed_mse=('speed_sq_error', 'mean'), flow_lane_mse=('flow_lane_sq_error', 'mean')).to_csv(
                        destination / f'{method}_{name}_errors.csv', index=False)
        rows.extend(local)
        table = comparison(pd.DataFrame(local))
        table.insert(0, 'panel', panel)
        tables.append(table)
        usage.append(dict(panel=panel, training_key=tm['key'], training_hit=th,
            evaluation_key=em['key'], evaluation_hit=eh, direct_labels_key=lm['key'], direct_labels_hit=lh,
            model_pair_key=mm['key'], model_pair_hit=mh, prediction_key=pm['key'], prediction_hit=ph,
            model_folder=str(md), prediction_folder=str(pred_dir)))
        detail = pd.DataFrame(rows)
        detail.to_csv(output / 'metrics_by_panel_regime.csv', index=False)
        c.aggregate(detail).to_csv(output / 'metrics_summary.csv', index=False)
        comparison(detail).to_csv(output / 'comparison_overall.csv', index=False)
        pd.concat(tables, ignore_index=True).to_csv(output / 'comparison_by_panel.csv', index=False)
        c.save_json(output / 'cache_usage.json', usage)
        c.save_json(output / 'pairing_check.json', dict(passed=True, records=pairing))
        print(table[['method', 'S_state', 'rmse_speed_kmh', 'rmse_flow_per_lane']].to_string(index=False), flush=True)
        del predictions, truth, cells, diag
    overall = comparison(pd.DataFrame(rows))
    if not overall.panel_count.eq(10).all() or not overall.family_count.eq(5).all() or not overall.all_regimes_present.all():
        raise ValueError('Incomplete all-road baseline results')
    c.save_json(output / 'COMPLETE.json', dict(experiment='all_task1_baseline', panels=panels,
        methods=c.METHODS, fusion_rules=c.VALIDATION_RULES, pairing_passed=True))
    try:
        status = append_log(output, PROJECT / '实验日志.txt')
    except (OSError, ValueError) as error:
        c.save_json(output / 'log_update.json', dict(status='failed', error=str(error)))
        raise RuntimeError('Results complete; retry logging with --log-run followed by this output directory') from error
    print(f'\nALL COMPLETE: {output}\nLog status: {status["status"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['check', 'recover', 'run'], default='run')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--log-run', type=Path, help='Retry logging a completed run without recovery or training')
    args = parser.parse_args()
    if Path.cwd().resolve() != PROJECT:
        parser.error(f'Run from project directory: {PROJECT}')
    if args.threads <= 0:
        parser.error('--threads must be positive')
    if args.log_run:
        run = args.log_run.resolve()
        if not run.is_relative_to(PROJECT / 'runs'):
            parser.error('--log-run must be in the project runs directory')
        print(append_log(run, PROJECT / '实验日志.txt'))
        return
    stamp = datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d_%H%M%S_%f')
    output = args.output.resolve() if args.output else PROJECT / 'runs' / ('all_baseline_resume_' + stamp)
    if not output.is_relative_to(PROJECT / 'runs') or output.exists():
        parser.error('--output must be a new directory inside project/runs')
    cache = RecoverableCache(PROJECT / 'cache' / 'task1')
    records = cache.recover_completed(publish=args.mode != 'check', panels=set(c.panel_families(RELEASE)))
    print(json.dumps(records, ensure_ascii=False, indent=2), flush=True)
    if args.mode == 'run':
        run_baseline(cache, output, records, args.threads)


if __name__ == '__main__':
    main()
