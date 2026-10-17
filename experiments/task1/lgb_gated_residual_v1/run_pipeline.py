"""Explicit foreground entry: preflight, OOF trees, gated residual, final report."""
import argparse
import ctypes
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import pandas as pd
from common import (HERE, REFERENCE, FINAL_LGB, METHOD, RunLock, console_log,
                    make_folds, panel_metadata, read_json, save_json, sha)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['preflight', 'crossfit', 'train', 'evaluate', 'all'], default='preflight')
    p.add_argument('--reference-run', type=Path, default=REFERENCE)
    p.add_argument('--final-lgb', type=Path, default=FINAL_LGB)
    p.add_argument('--output', type=Path, default=HERE/'runs/all_corridors_time80_20')
    p.add_argument('--panels', nargs='+', help='Default: all ten in original order')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--folds', type=int, default=4)
    p.add_argument('--inner-days', type=int, default=14)
    p.add_argument('--max-trees', type=int, default=2000)
    p.add_argument('--tree-patience', type=int, default=100)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--steps-per-epoch', type=int, default=1024)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--window', type=int, default=24)
    p.add_argument('--patch-nodes', type=int, default=64)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--dropout', type=float, default=.1)
    p.add_argument('--learning-rate', type=float, default=.0003)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--shrink', type=float, default=.001)
    p.add_argument('--eval-batch', type=int, default=2)
    p.add_argument('--seed', type=int, default=2026)
    args = p.parse_args()
    if min(args.threads, args.folds, args.inner_days, args.max_trees, args.tree_patience, args.epochs,
           args.steps_per_epoch, args.batch_size, args.window, args.patch_nodes, args.width, args.heads,
           args.layers, args.patience, args.eval_batch) < 1:
        p.error('All sizes must be positive')
    if args.folds < 2 or args.width < 4 or args.width%args.heads or not 2 <= args.window <= 288:
        p.error('Invalid folds, width/heads or temporal window')
    if not 0 <= args.dropout < 1 or args.learning_rate <= 0 or args.shrink < 0:
        p.error('Invalid optimization settings')
    for name in ['reference_run', 'final_lgb', 'output']: setattr(args, name, getattr(args, name).resolve())
    return args


def configuration(args, plan, panels, provenance):
    settings = {k:v for k, v in vars(args).items() if k not in ['stage', 'resume', 'panels']}
    settings = {k:str(v) if isinstance(v, Path) else v for k, v in settings.items()}
    return dict(version='lgb-gated-residual-v1', settings=settings, panels=panels,
        history_dates=plan['history_dates'], training_dates=plan['training_dates'],
        internal_validation_dates=plan['internal_validation_dates'], holdout_dates=plan['holdout_dates'],
        reference_plan_sha256=sha(args.reference_run/'plan.json'),
        source_sha256={p.name:sha(p) for p in sorted(HERE.glob('*.py'))}, inputs=provenance,
        crossfit='4 blocked outer folds by default; inner dates only choose trees; refit non-held supervised dates',
        graph='unchanged original symmetrized 1/2-hop; global isolated nodes use LightGBM exactly',
        input_dimensions=106, physics=dict(fd_weight=0., conservation_weight=0.),
        normalization='51 means/scales from original 168 supervised days; fixed units for four new features',
        baseline='OOF during residual learning; existing full168 LightGBM during internal/final prediction',
        target_scope='sample_submission_state keys only; no extra masking',
        limitations=['Internal dates already used for selecting existing LightGBM.',
            'OOF and full-data baseline predictions may differ in distribution.',
            'Random space/time blocks; one epoch is not a full target traversal.',
            'Final holdout previously inspected; local Task1 only, no official submission.'])


def summarize(args, panels):
    complete = [p for p in panels if (args.output/p/'COMPLETE.json').exists()]
    if not complete: return
    metrics, comparisons = [], []
    for panel in complete:
        frame = pd.read_csv(args.output/panel/'metrics.csv'); frame.insert(0, 'panel', panel); metrics.append(frame)
        new = frame[(frame.method == METHOD)&(frame.regime == 'macro_mean')].S_state.iloc[0]
        row = dict(panel=panel, gated_residual=float(new))
        for path, name in [(args.reference_run, 'Transformer'), (args.final_lgb, 'LightGBM')]:
            old = pd.read_csv(path/panel/'metrics.csv')
            row[name] = float(old[old.regime == 'macro_mean'].S_state.iloc[0])
        row['delta_vs_Transformer'] = row['gated_residual']-row['Transformer']
        row['delta_vs_LightGBM'] = row['gated_residual']-row['LightGBM']
        comparisons.append(row)
    all_metrics = pd.concat(metrics, ignore_index=True)
    all_metrics.to_csv(args.output/'all_metrics.csv', index=False)
    pd.DataFrame(comparisons).to_csv(args.output/'comparison.csv', index=False)
    all_metrics[all_metrics.regime.isin(['R1', 'R2', 'R3'])].groupby(['panel', 'method'], as_index=False)[
        ['rmse_speed_kmh', 'rmse_flow_per_lane', 'S_state']].mean().to_csv(args.output/'channel_comparison.csv', index=False)
    save_json(args.output/'summary.json', dict(completed_corridors=len(complete), selected_corridors=len(panels),
        mean_corridor_S_state=float(pd.DataFrame(comparisons).gated_residual.mean()),
        note='Descriptive local Task1 equal-corridor mean; not accuracy or official overall score.'))


def main():
    args = arguments(); plan = read_json(args.reference_run/'plan.json')
    panels = args.panels or plan['panels']
    if len(set(panels)) != len(panels) or any(p not in plan['panels'] for p in panels): raise ValueError('Invalid selected corridors')
    if len(plan['training_dates']) != 168 or len(plan['internal_validation_dates']) != 22 or len(plan['holdout_dates']) != 55:
        raise ValueError('Unexpected frozen chronological split')
    make_folds(plan['training_dates'], args.folds, args.inner_days)
    print('Read-only preflight: source signatures, data metadata, frozen LightGBM and cached dates...', flush=True)
    provenance = {panel:panel_metadata(args.reference_run, args.final_lgb, plan, panel) for panel in panels}
    config = configuration(args, plan, panels, provenance)
    if args.stage == 'preflight':
        save_json(HERE/'preflight_validation.json', dict(passed=True, training_started=False, inference_started=False,
            torch_imported=False, cuda_initialized=False, selected_panels=panels,
            fold_sizes=[len(f['predict_dates']) for f in make_folds(plan['training_dates'], args.folds, args.inner_days)], config=config))
        print('PREFLIGHT PASSED. No training, tree fitting, inference or CUDA initialization.', flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with RunLock(args.output/'queue.lock'):
        manifest = args.output/'plan.json'
        if manifest.exists():
            if not args.resume or read_json(manifest) != config: raise ValueError('Existing config/source/input differs; use unchanged --resume or a new output directory')
        else:
            save_json(manifest, config)
            snapshot = args.output/'sources'; snapshot.mkdir(exist_ok=True)
            for file in HERE.glob('*.py'): shutil.copyfile(file, snapshot/file.name)
            with (args.output/'environment_pip_freeze.txt').open('w', encoding='utf-8') as stream:
                subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream, check=True)
        status = dict(pid=os.getpid(), state='running', completed=[], current=None, stage=args.stage, epoch=None, step=None)
        def update(**values):
            status.update(values); status['updated_utc'] = datetime.now(timezone.utc).isoformat()
            save_json(args.output/'status.json', status)
        if os.name == 'nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            for panel in panels:
                folder = args.output/panel; folder.mkdir(exist_ok=True)
                if (folder/'COMPLETE.json').exists():
                    old = read_json(folder/'COMPLETE.json')
                    for name, key in [('best.pt', 'best_sha256'), ('holdout_predictions.parquet', 'prediction_sha256'), ('metrics.csv', 'metrics_sha256')]:
                        if sha(folder/name) != old[key]: raise ValueError('Completed artifact changed: '+panel+'/'+name)
                    status['completed'].append(panel); print('SKIP COMPLETE', panel, flush=True); continue
                update(current=panel, epoch=None, step=None)
                with RunLock(folder/'worker.lock'), console_log(folder/'console.log'):
                    if args.stage in ['crossfit', 'all']:
                        update(stage='crossfit')
                        from crossfit import prepare
                        prepare(args, plan, panel)
                    if args.stage in ['train', 'all']:
                        update(stage='train_initializing')
                        print('Loading PyTorch; then initializing', args.device, '(training explicitly requested)', flush=True)
                        from neural import train
                        train(args, plan, panel, update)
                    if args.stage in ['evaluate', 'all']:
                        update(stage='final_evaluation')
                        from neural import final_evaluate
                        final_evaluate(args, plan, panel)
                if (folder/'COMPLETE.json').exists(): status['completed'].append(panel)
                summarize(args, panels); update(current=None)
            update(state='complete' if len(status['completed']) == len(panels) else 'stage_complete', current=None)
        except BaseException as error:
            update(state='failed', error=str(error)); traceback.print_exc(); raise
        finally:
            if os.name == 'nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__': main()
