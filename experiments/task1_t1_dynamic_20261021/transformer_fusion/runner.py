from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import time
import numpy as np
import pandas as pd

from task1.cache import snapshot
from task1.data import KEY, TARGETS
from task1.transformer_priority.common import PILOT, read, save_json, sha256, upstream, versions
from .core import NEURAL, details, evaluate, predict_bundle, select, summarize
from .source import Source, verify

EXPERIMENT = 'transformer_prediction_fusion_v1'
CONTROLS = {name: dict(beta=beta) for name, beta in NEURAL.items()}
CONTROLS['LightGBM'] = dict(beta=0.0, speed=1.0, flow=1.0)


def verify_complete(out):
    complete = read(out/'COMPLETE.json')
    if complete['experiment'] != EXPERIMENT or not complete['integrity_passed']:
        raise ValueError('Fusion experiment incomplete')
    if sha256(out/'artifact_manifest.json') != complete['artifact_manifest_sha256']:
        raise ValueError('Result manifest changed')
    artifacts = read(out/'artifact_manifest.json')
    verify(artifacts)
    required = {'run_config.json', 'selection.json', 'source_manifest.json', 'cache_usage.json',
                'integrity_check.json', 'comparison_methods.csv', 'metrics_by_panel_regime.csv',
                'internal_neural_metrics.csv', 'internal_grid_metrics.csv', 'internal_stability.csv',
                'preservation_checks.json', 'source_score_replay.csv'}
    required.update(str(Path(panel)/(method+'_predictions.parquet')) for panel in PILOT
                    for method in list(CONTROLS)+['Selected'])
    protected = {str(Path(x['path']).relative_to(out)) for x in artifacts}
    if not required.issubset(protected):
        raise ValueError('Missing protected result artifacts')
    verify(read(out/'source_manifest.json'))
    cfg = read(out/'run_config.json')
    if cfg['panels'] != PILOT or cfg['training_started'] or cfg['selection_protocol'] != protocol():
        raise ValueError('Result scope/recipe changed')
    selection = read(out/'selection.json')
    verify(selection['internal_artifacts'])
    table = pd.read_csv(out/'comparison_methods.csv')
    detail = pd.read_csv(out/'metrics_by_panel_regime.csv')
    rebuilt = summarize(detail).set_index('method')
    if set(table.method) != set(CONTROLS)|{'Selected'} or table.method.duplicated().any():
        raise ValueError('Missing evaluated controls/candidate')
    for row in table.itertuples(index=False):
        for field in ['S_state', 'rmse_speed_kmh', 'rmse_flow_per_lane']:
            if abs(getattr(row, field)-rebuilt.loc[row.method, field]) > 1e-10:
                raise ValueError('Macro result reconstruction differs')
        if abs(row.delta_vs_T0-(row.S_state-rebuilt.loc['T0', 'S_state'])) > 1e-12:
            raise ValueError('Reference delta differs')
    return complete, cfg, table


def protocol():
    return dict(neural=['T0','T01_equal','T1'], weights=[0.0,0.1,0.25,0.5],
                shared_across_panels=True, internal_only=True, holdout_candidates=1,
                isolated_fallback='exact shared LightGBM both channels')


def log_results(out, project):
    out, project = Path(out).resolve(), Path(project).resolve()
    complete, cfg, table = verify_complete(out)
    if cfg['smoke']:
        result = dict(status='skipped_smoke')
        save_json(out/'log_update.json', result)
        return result
    selection = read(out/'selection.json')
    if cfg['split']['internal_validation_dates'] != selection['selection_dates']:
        raise ValueError('Formal selection lacks all22 internal dates')
    integrity = read(out/'integrity_check.json')
    if not all(integrity[k] for k in ['passed','source_scores_replayed','all_isolated_exact','selection_frozen_before_holdout']):
        raise ValueError('Formal invariant failed')
    detail = pd.read_csv(out/'metrics_by_panel_regime.csv')
    expected = pd.read_csv(project/'task1/transformer_priority/reference_routing_counts.csv').set_index('panel')
    checks = read(out/'preservation_checks.json')
    for panel in PILOT:
        for method in list(CONTROLS)+['Selected']:
            rows = detail.loc[detail.panel.eq(panel)&detail.method.eq(method)]
            if set(rows.regime) != {'R1','R2','R3'} or len(rows) != 3 or rows.n.sum() != expected.loc[panel,'n']:
                raise ValueError('Formal target/regime coverage differs')
            item = checks[panel+'/'+method]
            if not item['isolated_exact'] or item['isolated_rows'] != expected.loc[panel,'fallback_n']:
                raise ValueError('Formal isolation coverage differs')
    log = project/'实验日志.txt'
    marker = f'实验编号：{out.name}；结果目录：{out}'
    # Separate lock also protects two different outputs logging concurrently.
    with upstream().lock(project/'.transformer_fusion_log.lock'):
        previous = log.read_text(encoding='utf-8-sig') if log.exists() else ''
        if marker in previous:
            result = dict(status='already_logged', log_path=str(log))
        else:
            date = datetime.now(ZoneInfo('America/Los_Angeles')).date()
            lines = ['', f'【Transformer两路已有预测分通道融合｜{date}｜已完成】', marker,
                     '来源：'+str(cfg['source_run']),
                     'D7_I405_N/S；历史28、监督168、内部22、留出55天；未训练任何模型。',
                     '内部先选T0/T1/等权神经配方，再选速度/流量各0/0.1/0.25/0.5的共享LightGBM权重；孤立节点始终精确回退LightGBM。',
                     '冻结主候选：'+json.dumps(selection['recipe'], ensure_ascii=False),
                     '内部前后时段仅作诊断；留出只评测固定对照和冻结主候选，没有用留出选择权重。',
                     '55天已经反复查看，属探索性本地验证；本轮两路成绩不能宣称超过十路0.936910。']
            for row in table.itertuples(index=False):
                lines.append(f'{row.method}：S={row.S_state:.9f}；对T0 ΔS={row.delta_vs_T0:+.9f}；速度宏RMSE={row.rmse_speed_kmh:.6f}；每车道流量宏RMSE={row.rmse_flow_per_lane:.6f}。')
            lines += ['来源/目标/原评分复现/孤立回退检查通过；逐日、regime、缺口、拥堵、配对日期重采样及缓存记录见结果目录。', '']
            with log.open('a', encoding='utf-8', newline='') as stream:
                stream.write('\n'.join(lines)+'\n')
            result = dict(status='appended', log_path=str(log))
    save_json(out/'log_update.json', result)
    return result


def run(args):
    started = time.perf_counter()
    out = args.output
    if out == args.source_run or out.is_relative_to(args.source_run) or args.source_run.is_relative_to(out):
        raise ValueError('New output must be separate from protected source run')
    if args.cache_dir.is_relative_to(args.source_run):
        raise ValueError('New cache must not modify source run')
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(experiment=EXPERIMENT, source_run=str(args.source_run), project=str(args.project),
               cache_dir=str(args.cache_dir), smoke=args.smoke, device=args.device, action=args.action,
               panels=PILOT, selection_protocol=protocol(), training_started=False,
               source_anchor=snapshot([args.source_run/'COMPLETE.json', args.source_run/'artifact_manifest.json']),
               implementation=snapshot(list(Path(__file__).parent.glob('*.py'))), versions=versions())
    with upstream().lock(out/'execution.lock'):
        if (out/'run_config.json').exists():
            if not args.resume:
                raise ValueError('Existing output; use unchanged --resume or choose a new output')
            old = read(out/'run_config.json')
            old.pop('split', None)
            if old != cfg:
                raise ValueError('Existing output config/source changed')
            if (out/'COMPLETE.json').exists():
                print(log_results(out, args.project), flush=True)
                return
        def status(stage, **values):
            save_json(out/'status.json', dict(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **values))
        source = None
        try:
            status('validating_source')
            source = Source(args.source_run, args.project, args.cache_dir, out, args.smoke, args.device)
            cfg['split'] = source.plan
            save_json(out/'run_config.json', cfg)
            preflight = {}
            for panel in PILOT:
                status('preparing_read_only_features', panel=panel)
                print('[prepare]', panel, flush=True)
                data = source.prepare(panel)
                roles = source.roles[panel]
                # Verify saved predictions and fallback without loading holdout truth.
                source.saved_holdout(panel, data, roles['holdout'])
                preflight[panel] = dict(dates=roles, rows={r:len(data.expected(d)) for r,d in roles.items()},
                                        isolated_nodes=int(data.graph['isolated'].sum()))
            save_json(out/'preflight.json', dict(passed=True, panels=preflight, holdout_labels_read=False,
                                               training_started=False))
            if args.action == 'prepare':
                source.finish()
                status('prepared', passed=True)
                print('Full preflight passed; no selection, no holdout labels, no training.', flush=True)
                return
            internal = {}
            for panel in PILOT:
                status('internal_prediction_and_labels', panel=panel)
                print('[internal]', panel, flush=True)
                internal[panel] = source.bundle(panel, 'internal')
            selection, neural, grid, stability = select(internal, source.internal_dates())
            neural.to_csv(out/'internal_neural_metrics.csv', index=False)
            grid.to_csv(out/'internal_grid_metrics.csv', index=False)
            summarize(neural).to_csv(out/'internal_neural_comparison.csv', index=False)
            summarize(grid).to_csv(out/'internal_grid_comparison.csv', index=False)
            stability.to_csv(out/'internal_stability.csv', index=False)
            selection['internal_artifacts'] = snapshot([out/name for name in ['internal_neural_metrics.csv',
                'internal_grid_metrics.csv','internal_neural_comparison.csv','internal_grid_comparison.csv','internal_stability.csv']])
            selection['frozen_utc'] = datetime.now(timezone.utc).isoformat()
            save_json(out/'selection.json', selection)
            frozen_hash = sha256(out/'selection.json')
            del internal
            status('selection_frozen', recipe=selection['recipe'])
            print('[frozen]', selection['recipe'], flush=True)
            all_details, daily, preservation, replay = [], [], {}, []
            source_metric = pd.read_csv(source.source_metrics)
            recipes = dict(CONTROLS, Selected=selection['recipe'])
            for panel in PILOT:
                status('holdout_evaluation', panel=panel)
                print('[holdout]', panel, flush=True)
                if sha256(out/'selection.json') != frozen_hash:
                    raise ValueError('Frozen selection changed')
                bundle = source.bundle(panel, 'holdout', out/'selection.json')
                destination = out/panel
                destination.mkdir(exist_ok=True)
                for method, recipe in recipes.items():
                    detail = details({panel:bundle}, recipe, method)
                    all_details.append(detail)
                    values, groups = evaluate(bundle, recipe, method)
                    prediction = bundle['context'][KEY].copy()
                    prediction[TARGETS] = values
                    prediction.to_parquet(destination/(method+'_predictions.parquet'), index=False)
                    iso = bundle['context'].isolated.to_numpy(bool)
                    preservation[panel+'/'+method] = dict(rows=len(values), isolated_rows=int(iso.sum()),
                        isolated_exact=bool(np.array_equal(values[iso], bundle['LightGBM'][iso])),
                        speed_tree_weight=recipe.get('speed',0), flow_tree_weight=recipe.get('flow',0))
                    for name, frame in groups.items():
                        frame.to_csv(destination/(method+'_'+name+'_errors.csv'), index=False)
                    daily.append(groups['daily'])
                    if method in ['T0','T1']:
                        original = 'T0_original' if method == 'T0' else 'T1_balanced'
                        for row in detail.itertuples(index=False):
                            saved = source_metric.loc[source_metric.panel.eq(panel)&source_metric.method.eq(original)&source_metric.regime.eq(row.regime)]
                            if len(saved) != 1:
                                raise ValueError('Source score row missing')
                            error = max(abs(getattr(row,f)-float(saved.iloc[0][f])) for f in ['n','S_state','rmse_speed_kmh','rmse_flow_per_lane'])
                            if not args.smoke and error > 1e-9:
                                raise ValueError('Source raw truth rescoring differs')
                            replay.append(dict(panel=panel,method=method,regime=row.regime,max_difference=error,
                                               checked_full_scope=not args.smoke))
                del bundle
            detail = pd.concat(all_details, ignore_index=True)
            detail.to_csv(out/'metrics_by_panel_regime.csv', index=False)
            comparison = summarize(detail)
            comparison['delta_vs_T0'] = comparison.S_state-float(comparison.loc[comparison.method.eq('T0'),'S_state'].iloc[0])
            comparison.to_csv(out/'comparison_methods.csv', index=False)
            daily = pd.concat(daily, ignore_index=True)
            daily.to_csv(out/'daily_errors.csv', index=False)
            # Resample joint panel/regime calendar strata to preserve paired dates
            # and avoid missing entire regimes in small smoke subsets.
            bootstrap_rows = paired_bootstrap(daily, ['T1','T01_equal','Selected'])
            pd.DataFrame(bootstrap_rows).to_csv(out/'paired_calendar_bootstrap.csv', index=False)
            pd.DataFrame(replay).to_csv(out/'source_score_replay.csv', index=False)
            save_json(out/'preservation_checks.json', preservation)
            if not all(x['isolated_exact'] for x in preservation.values()):
                raise ValueError('Isolated fallback preservation failed')
            if sha256(out/'selection.json') != frozen_hash:
                raise ValueError('Selection changed during holdout evaluation')
            status('final_integrity_validation')
            source.finish()
            verify(cfg['implementation'])
            save_json(out/'integrity_check.json', dict(passed=True, training_started=False,
                source_scores_replayed=not args.smoke, all_isolated_exact=True,
                selection_frozen_before_holdout=True, full_dates=not args.smoke,
                limitation='Repeatedly inspected local holdout; two directions only, no ten-direction breakthrough claim'))
            artifacts = snapshot([p for p in out.rglob('*') if p.is_file() and p.name not in
                ['execution.lock','status.json','COMPLETE.json','artifact_manifest.json','log_update.json']])
            save_json(out/'artifact_manifest.json', artifacts)
            save_json(out/'COMPLETE.json', dict(experiment=EXPERIMENT, smoke=args.smoke, integrity_passed=True,
                artifact_manifest_sha256=sha256(out/'artifact_manifest.json'), seconds=time.perf_counter()-started))
            result = log_results(out, args.project)
            status('completed', log_status=result['status'], seconds=time.perf_counter()-started)
            print(comparison.to_string(index=False), flush=True)
            print('Log status:', result['status'], '\nFusion complete:', out, flush=True)
        except Exception as error:
            status('failed', error=repr(error))
            raise


def paired_bootstrap(daily, candidates, replicates=2000):
    """Pair dates across panels; stratify by their joint regime signatures."""
    rng = np.random.default_rng(20261020)
    dates = sorted(daily.date.unique())
    # A given calendar date may have different regimes in the two directions.
    mapping = daily[['date','panel','mask_regime']].drop_duplicates()
    if mapping.duplicated(['date','panel']).any():
        raise ValueError('Multiple regimes for one direction/day')
    joint = mapping.pivot(index='date',columns='panel',values='mask_regime').reindex(dates).fillna('absent')
    strata = {}
    for index, signature in enumerate(joint.itertuples(index=False,name=None)):
        strata.setdefault(signature, []).append(index)
    counts = np.zeros((replicates, len(dates)), dtype=np.int64)
    for indices in strata.values():
        counts[:,indices] = rng.multinomial(len(indices), np.full(len(indices),1/len(indices)), size=replicates)
    method_scores = {}
    panels = sorted(daily.panel.unique())
    families = {p:p.rsplit('_',1)[0] for p in panels}
    for method in ['T0']+candidates:
        score_by_family = {}
        for panel in panels:
            score_panel = np.zeros(replicates)
            for regime in ['R1','R2','R3']:
                frame = daily.loc[daily.method.eq(method)&daily.panel.eq(panel)&daily.mask_regime.eq(regime)].set_index('date').reindex(dates)
                n = frame.n.fillna(0).to_numpy()
                total = counts@n
                if (total == 0).any():
                    raise ValueError('Bootstrap sample has no targets')
                speed = np.sqrt((counts@(frame.speed_mse.fillna(0).to_numpy()*n))/total)
                flow = np.sqrt((counts@(frame.flow_lane_mse.fillna(0).to_numpy()*n))/total)
                score_panel += (.54*np.maximum(0,1-speed/25)+.46*np.maximum(0,1-flow/600))/3
            score_by_family.setdefault(families[panel], []).append(score_panel)
        method_scores[method] = np.mean([np.mean(values,axis=0) for values in score_by_family.values()],axis=0)
    rows = []
    for candidate in candidates:
        lo, hi = np.quantile(method_scores[candidate]-method_scores['T0'], [.025,.975])
        rows.append(dict(reference='T0',candidate=candidate,replicates=replicates,ci95_low=float(lo),ci95_high=float(hi),
            sampling='Calendar dates paired across panels, stratified by joint panel/regime signatures',
            limitation='Descriptive fixed-model local holdout uncertainty; no retraining/selection correction; not independent-test evidence'))
    return rows
