"""Frozen baseline + GRIN, selecting GRIN only for finite bracketing gaps >60min."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..cache import FeatureCache, snapshot
from ..data import KEY, TARGETS, network, save_json, sha256, day_files
from ..fusion_experiments import aligned_prediction, cached_diagnostics
from ..metrics import aggregate, score
from ..pipeline import panel_families, truth_rows
from .hybrid_log import append_hybrid_log
from .runner import comparison, run as run_grin, validate_protocol

PROJECT = Path(__file__).resolve().parents[2]
BASELINE, GRIN_METHOD, HYBRID = 'Fusion_gap60mean', 'GRIN', 'Baseline_GRIN_gap60'
RULE = 'finite speed bracketing gap >60min: GRIN; otherwise frozen baseline; missing side stays baseline'
CONFIG_KEYS = ['panel','release_root','fit_start','fit_end','eval_start','eval_end','max_fit_days','max_eval_days',
    'sample_per_day','seed','hidden','batch_size','window','stride','epochs','steps_per_epoch',
    'patience','learning_rate','dropout','drop_observed','aux_weight']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def inside(path):
    path = Path(path).resolve()
    if not path.is_relative_to(PROJECT):
        raise ValueError(f'Artifact path must be inside project: {path}')
    return path


def gap_labels(minutes):
    minutes = np.asarray(minutes, float)
    if np.isneginf(minutes).any() or (minutes[np.isfinite(minutes)] <= 0).any():
        raise ValueError('Invalid bracketing gap')
    labels = np.asarray(pd.cut(minutes,[0,15,30,60,np.inf],
        labels=['<=15min','15-30min','30-60min','>60min']).astype(str),dtype=object).copy()
    labels[~np.isfinite(minutes)] = 'one_side_or_none'
    return labels


def switch_predictions(expected, baseline, grin, minutes):
    baseline, grin = aligned_prediction(expected, baseline), aligned_prediction(expected, grin)
    minutes = np.asarray(minutes,float)
    if minutes.shape != (len(expected),):
        raise ValueError('Gap length does not match expected targets')
    labels = gap_labels(minutes)
    gate = np.isfinite(minutes) & (minutes > 60)
    result = expected[KEY].copy()
    result[TARGETS] = np.where(gate[:,None],grin[TARGETS].to_numpy(float),baseline[TARGETS].to_numpy(float))
    result['gap'] = labels
    result['selected_model'] = np.where(gate,GRIN_METHOD,BASELINE)
    return result


def check_scores(reference, current, method):
    a=reference.loc[reference.method.eq(method)]
    b=current.loc[current.method.eq(method)]
    if a.empty or a.duplicated(['panel','regime']).any() or set(zip(a.panel,a.regime)) != set(zip(b.panel,b.regime)):
        raise ValueError(f'Missing or duplicate control regimes: {method}')
    pair=a.merge(b,on=['panel','regime'],validate='one_to_one',suffixes=('_old','_new'))
    diffs={c:float(abs(pair[c+'_old']-pair[c+'_new']).max()) for c in ['n','S_state','rmse_speed_kmh','rmse_flow_per_lane']}
    passed=all(np.isfinite(list(diffs.values()))) and diffs['n']==0 and max(v for k,v in diffs.items() if k!='n')<=1e-7
    return dict(method=method,passed=passed,max_absolute_difference=diffs,tolerance=1e-7)


def load_baseline(source, panel):
    source=inside(source); complete=read_json(source/'COMPLETE.json');cfg=read_json(source/'run_config.json')
    if complete.get('experiment') not in ['target_comparison','corridor_validation'] or not complete.get('pairing_passed'):
        raise ValueError('Baseline source must be a completed paired target/corridor experiment')
    if panel not in complete['panels'] or cfg.get('max_fit_days') is not None or cfg.get('max_eval_days') is not None:
        raise ValueError('Baseline source must contain the full formal panel')
    if complete.get('experiment')=='corridor_validation' and not complete.get('integrity_passed'):
        raise ValueError('Baseline integrity checks failed')
    records=read_json(source/'cache_usage.json'); matches=[r for r in records if r['panel']==panel]
    if len(matches)!=1:
        raise ValueError('Baseline cache record missing or ambiguous')
    record=matches[0];cache=inside(cfg['cache_dir'])
    evalfolder=inside(cache/record['evaluation_key']);manifest=read_json(evalfolder/'manifest.json')
    FeatureCache(cache)._validate(evalfolder,manifest['identity'])
    if manifest['key']!=record['evaluation_key'] or manifest['identity']['panel']!=panel:
        raise ValueError('Baseline evaluation cache identity differs')
    for item in manifest['identity']['inputs']:
        if sha256(Path(item['path']))!=item['sha256']:
            raise ValueError(f'Baseline observable input changed: {item["path"]}')
    provenance=[source/'run_config.json',source/'COMPLETE.json',evalfolder/'manifest.json']
    if record.get('prediction_key'):
        prediction_folder=inside(cache/record['prediction_key'])
        pm=read_json(prediction_folder/'manifest.json')
        FeatureCache(cache)._validate(prediction_folder,pm['identity'])
        if pm['key']!=record['prediction_key'] or pm['identity']['evaluation_key']!=manifest['key']:
            raise ValueError('Baseline saved prediction cache identity differs')
        diagnostics=pd.read_parquet(prediction_folder/'diagnostics.parquet')
        provenance += [prediction_folder/'manifest.json',prediction_folder/'diagnostics.parquet']
    else:
        prediction_folder=source/'evaluation'/panel
        diagnostics=cached_diagnostics(evalfolder,manifest)
    if len(diagnostics)!=manifest['metadata']['target_rows'] or not diagnostics.panel.eq(panel).all():
        raise ValueError('Baseline diagnostic target scope differs')
    if not np.array_equal(gap_labels(diagnostics.gap_minutes),diagnostics.gap.to_numpy()):
        raise ValueError('Baseline diagnostic labels disagree with observable gap')
    expected=diagnostics[KEY]
    frozen=source/'evaluation'/panel/f'{BASELINE}_predictions.parquet'
    if frozen.exists():
        baseline=aligned_prediction(expected,pd.read_parquet(frozen));provenance.append(frozen)
    else:
        paths={m:prediction_folder/f'{m}_predictions.parquet' for m in ['B2_multihop','Direct_multihop']}
        r=aligned_prediction(expected,pd.read_parquet(paths['B2_multihop']))
        d=aligned_prediction(expected,pd.read_parquet(paths['Direct_multihop']))
        values=.5*r[TARGETS].to_numpy(float)+.5*d[TARGETS].to_numpy(float)
        use_direct=~np.isfinite(diagnostics.gap_minutes)|(diagnostics.gap_minutes>60)
        baseline=expected.copy();baseline[TARGETS]=np.where(use_direct.to_numpy()[:,None],d[TARGETS],values)
        provenance+=list(paths.values())
    training_table=inside(cache/record['training_key']/'training.parquet')
    return dict(source=source,config=cfg,baseline=baseline,diagnostics=diagnostics,
        training_table=training_table,provenance=provenance,dates=manifest['metadata']['eval_dates'])


def requested_config(args, bundle):
    template=inside(args.template_grin_run)
    if read_json(template/'COMPLETE.json').get('experiment')!='grin_dev':
        raise ValueError('Template must be a completed I5 GRIN experiment')
    cfg=read_json(template/'run_config.json')
    if cfg['panel']!='D12_I5_N' or cfg.get('max_fit_days') is not None or cfg.get('max_eval_days') is not None:
        raise ValueError('Use the formal I5 GRIN experiment as fixed training template')
    cfg.update(panel=args.panel,output=str(args.output/'grin_run'),action='run',checkpoint=None,run=None,
        device=args.device,threads=args.threads,batch_size=args.batch_size,no_log=True,
        baseline_run=str(inside(args.frozen_reference)),baseline_sample_table=str(bundle['training_table']))
    if args.smoke:
        cfg.update(max_fit_days=7,max_eval_days=3,sample_per_day=300,epochs=2,steps_per_epoch=3,
            hidden=16,window=24,stride=24)
    validate_protocol(cfg)
    if Path(cfg['release_root']).resolve()!=Path(bundle['config']['release_root']).resolve():
        raise ValueError('Baseline/GRIN release roots differ')
    for field in ['fit_start','fit_end','eval_start','eval_end','seed']:
        if cfg[field]!=bundle['config'][field]:
            raise ValueError('Baseline/GRIN date or sampling protocol differs')
    return cfg


def require_compatible_grin(source, requested):
    source=inside(source);complete=read_json(source/'COMPLETE.json');cfg=read_json(source/'run_config.json')
    if complete.get('experiment')!='grin_dev' or complete.get('panel')!=requested['panel']:
        raise ValueError('GRIN source is not completed for this panel')
    differing={k:[cfg.get(k),requested.get(k)] for k in CONFIG_KEYS if cfg.get(k)!=requested.get(k)}
    if differing:
        raise ValueError(f'GRIN source configuration differs: {differing}')
    env=read_json(source/'environment.json')
    for relative in ['task1/grin/model.py','task1/grin/data.py']:
        if env['source_sha256'].get(str(Path(relative)))!=sha256(PROJECT/relative):
            raise ValueError('GRIN data/model implementation changed; choose a matching source')
    if env['metric_sha256']!=sha256(PROJECT/'task1/metrics.py'):
        raise ValueError('Scoring implementation differs from GRIN source')
    if not requested.get('max_fit_days'):
        if not read_json(source/'sample_check.json').get('passed'):
            raise ValueError('GRIN original baseline sample check did not pass')
    usage=read_json(source/'cache_usage.json');cache=inside(cfg['cache_dir'])
    folder=inside(cache/usage['evaluation_key']);manifest=read_json(folder/'manifest.json')
    FeatureCache(cache)._validate(folder,manifest['identity'])
    if manifest['key']!=usage['evaluation_key'] or manifest['identity']['panel']!=requested['panel']:
        raise ValueError('GRIN evaluation cache identity differs')
    for item in manifest['identity']['inputs']:
        if sha256(Path(item['path']))!=item['sha256']:
            raise ValueError('GRIN observable evaluation input changed')
    if not (source/'GRIN_predictions.parquet').is_file():
        raise ValueError('GRIN source predictions missing')
    return source


def find_grin(requested):
    candidates=list((PROJECT/'runs').glob('grin_*/COMPLETE.json'))
    candidates+=list((PROJECT/'runs').glob('*/grin_run/COMPLETE.json'))
    for marker in sorted(set(candidates),key=lambda p:p.stat().st_mtime,reverse=True):
        try:
            cfg=read_json(marker.parent/'run_config.json')
            if all(cfg.get(k)==requested.get(k) for k in CONFIG_KEYS):
                return require_compatible_grin(marker.parent,requested)
        except (OSError,ValueError,KeyError):
            continue
    return None


def execute(args):
    output=inside(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new, empty output directory')
    bundle=load_baseline(args.baseline_run,args.panel)
    requested=requested_config(args,bundle)
    diagnostics=bundle['diagnostics']
    if args.smoke:
        dates=sorted(diagnostics.timestamp.str[:10].unique())[:3]
        diagnostics=diagnostics.loc[diagnostics.timestamp.str[:10].isin(dates)].reset_index(drop=True)
    expected=diagnostics[KEY]
    baseline=expected.merge(bundle['baseline'],on=KEY,validate='one_to_one')
    root=Path(requested['release_root']);truth=truth_rows(root,args.panel,expected)
    net=network(root,args.panel);family=panel_families(root)[args.panel]
    base_rows,_=score(baseline,truth,net,family,BASELINE)
    checks=[]
    reference=inside(args.frozen_reference)
    if not args.smoke:
        for key in ['fit_start','fit_end','eval_start','eval_end']:
            if read_json(reference/'run_config.json')[key]!=requested[key]:
                raise ValueError('Frozen reference date protocol differs')
        saved=pd.read_csv(reference/'metrics_by_panel_regime.csv')
        saved=saved.loc[saved.panel.eq(args.panel)]
        checks.append(check_scores(saved,pd.DataFrame(base_rows),BASELINE))
        if not all(c['passed'] for c in checks):
            raise ValueError('Frozen baseline did not reproduce before training')
    output.mkdir(parents=True,exist_ok=True)
    save_json(output/'run_config.json',dict(experiment='grin_hybrid',panel=args.panel,
        baseline_run=str(bundle['source']),frozen_reference=str(reference),template_grin_run=str(args.template_grin_run),
        fit_start=requested['fit_start'],fit_end=requested['fit_end'],eval_start=requested['eval_start'],eval_end=requested['eval_end'],
        smoke=args.smoke,no_log=args.no_log,rule=RULE,threshold_minutes=60,
        training_config=requested,threshold_frozen_before_D7=True))
    if args.grin_run:
        source=require_compatible_grin(args.grin_run,requested)
    else:
        source=find_grin(requested)
    trained=False
    if source is None:
        if args.action=='combine':
            raise ValueError('No compatible saved GRIN predictions; specify --grin-run or use validate action')
        print(f'[{args.panel}] no matching completed GRIN predictions; training only GRIN with fixed I5 settings',flush=True)
        child=dict(requested,output=output/'grin_run')
        run_grin(argparse.Namespace(**child))
        source=require_compatible_grin(output/'grin_run',requested);trained=True
    else:
        print(f'[{args.panel}] reuse completed GRIN predictions: {source.name}; no training or neural inference',flush=True)
    raw_grin=pd.read_parquet(source/'GRIN_predictions.parquet')
    grin=aligned_prediction(expected,raw_grin)
    if 'gap' in raw_grin:
        joined=expected.merge(raw_grin[KEY+['gap']],on=KEY,validate='one_to_one')
        if not np.array_equal(joined.gap.to_numpy(),diagnostics.gap.to_numpy()):
            raise ValueError('GRIN and baseline observable gap definitions differ')
    hybrid=switch_predictions(expected,baseline,grin,diagnostics.gap_minutes)
    rows=[]
    for method,pred in [(BASELINE,baseline),(GRIN_METHOD,grin),(HYBRID,hybrid[KEY+TARGETS])]:
        local,cells=score(pred,truth,net,family,method);rows.extend(local)
        cells=cells.merge(diagnostics[KEY+['gap']],on=KEY,validate='one_to_one')
        for label,grouping in [('daily',['date','mask_regime']),('gap',['gap']),('congestion',['congested'])]:
            cells.groupby(grouping,as_index=False).agg(n=('speed_sq_error','size'),
                speed_mse=('speed_sq_error','mean'),flow_lane_mse=('flow_lane_sq_error','mean')).to_csv(
                    output/f'{method}_{label}_errors.csv',index=False)
    detail=pd.DataFrame(rows)
    checks.append(check_scores(pd.read_csv(source/'metrics_by_panel_regime.csv'),detail,GRIN_METHOD))
    save_json(output/'reference_check.json',dict(passed=all(c['passed'] for c in checks),checks=checks,
        baseline_check='subset smoke; formal reference comparison skipped' if args.smoke else 'all regimes reproduced'))
    if not all(c['passed'] for c in checks):
        raise ValueError('GRIN prediction control failed to reproduce')
    hybrid.to_parquet(output/'combined_predictions.parquet',index=False)
    detail.to_csv(output/'metrics_by_panel_regime.csv',index=False)
    aggregate(detail).to_csv(output/'metrics_summary.csv',index=False)
    table=comparison(detail)
    table['delta_S_state_vs_GRIN']=table.S_state-float(table.loc[table.method.eq(GRIN_METHOD),'S_state'].iloc[0])
    table.to_csv(output/'comparison_hybrid.csv',index=False)
    hybrid.groupby(['mask_regime','gap','selected_model'],as_index=False).size().rename(columns={'size':'n'}).to_csv(output/'gate_counts.csv',index=False)
    reuse=dict(baseline_predictions_reused=True,baseline_models_retrained=False,grin_source=str(source),
        grin_trained_this_run=trained,grin_predictions_reused=not trained,
        grin_cache_usage=read_json(source/'cache_usage.json'),gated_rows=int(hybrid.selected_model.eq(GRIN_METHOD).sum()),
        target_rows=len(hybrid))
    save_json(output/'reuse.json',reuse)
    full_truth=day_files(root,args.panel,'train',False)
    save_json(output/'source_manifest.json',snapshot(bundle['provenance']+[
        source/'GRIN_predictions.parquet',source/'run_config.json',source/'environment.json',source/'COMPLETE.json',
        reference/'metrics_by_panel_regime.csv',Path(__file__),Path(__file__).with_name('hybrid_log.py')]+
        [full_truth[d] for d in sorted(expected.timestamp.str[:10].unique())]))
    save_json(output/'COMPLETE.json',dict(experiment='grin_hybrid',panel=args.panel,rule=RULE,
        reference_passed=True,methods=[BASELINE,GRIN_METHOD,HYBRID],evaluated_rows=len(expected),
        gated_rows=reuse['gated_rows'],grin_source=str(source)))
    try:
        status=append_hybrid_log(output,PROJECT/'实验日志.txt')
    except (OSError,ValueError) as error:
        save_json(output/'log_update.json',dict(status='failed',error=str(error)))
        raise RuntimeError('Combination completed; log failed. Retry the log action without training.') from error
    print(table.to_string(index=False),flush=True)
    print(f'Log status: {status["status"]}. Hybrid complete: {output}',flush=True)


def main():
    parser=argparse.ArgumentParser(description='Fixed baseline/GRIN gap switching, cached I5 combination and D7 validation')
    parser.add_argument('action',choices=['combine','validate','log'])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--panel',choices=['D12_I5_N','D7_I405_N'],default='D7_I405_N')
    parser.add_argument('--baseline-run',type=Path)
    parser.add_argument('--frozen-reference',type=Path,default=PROJECT/'runs/cross_corridor_review_20261016_210254')
    parser.add_argument('--template-grin-run',type=Path,default=PROJECT/'runs/grin_dev_20261017_015925')
    parser.add_argument('--grin-run',type=Path)
    parser.add_argument('--device',choices=['cuda','cpu','auto'],default='cuda')
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--no-log',action='store_true')
    args=parser.parse_args()
    if Path.cwd().resolve()!=PROJECT:
        parser.error(f'Run from {PROJECT}')
    if args.action=='log':
        if not args.run: parser.error('log requires --run')
        print(append_hybrid_log(inside(args.run),PROJECT/'实验日志.txt'));return
    if not args.output or args.batch_size<=0 or args.threads<=0:
        parser.error('Require --output and positive batch size/threads')
    if args.baseline_run is None:
        args.baseline_run=PROJECT/'runs'/('target_dev_20261016_183242' if args.panel=='D12_I5_N' else 'corridor_dev_20261016_192125')
    args.output=inside(args.output)
    execute(args)


if __name__=='__main__':
    main()
