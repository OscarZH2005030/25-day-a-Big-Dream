from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..cache import FeatureCache
from ..cache_builder import IDENTIFIER_COLUMNS
from ..data import KEY,TARGETS,sha256
from ..fusion_experiments import aligned_prediction
from .rules import CONTROL,NEURAL,SIDES,context_for

PROJECT=Path(__file__).resolve().parents[2]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def inside(path):
    path=Path(path).resolve()
    if not path.is_relative_to(PROJECT):raise ValueError('Source path escapes project: '+str(path))
    return path


def verify_files(items):
    for item in items:
        path=inside(item['path'])
        if sha256(path)!=item['sha256']:raise ValueError('Frozen source changed: '+str(path))


def load_context(folder,manifest):
    parts=[]
    for filename in manifest['metadata']['files']:
        path=(folder/filename).resolve()
        if not path.is_relative_to(folder.resolve()):raise ValueError('Context file escapes cache')
        table=pd.read_parquet(path,columns=IDENTIFIER_COLUMNS+SIDES)
        table=table.rename(columns={c:c[2:] for c in IDENTIFIER_COLUMNS})
        parts.append(table[KEY+SIDES])
    result=pd.concat(parts,ignore_index=True)
    if len(result)!=manifest['metadata']['target_rows']:raise ValueError('Context row count differs')
    result=context_for(result[KEY],result)
    if sorted(result.timestamp.str[:10].unique())!=manifest['metadata']['eval_dates']:raise ValueError('Context dates differ')
    return result


def load_sources(source_run):
    source_run=inside(source_run)
    complete,cfg=read(source_run/'COMPLETE.json'),read(source_run/'run_config.json')
    if complete.get('experiment')!='wave_context' or complete.get('smoke') or cfg.get('smoke'):
        raise ValueError('Source must be completed formal long-context I5 comparison')
    if complete.get('training_rows')!=214000 or complete.get('evaluation_rows')!=550149 or cfg['panel']!='D12_I5_N':
        raise ValueError('Source target scope differs')
    if not all(complete.get(k) for k in ['reference_passed','pairing_passed','integrity_passed','all_regimes_present']):
        raise ValueError('Incomplete source comparison')
    expected_cfg=dict(seed=20261016,sample_per_day=1000,fit_start='2030-06-01',fit_end='2030-12-31',eval_start='2031-01-01',eval_end='2031-01-31')
    if any(cfg.get(k)!=v for k,v in expected_cfg.items()):raise ValueError('Source split/target protocol differs')
    checks=['reference_check.json','sample_check.json','anchor_check.json','receptive_field_check.json','integrity_check.json']
    if not all(read(source_run/f).get('passed') for f in checks):raise ValueError('Source integrity check failed')
    verify_files(read(source_run/'environment.json')['source_files'])
    verify_files(read(source_run/'source_manifest.json'))
    # The saved review ledger anchors prediction content when available; controls are always re-scored too.
    ledger=source_run/'review/input_hashes.json'
    if ledger.exists():verify_files(read(ledger))
    root=inside(cfg['release_root']);baseline=inside(cfg['baseline_run']);bc=read(baseline/'run_config.json')
    if inside(bc['release_root'])!=root:raise ValueError('Baseline release root differs')
    matches=[r for r in read(baseline/'cache_usage.json') if r['panel']==cfg['panel']]
    if len(matches)!=1:raise ValueError('Expected exactly one I5 evaluation cache')
    folder=inside(Path(bc['cache_dir'])/matches[0]['evaluation_key']);manifest=read(folder/'manifest.json')
    FeatureCache(folder.parent)._validate(folder,manifest['identity'])
    identity=manifest['identity']
    if manifest['key']!=matches[0]['evaluation_key'] or identity['kind']!='evaluation' or identity['panel']!=cfg['panel'] or inside(identity['release_root'])!=root:
        raise ValueError('Observable feature cache belongs to another source')
    verify_files(identity['inputs'])
    diagnostics=load_context(folder,manifest);expected=diagnostics[KEY]
    if len(expected)!=550149 or manifest['metadata']['eval_dates']!=cfg['actual_eval_dates'] or cfg['actual_eval_dates']!=pd.date_range('2031-01-01','2031-01-31').strftime('%Y-%m-%d').tolist() or len(cfg['actual_fit_dates'])!=214:
        raise ValueError('Source dates or target keys incomplete')
    predictions={m:aligned_prediction(expected,pd.read_parquet(source_run/(m+'_predictions.parquet'))) for m in [CONTROL,NEURAL]}
    original=inside(Path(cfg['wave_run'])/'Wave_enhanced_hybrid_predictions.parquet')
    replay=aligned_prediction(expected,pd.read_parquet(original))
    if not np.array_equal(predictions[CONTROL][TARGETS].to_numpy(),replay[TARGETS].to_numpy()):
        raise ValueError('Current best does not match frozen original predictions')
    provenance=[source_run/f for f in ['COMPLETE.json','run_config.json','environment.json','source_manifest.json','metrics_by_panel_regime.csv']+checks]
    provenance += [source_run/(m+s) for m in [CONTROL,NEURAL] for s in ['_predictions.parquet','_daily_errors.csv']]
    provenance += [baseline/'run_config.json',baseline/'cache_usage.json',folder/'manifest.json',original]
    if ledger.exists():provenance.append(ledger)
    print('[speed fusion] verified full I5 sources; two saved prediction files reused; no training/inference',flush=True)
    return dict(config=cfg,root=root,expected=expected,diagnostics=diagnostics,predictions=predictions,
        reference=pd.read_csv(source_run/'metrics_by_panel_regime.csv'),
        daily_reference={m:pd.read_csv(source_run/(m+'_daily_errors.csv')) for m in [CONTROL,NEURAL]},
        provenance=provenance,evaluation_cache_key=manifest['key'],prediction_ledger_verified=ledger.exists())


def control_check(source,method,actual,dates,smoke):
    if smoke:
        daily=source['daily_reference'][method];daily=daily.loc[daily.date.isin(dates)]
        rows=[]
        for regime,g in daily.groupby('mask_regime'):
            v=np.sqrt(np.average(g.speed_mse,weights=g.n));q=np.sqrt(np.average(g.flow_lane_mse,weights=g.n))
            rows.append(dict(regime=regime,n=int(g.n.sum()),S_state=.54*max(0,1-v/25)+.46*max(0,1-q/600),rmse_speed_kmh=v,rmse_flow_per_lane=q))
        saved=pd.DataFrame(rows)
    else:saved=source['reference'].loc[source['reference'].method.eq(method)]
    if saved.regime.duplicated().any() or actual.regime.duplicated().any():raise ValueError('Duplicate source regimes')
    t=saved.merge(actual,on='regime',how='outer',suffixes=('_saved','_new'),indicator=True,validate='one_to_one')
    if not t['_merge'].eq('both').all():raise ValueError('Missing control regimes')
    diffs={k:float(np.max(np.abs(t[k+'_saved']-t[k+'_new']))) for k in ['n','S_state','rmse_speed_kmh','rmse_flow_per_lane']}
    if not np.isfinite(list(diffs.values())).all() or diffs['n']!=0 or max(diffs.values())>1e-7:raise ValueError('Saved source score failed to reproduce: '+method)
    return dict(method=method,passed=True,max_absolute_difference=diffs)
