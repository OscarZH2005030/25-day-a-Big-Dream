from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import numpy as np
import pandas as pd
from ..data import save_json,sha256
from .rules import CONTROL,NEURAL,PRIMARY,METHODS,WEIGHTS


def append_log(run,log_path):
    run,log_path=Path(run).resolve(),Path(log_path).resolve()
    read=lambda name:json.loads((run/name).read_text(encoding='utf-8'))
    complete,cfg=read('COMPLETE.json'),read('run_config.json')
    if complete.get('experiment')!='speed_fusion':raise ValueError('Completed speed-fusion experiment required')
    status='skipped_smoke' if complete.get('smoke') or cfg.get('smoke') else 'skipped_by_option' if cfg.get('no_log') else None
    if status:
        result=dict(status=status);save_json(run/'log_update.json',result);return result
    if complete.get('evaluation_rows')!=550149 or len(cfg['actual_eval_dates'])!=31 or cfg.get('panel')!='D12_I5_N':
        raise ValueError('Formal target/date scope incomplete')
    if cfg['actual_eval_dates']!=pd.date_range('2031-01-01','2031-01-31').strftime('%Y-%m-%d').tolist() or cfg.get('fit_start')!='2030-06-01' or cfg.get('fit_end')!='2030-12-31' or cfg.get('eval_start')!='2031-01-01' or cfg.get('eval_end')!='2031-01-31':
        raise ValueError('Fixed training/development split differs')
    if set(complete['methods'])!=set(METHODS) or len(complete['methods'])!=4 or cfg.get('weights')!=WEIGHTS or cfg.get('primary_method')!=PRIMARY:
        raise ValueError('Fixed methods/primary/weights differ')
    if not all(complete.get(k) for k in ['reference_passed','pairing_passed','integrity_passed','all_regimes_present']):
        raise ValueError('Formal comparison incomplete')
    if complete.get('artifact_manifest_sha256')!=sha256(run/'artifact_manifest.json'):
        raise ValueError('Artifact manifest changed')
    artifacts=read('artifact_manifest.json')['artifacts']
    names=set()
    for item in artifacts:
        path=Path(item['path']).resolve()
        if not path.is_relative_to(run) or not path.is_file() or sha256(path)!=item['sha256']:
            raise ValueError('Formal artifact changed/missing: '+str(path))
        names.add(str(path.relative_to(run)))
    required={'run_config.json','comparison_methods.csv','metrics_by_panel_regime.csv','reference_check.json',
        'sample_check.json','integrity_check.json','preservation_checks.json','prediction_reuse.json','fusion_gate.parquet','gate_counts.csv'}
    required.update(m+'_predictions.parquet' for m in METHODS)
    if not required.issubset(names):raise ValueError('Required artifacts are not hash-protected')
    for name in ['reference_check.json','sample_check.json','integrity_check.json','preservation_checks.json']:
        if not read(name).get('passed'):raise ValueError('Formal integrity failed: '+name)
    integrity=read('preservation_checks.json')
    if set(integrity['methods'])!=set(METHODS):raise ValueError('Missing preservation checks')
    eligible=complete['gated_rows']
    if not 0<eligible<550149:raise ValueError('Invalid gate coverage')
    for method,alpha in [(CONTROL,0),*WEIGHTS.items()]:
        item=integrity['methods'][method]
        if not item.get('passed') or item['alpha']!=alpha or item['target_rows']!=550149 or item['eligible_rows']!=eligible or item['preserved_speed_rows']!=550149-eligible:
            raise ValueError('Preservation scope differs')
        if not all(item.get(k) for k in ['all_flow_values_equal','all_outside_speed_values_equal','all_inside_speed_values_match_formula']):
            raise ValueError('Flow or non-gated speed preservation failed')
    table=pd.read_csv(run/'comparison_methods.csv');detail=pd.read_csv(run/'metrics_by_panel_regime.csv')
    if set(table.method)!=set(METHODS) or table.method.duplicated().any() or not table.all_regimes_present.all():raise ValueError('Missing method')
    columns=['S_state','rmse_speed_kmh','rmse_flow_per_lane']
    if not np.isfinite(table[columns].to_numpy()).all():raise ValueError('Nonfinite scores')
    base=table.loc[table.method.eq(CONTROL)].iloc[0]
    if abs(base.S_state-read('reference_check.json')['control_S_state'][CONTROL])>1e-12:
        raise ValueError('Current-best score differs from checked source')
    for method in METHODS:
        rows=detail.loc[detail.method.eq(method)];row=table.loc[table.method.eq(method)].iloc[0]
        if len(rows)!=3 or set(rows.regime)!={'R1','R2','R3'} or rows.n.sum()!=550149:raise ValueError('Regime target coverage differs')
        if any(abs(row[k]-rows[k].mean())>1e-9 for k in columns):raise ValueError('Macro score differs from regimes')
        if abs(row.rmse_flow_per_lane-base.rmse_flow_per_lane)>1e-12:raise ValueError('Flow score changed')
        if abs(row.delta_S_state_vs_current_best-(row.S_state-base.S_state))>1e-12:raise ValueError('Delta score differs')
        source_flow=detail.loc[detail.method.eq(CONTROL)].set_index('regime').rmse_flow_per_lane
        if not np.array_equal(rows.set_index('regime').rmse_flow_per_lane.reindex(source_flow.index).to_numpy(),source_flow.to_numpy()):
            raise ValueError('Regime flow scores changed')
    reuse=read('prediction_reuse.json')
    if reuse.get('prediction_files_reused')!=2 or reuse.get('models_retrained') or reuse.get('neural_inference_run') or reuse.get('feature_cache_rebuilt'):
        raise ValueError('Prediction reuse provenance differs')
    marker=f'实验编号：{run.name}；结果目录：{run}'
    previous=log_path.read_text(encoding='utf-8-sig') if log_path.exists() else ''
    if marker in previous:
        result=dict(status='already_logged',log_path=str(log_path));save_json(run/'log_update.json',result);return result
    date=datetime.now(ZoneInfo('America/Los_Angeles')).date().isoformat()
    lines=['',f'【I5 15–60分钟缺口速度通道固定权重融合｜{date}｜已完成】',marker,
        '直接复用已保存的当前最佳交通波增强LightGBM＋GRIN、长时间范围无波网络预测；未训练、未做神经推理、未重建特征。',
        '来源模型训练2030年6–12月；本轮仍评估2031完整一月550149目标；二月观测/真值未读。',
        '仅两侧观测时间均为有限正数且15<gap<=60分钟时融合速度；15分钟不融合、60分钟融合。流量及其余速度逐值保留原组合。',
        f'神经速度权重固定10%/25%/50%；事前主候选为25%（{PRIMARY}）。不根据一月自动替换当前最佳；另外两组只检验权重敏感性。',
        f'融合门控={eligible}/550149行；其余{550149-eligible}行速度保持原值，全部550149行流量保持原值；逐值检查通过。',
        '两个来源控制评分复现；评分沿用task1.metrics的regime→方向→家族等权；不按行数加权。']
    for row in table.itertuples(index=False):
        lines.append(f'{row.method}：S_state={row.S_state:.6f}；速度RMSE={row.rmse_speed_kmh:.6f}；每车道流量RMSE={row.rmse_flow_per_lane:.6f}；对当前最佳ΔS={row.delta_S_state_vs_current_best:+.6f}。')
    for regime in ['R1','R2','R3']:
        g=detail.loc[detail.regime.eq(regime)].set_index('method')
        lines.append(regime+'：'+'；'.join(f'{m}={g.loc[m,"S_state"]:.6f}' for m in METHODS)+'。')
    lines+=['该比较属于已反复使用的一月开发集，后续应固定规则在D7验证；真实拥堵标签未用于门控。',
        '分项见comparison_methods.csv、各方法分日/缺口/拥堵误差、regime_gap_errors.csv和preservation_checks.json。','']
    with log_path.open('a',encoding='utf-8',newline='') as stream:stream.write('\n'.join(lines)+'\n')
    result=dict(status='appended',log_path=str(log_path));save_json(run/'log_update.json',result);return result
