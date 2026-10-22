from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from ..data import save_json


def append_hybrid_log(run,log_path):
    run,log_path=Path(run).resolve(),Path(log_path).resolve()
    def read(name):return json.loads((run/name).read_text(encoding='utf-8'))
    complete=read('COMPLETE.json');cfg=read('run_config.json')
    if complete.get('experiment')!='grin_hybrid' or not complete.get('reference_passed'):
        raise ValueError('Only completed, checked hybrid runs can be logged')
    if not read('reference_check.json').get('passed'):
        raise ValueError('Hybrid source reference checks failed')
    if cfg.get('smoke'):
        status=dict(status='skipped_smoke')
    elif cfg.get('no_log'):
        status=dict(status='skipped_by_option')
    else:status=None
    if status:
        save_json(run/'log_update.json',status);return status
    table=pd.read_csv(run/'comparison_hybrid.csv')
    if set(table.method)!=set(complete['methods']) or table.method.duplicated().any() or not table.all_regimes_present.all():
        raise ValueError('Incomplete formal hybrid comparison')
    detail=pd.read_csv(run/'metrics_by_panel_regime.csv')
    for _,group in detail.groupby('method'):
        if int(group.n.sum())!=complete['evaluated_rows'] or set(group.regime)!={'R1','R2','R3'}:
            raise ValueError('Hybrid target counts or regimes differ')
    marker=f'实验编号：{run.name}；结果目录：{run}'
    existing=log_path.read_text(encoding='utf-8-sig') if log_path.exists() else ''
    if marker in existing:
        status=dict(status='already_logged',log_path=str(log_path))
        save_json(run/'log_update.json',status);return status
    reuse=read('reuse.json');train=cfg['training_config']
    source=Path(reuse['grin_source'])
    selection=json.loads((source/'selection.json').read_text(encoding='utf-8'))
    date=datetime.now(ZoneInfo('America/Los_Angeles')).date().isoformat()
    lines=['',f'【baseline＋GRIN 固定缺口切换｜{date}｜已完成】',marker,
        '方向：'+complete['panel'],'baseline 来源：'+cfg['baseline_run'],'GRIN 来源：'+str(source),
        f'训练：{cfg["fit_start"]} 至 {cfg["fit_end"]}；开发：{cfg["eval_start"]} 至 {cfg["eval_end"]}；二月未使用。',
        '规则固定：双侧可见速度观测间隔 >60 分钟用 GRIN；其余用冻结 baseline；恰好 60 分钟、缺任一侧均用 baseline。',
        '冻结 baseline 自身为多跳残差/直接的 60 分钟融合；两个通道共用选择条件，不使用真值选模型。',
        f'目标数={complete["evaluated_rows"]}；切换 GRIN 的目标数={complete["gated_rows"]}。',
        f'复用 baseline 预测={reuse["baseline_predictions_reused"]}；重训 baseline={reuse["baseline_models_retrained"]}；'
        f'本次训练 GRIN={reuse["grin_trained_this_run"]}；复用已完成 GRIN 预测={reuse["grin_predictions_reused"]}。',
        f'GRIN：hidden={train["hidden"]}；window={train["window"]}；stride={train["stride"]}；batch={train["batch_size"]}；'
        f'最多 {train["epochs"]} epochs、每轮最多 {train["steps_per_epoch"]} 步；lr={train["learning_rate"]}；最佳 epoch={selection["best_epoch"]}。',
        'GRIN 按单模型一月 S_state 选择 checkpoint；组合权重/阈值没有在新走廊调参。D7 在自身训练期独立训练，不直接迁移 I5 权重。',
        '评分复用 task1.metrics；原预测逐 regime 复现检查通过；流量按每车道评测。']
    for row in table.itertuples(index=False):
        lines.append(f'{row.method}：S_state={row.S_state:.6f}；速度 RMSE={row.rmse_speed_kmh:.6f}；每车道流量 RMSE={row.rmse_flow_per_lane:.6f}；'
            f'相对冻结 baseline ΔS={row.delta_S_state_vs_frozen_baseline:+.6f}；相对 GRIN ΔS={row.delta_S_state_vs_GRIN:+.6f}。')
    for regime in ['R1','R2','R3']:
        local=detail.loc[detail.regime.eq(regime)].set_index('method')
        lines.append(f'{regime}：baseline={local.loc["Fusion_gap60mean","S_state"]:.6f}；'
            f'GRIN={local.loc["GRIN","S_state"]:.6f}；组合={local.loc["Baseline_GRIN_gap60","S_state"]:.6f}。')
    lines+=['详细结果见 comparison_hybrid.csv、metrics_by_panel_regime.csv、gate_counts.csv 及三个方法的分日/缺口/拥堵误差表。','']
    log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('a',encoding='utf-8',newline='') as stream:stream.write('\n'.join(lines)+'\n')
    status=dict(status='appended',log_path=str(log_path))
    save_json(run/'log_update.json',status);return status
