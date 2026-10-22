from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..data import save_json


def append_log(run, log_path):
    run,log_path=Path(run).resolve(),Path(log_path).resolve()
    def read(name):return json.loads((run/name).read_text(encoding='utf-8'))
    complete,cfg=read('COMPLETE.json'),read('run_config.json')
    if complete.get('experiment')!='imputeformer_dev':raise ValueError('Only completed ImputeFormer runs can be logged')
    if complete.get('smoke') or cfg.get('max_fit_days') is not None or cfg.get('max_eval_days') is not None:
        status=dict(status='skipped_smoke')
    elif cfg.get('no_log'):status=dict(status='skipped_by_option')
    else:status=None
    if status:
        save_json(run/'log_update.json',status);return status
    if not all(complete.get(k) for k in ['pairing_passed','reference_passed']):
        raise ValueError('Formal sample pairing or saved control reproduction failed')
    if not read('sample_check.json').get('passed') or not read('reference_check.json').get('passed'):
        raise ValueError('Missing verified sample/reference checks')
    table=pd.read_csv(run/'comparison_models.csv')
    required={'ImputeFormer','Fusion_gap60mean','GRIN','Baseline_GRIN_gap60','Baseline_ImputeFormer_gap60'}
    if (set(table.method)!=required or set(complete['methods'])!=required or table.method.duplicated().any()
        or not table.all_regimes_present.all()
        or not np.isfinite(table[['S_state','rmse_speed_kmh','rmse_flow_per_lane']].to_numpy()).all()):
        raise ValueError('Missing or invalid formal model comparison')
    detail=pd.read_csv(run/'metrics_by_panel_regime.csv')
    for method in required:
        local=detail.loc[detail.method.eq(method)]
        if len(local)!=3 or set(local.regime)!={'R1','R2','R3'} or not local.panel.eq('D12_I5_N').all():
            raise ValueError('Incomplete per-regime comparison')
    marker=f'实验编号：{run.name}；结果目录：{run}'
    previous=log_path.read_text(encoding='utf-8-sig') if log_path.exists() else ''
    if marker in previous:
        status=dict(status='already_logged',log_path=str(log_path))
        save_json(run/'log_update.json',status);return status
    selection,cache=read('selection.json'),read('cache_usage.json')
    date=datetime.now(ZoneInfo('America/Los_Angeles')).date().isoformat()
    lines=['',f'【ImputeFormer 核心结构适配｜{date}｜已完成】',marker,'方向：'+cfg['panel'],
        f'训练：{cfg["fit_start"]} 至 {cfg["fit_end"]}；开发：{cfg["eval_start"]} 至 {cfg["eval_end"]}；二月未使用。',
        '结构：投影时间注意力＋嵌入空间注意力＋可学习时空嵌入；作者结构的轻量适配，不是原论文训练配方的逐项复现。',
        f'hidden={cfg["hidden"]}；embedding={cfg["embedding_dim"]}；layers={cfg["layers"]}；heads={cfg["heads"]}；rank={cfg["rank"]}；'
        f'window={cfg["window"]}；stride={cfg["stride"]}；batch={cfg["batch_size"]}；lr={cfg["learning_rate"]}。',
        f'最多 {cfg["epochs"]} epochs、每轮最多 {cfg["steps_per_epoch"]} 步；最佳 epoch={selection["best_epoch"]}；'
        f'原缺失训练目标={selection["training_rows"]}；每天 {cfg["sample_per_day"]}；seed={cfg["seed"]}；与 baseline 的有序目标键核对通过。',
        f'额外随机点遮挡概率={cfg["drop_observed"]}；评分尺度加权 MSE；无方向辅助损失、无频域正则；只用训练期可见观测标准化。',
        '直接复用 GRIN 日张量、mask、训练标签和滑窗推理；输入不使用隐藏真值，一月无梯度更新；评分复用 task1.metrics。',
        f'日张量缓存：training_hit={cache["training_hit"]}；evaluation_hit={cache["evaluation_hit"]}；未重训 LightGBM 或 GRIN。',
        'checkpoint 按 ImputeFormer 单模型的一月 S_state 选择。固定组合：双侧速度观测间隔 >60 分钟用 ImputeFormer，其余/缺一侧用冻结 baseline；未调阈值。']
    for row in table.itertuples(index=False):
        lines.append(f'{row.method}：S_state={row.S_state:.6f}；速度 RMSE={row.rmse_speed_kmh:.6f}；每车道流量 RMSE={row.rmse_flow_per_lane:.6f}；'
            f'相对冻结 baseline ΔS={row.delta_S_state_vs_frozen_baseline:+.6f}；相对 GRIN ΔS={row.delta_S_state_vs_GRIN:+.6f}；'
            f'相对原组合 ΔS={row.delta_S_state_vs_current_hybrid:+.6f}。')
    for regime in ['R1','R2','R3']:
        local=detail.loc[detail.regime.eq(regime)].set_index('method')
        lines.append(f'{regime}：baseline={local.loc["Fusion_gap60mean","S_state"]:.6f}；GRIN={local.loc["GRIN","S_state"]:.6f}；'
            f'原组合={local.loc["Baseline_GRIN_gap60","S_state"]:.6f}；ImputeFormer={local.loc["ImputeFormer","S_state"]:.6f}；'
            f'新组合={local.loc["Baseline_ImputeFormer_gap60","S_state"]:.6f}。')
    resources=read('memory.json')
    if resources.get('measured'):
        lines.append(f'本进程 PyTorch 显存峰值：allocated={resources["peak_allocated_gib"]:.3f} GiB；reserved={resources["peak_reserved_gib"]:.3f} GiB；设备={resources["device"]}。')
    lines+=['模型结构、目标函数和计算开销存在差异；这不是等算力比较。',
        '明细见 comparison_models.csv、metrics_by_panel_regime.csv、training_history.csv、memory.json 和各方法的缺口/逐日/拥堵误差表。','']
    log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('a',encoding='utf-8',newline='') as stream:stream.write('\n'.join(lines)+'\n')
    status=dict(status='appended',log_path=str(log_path));save_json(run/'log_update.json',status)
    return status
