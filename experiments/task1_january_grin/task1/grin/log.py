from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from ..data import save_json


def append_grin_log(run, log_path):
    run, log_path = Path(run).resolve(), Path(log_path).resolve()
    complete = json.loads((run / 'COMPLETE.json').read_text(encoding='utf-8'))
    if complete.get('experiment') != 'grin_dev':
        raise ValueError('GRIN run is not completed')
    cfg = json.loads((run / 'run_config.json').read_text(encoding='utf-8'))
    if cfg.get('max_fit_days') is not None or cfg.get('max_eval_days') is not None:
        status = dict(status='skipped_smoke')
    elif cfg.get('no_log'):
        status = dict(status='skipped_by_option')
    else:
        status = None
    if status:
        save_json(run / 'log_update.json', status)
        return status
    table = pd.read_csv(run / 'comparison_vs_baseline.csv')
    if not {'GRIN','Fusion_gap60mean'}.issubset(set(table.method)) or table.method.duplicated().any() or not table.all_regimes_present.all():
        raise ValueError('Missing GRIN/baseline or incomplete regimes')
    selection = json.loads((run / 'selection.json').read_text(encoding='utf-8'))
    marker = f'实验编号：{run.name}；结果目录：{run}'
    previous = log_path.read_text(encoding='utf-8-sig') if log_path.exists() else ''
    if marker in previous:
        status = dict(status='already_logged', log_path=str(log_path))
        save_json(run / 'log_update.json', status)
        return status
    date = datetime.now(ZoneInfo('America/Los_Angeles')).date().isoformat()
    lines = ['', f'【GRIN 第一轮｜{date}｜已完成】', marker, '方向：' + cfg['panel'],
        f'训练：{cfg["fit_start"]} 至 {cfg["fit_end"]}；开发：{cfg["eval_start"]} 至 {cfg["eval_end"]}；二月未使用。',
        '架构：GRIN 核心结构的独立 PyTorch 适配实现；双向图循环、两阶段填补、MLP 融合；不是原论文实验的逐项复现。',
        f'训练标签每天抽样 {cfg["sample_per_day"]}；seed={cfg["seed"]}；训练样本数={selection["training_rows"]}。',
        f'窗口={cfg["window"]}（每步 5 分钟）；stride={cfg["stride"]}；hidden={cfg["hidden"]}；batch={cfg["batch_size"]}；'
        f'最多 {cfg["epochs"]} epochs、每 epoch 最多 {cfg["steps_per_epoch"]} 步；最佳 epoch={selection["best_epoch"]}。',
        '只用训练期可见观测拟合标准化；输入来自 masked 文件；训练真值只用于选定训练标签。',
        '训练额外随机遮挡少量可见观测，采用评分尺度加权 MSE 及方向辅助损失。',
        '按一月开发 S_state 选择 checkpoint；一月没有梯度更新；滑窗结果重叠加权合并后按完整目标评分。',
        '评分直接复用 task1.metrics；RMSE 为 regime、方向、走廊等权聚合。冻结 baseline 采用多跳残差/直接＋60 分钟融合。']
    for row in table.itertuples(index=False):
        lines.append(f'{row.method}：S_state={row.S_state:.6f}；速度 RMSE={row.rmse_speed_kmh:.6f}；'
            f'流量 RMSE={row.rmse_flow_per_lane:.6f}；相对冻结 baseline ΔS={row.delta_S_state_vs_frozen_baseline:+.6f}。')
    lines += ['数据期与评分对齐；模型上下文、训练目标和计算预算不同，不能将结果解释为同等算力比较。',
              '明细见 comparison_vs_baseline.csv、metrics_by_panel_regime.csv、training_history.csv、selection.json。', '']
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8', newline='') as stream:
        stream.write('\n'.join(lines) + '\n')
    status = dict(status='appended', log_path=str(log_path))
    save_json(run / 'log_update.json', status)
    return status
