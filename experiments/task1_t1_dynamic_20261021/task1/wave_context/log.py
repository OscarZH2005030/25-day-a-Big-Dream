from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

from ..data import save_json
from .reference import METHODS


def append_log(run, log_path):
    run, log_path = Path(run).resolve(), Path(log_path).resolve()
    read = lambda name:json.loads((run/name).read_text(encoding='utf-8'))
    complete, cfg = read('COMPLETE.json'), read('run_config.json')
    if complete.get('experiment')!='wave_context':
        raise ValueError('Completed three-method long-context experiment required')
    status = 'skipped_smoke' if complete.get('smoke') or cfg.get('smoke') else 'skipped_by_option' if cfg.get('no_log') else None
    if status:
        result = dict(status=status); save_json(run/'log_update.json', result); return result
    if complete.get('training_rows')!=214000 or complete.get('evaluation_rows')!=550149 or len(cfg['actual_fit_dates'])!=214 or len(cfg['actual_eval_dates'])!=31:
        raise ValueError('Formal dates/targets incomplete')
    if set(complete['methods'])!=set(METHODS) or not all(complete.get(k) for k in ['reference_passed','pairing_passed','integrity_passed','all_regimes_present']):
        raise ValueError('Formal comparison incomplete')
    for name in ['reference_check.json','sample_check.json','integrity_check.json','anchor_check.json','receptive_field_check.json']:
        if not read(name).get('passed'):
            raise ValueError('Formal integrity check failed: '+name)
    if cfg['layers']!=6 or cfg['hidden']!=32 or cfg['epochs']!=30 or cfg['window']!=48:
        raise ValueError('Controlled architecture/budget differs')
    table = pd.read_csv(run/'comparison_methods.csv'); detail = pd.read_csv(run/'metrics_by_panel_regime.csv')
    if set(table.method)!=set(METHODS) or table.method.duplicated().any() or not table.all_regimes_present.all():
        raise ValueError('Missing/duplicate method')
    if not np.isfinite(table[['S_state','rmse_speed_kmh','rmse_flow_per_lane']].to_numpy()).all():
        raise ValueError('Nonfinite formal metrics')
    for method in METHODS:
        group = detail.loc[detail.method.eq(method)]
        if len(group)!=3 or set(group.regime)!={'R1','R2','R3'} or group.n.sum()!=550149:
            raise ValueError('Regime/target coverage incomplete')
        row = table.loc[table.method.eq(method)].iloc[0]
        if any(abs(row[k]-group[k].mean())>1e-9 for k in ['S_state','rmse_speed_kmh','rmse_flow_per_lane']):
            raise ValueError('Macro scores differ from regime detail')
        if not (run/(method+'_predictions.parquet')).is_file():
            raise ValueError('Missing saved prediction')
    usage, memory = read('cache_usage.json'), read('memory.json')
    models = usage['models']
    if len(models)!=2 or {r['method'] for r in models}!=set(METHODS[1:]) or len({r['initial_weights_sha256'] for r in models})!=1:
        raise ValueError('Paired initial weights or model count differ')
    if len(usage['training_days'])!=214 or len(usage['evaluation_days'])!=31 or usage['daily_inputs_rebuilt']:
        raise ValueError('Expected immutable original daily inputs')
    histories = []
    for r in models:
        if not 0<r['best_checkpoint_coverage']['unique_targets_seen']<=r['final_training_coverage']['unique_targets_seen']<=214000:
            raise ValueError('Invalid actual target coverage')
        h = pd.read_csv(run/(r['method']+'_training_history.csv'))
        if len(h)!=r['epochs_run'] or int(h.steps.sum())!=r['training_steps']:
            raise ValueError('Training history differs from selection')
        histories.append(h)
    shared = min(len(h) for h in histories)
    if histories[0].window_order_sha256.iloc[:shared].tolist()!=histories[1].window_order_sha256.iloc[:shared].tolist():
        raise ValueError('Paired training orders differ')
    marker = f'实验编号：{run.name}；结果目录：{run}'
    previous = log_path.read_text(encoding='utf-8-sig') if log_path.exists() else ''
    if marker in previous:
        result = dict(status='already_logged', log_path=str(log_path)); save_json(run/'log_update.json', result); return result
    date = datetime.now(ZoneInfo('America/Los_Angeles')).date().isoformat()
    lines = ['', f'【相同插值先验＋扩大时间感受野＋波特征对照｜{date}｜已完成】', marker, '方向：D12_I5_N。',
        '训练2030-06-01至2030-12-31，开发2031完整一月；二月未读取。原214000目标池，每天1000，seed=20261016；实际访问覆盖另列。',
        '冻结对照复用交通波增强LightGBM＋GRIN；两个新网络在所有缺口输出，只有13个波输入通道开关不同。',
        '两组始终共用时间插值先验、18个公共通道、初始化、图结构、监督目标和每轮批次顺序；无额外遮挡/可见监督/物理惩罚。',
        f'6层hidden32图/时间卷积，dilation=1,2,4,8,16,32；理论127步，但实际只输入48步。首/中/尾梯度探针均覆盖全部48步；stride=24，batch={cfg["batch_size"]}。',
        '本轮最多30epochs（上一轮20），每轮最多100steps，patience=5，AdamW lr=0.001；按一月官方S_state选checkpoint。与旧网络的变化含模型深度和训练上限，不能单独归因为时间范围。',
        '评分沿用task1.metrics：速度与每车道流量，regime→方向→家族等权；流量仅导出时乘一次车道数。',
        f'245日输入缓存全部复用，无旧模型重训；新模型缓存命中={sum(r["hit"] for r in models)}/2。输入来源：{cfg["input_run"]}。']
    for row in table.itertuples(index=False):
        lines.append(f'{row.method}：S_state={row.S_state:.6f}；速度RMSE={row.rmse_speed_kmh:.6f}；每车道流量RMSE={row.rmse_flow_per_lane:.6f}；对当前最佳ΔS={row.delta_S_state_vs_current_best:+.6f}；对无波网络ΔS={row.delta_S_state_vs_no_wave:+.6f}。')
    for regime in ['R1','R2','R3']:
        group = detail.loc[detail.regime.eq(regime)].set_index('method')
        lines.append(regime+'：'+'；'.join(f'{m}={group.loc[m,"S_state"]:.6f}' for m in METHODS)+'。')
    for r in models:
        lines.append(f'{r["method"]}：best_epoch={r["best_epoch"]}；实际epochs={r["epochs_run"]}；steps={r["training_steps"]}；参数={r["parameters"]}；最佳时唯一训练目标={r["best_checkpoint_coverage"]["unique_targets_seen"]}/214000；训练结束时={r["final_training_coverage"]["unique_targets_seen"]}/214000。')
    if memory.get('measured'):
        lines.append(f'PyTorch峰值allocated={memory["peak_allocated_gib"]:.3f}GiB，reserved={memory["peak_reserved_gib"]:.3f}GiB；设备={memory["device"]}。')
    lines += ['分项见comparison_methods.csv、metrics_by_panel_regime.csv、各方法分日/缺口/拥堵误差与训练历史。', '']
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8', newline='') as stream:
        stream.write('\n'.join(lines)+'\n')
    result = dict(status='appended', log_path=str(log_path)); save_json(run/'log_update.json', result); return result
