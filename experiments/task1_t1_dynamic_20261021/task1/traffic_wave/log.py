from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from ..data import save_json


def append_log(run,log_path):
    run,log_path=Path(run).resolve(),Path(log_path).resolve()
    read=lambda name:json.loads((run/name).read_text(encoding='utf-8'))
    complete,cfg=read('COMPLETE.json'),read('run_config.json')
    if complete.get('experiment')!='traffic_wave':raise ValueError('Completed traffic-wave experiment required')
    if cfg.get('smoke') or complete.get('smoke'):status=dict(status='skipped_smoke')
    elif cfg.get('no_log'):status=dict(status='skipped_by_option')
    else:status=None
    if status:save_json(run/'log_update.json',status);return status
    required={'Current_best_hybrid','Traffic_wave','Wave_enhanced_hybrid'}
    table=pd.read_csv(run/'comparison_methods.csv');detail=pd.read_csv(run/'metrics_by_panel_regime.csv')
    if set(table.method)!=required or set(complete['methods'])!=required or table.method.duplicated().any() or not table.all_regimes_present.all():
        raise ValueError('Incomplete formal three-method comparison')
    if not np.isfinite(table[['S_state','rmse_speed_kmh','rmse_flow_per_lane']].to_numpy()).all():raise ValueError('Non-finite metrics')
    if not all(complete.get(k) for k in ['reference_passed','pairing_passed','integrity_passed']):raise ValueError('Formal integrity checks failed')
    if not all(read(file).get('passed') for file in ['reference_check.json','sample_check.json','integrity_check.json']):raise ValueError('Missing successful checks')
    if complete['training_rows']!=214000 or len(cfg['actual_fit_dates'])!=214 or len(cfg['actual_eval_dates'])!=31:
        raise ValueError('Formal dates/sample count differ')
    for method in required:
        local=detail.loc[detail.method.eq(method)]
        if len(local)!=3 or set(local.regime)!={'R1','R2','R3'}:raise ValueError('Incomplete regime detail')
    marker=f'实验编号：{run.name}；结果目录：{run}'
    previous=log_path.read_text(encoding='utf-8-sig') if log_path.exists() else ''
    if marker in previous:
        status=dict(status='already_logged',log_path=str(log_path));save_json(run/'log_update.json',status);return status
    usage=read('cache_usage.json');memory=read('memory.json');integrity=read('integrity_check.json')
    date=datetime.now(ZoneInfo('America/Los_Angeles')).date().isoformat()
    lines=['',f'【交通波传播重建与增强｜{date}｜已完成】',marker,'方向：D12_I5_N',
        '训练：2030-06-01 至 2030-12-31；开发：2031-01完整月份；二月未使用。',
        f'原训练目标214000个（每天1000，seed=20261016），与原 baseline/GRIN 有序自然键一致；LightGBM {cfg["rounds"]}轮。',
        f'波参数：c_free={cfg["c_free"]}km/h；c_cong={cfg["c_congested"]}km/h；sigma={cfg["sigma_km"]}km；'
        f'tau={cfg["tau_minutes"]}分钟；空间半径={cfg["radius_km"]}km；时间半径={cfg["time_radius_minutes"]}分钟。',
        '波输入仅当天masked可见速度/每车道流量和mask；真实主线距离；可见速度软切换；无支持时退回原训练历史均值。',
        f'增强方案：原86个多跳特征增加{integrity["new_feature_count"]}个波特征，按原参数训练残差/直接速度/流量四个LightGBM回归器。',
        '不将旧模型训练集预测作为输入或目标；不训练残差堆叠器；不需要OOF；真实标签使用原精确sidecar。',
        '原残差/直接融合保留；双侧速度观测间隔>60分钟仍用原GRIN，其余用增强LightGBM；未调阈值，未重训GRIN。',
        f'原最佳组合预测复用；增强模型缓存命中={usage["enhanced_models"]["hit"]}；波训练日缓存命中={sum(d["hit"] for d in usage["training_days"])}/{len(usage["training_days"])}；'
        f'波开发日缓存命中={sum(d["hit"] for d in usage["evaluation_days"])}/{len(usage["evaluation_days"])}。',
        '评分直接复用task1.metrics；速度与每车道流量按regime→方向→家族等权聚合。']
    for row in table.itertuples(index=False):
        lines.append(f'{row.method}：S_state={row.S_state:.6f}；速度RMSE={row.rmse_speed_kmh:.6f}；每车道流量RMSE={row.rmse_flow_per_lane:.6f}；相对原最佳ΔS={row.delta_S_state_vs_current_best:+.6f}。')
    for regime in ['R1','R2','R3']:
        group=detail.loc[detail.regime.eq(regime)].set_index('method')
        lines.append(f'{regime}：原最佳={group.loc["Current_best_hybrid","S_state"]:.6f}；纯交通波={group.loc["Traffic_wave","S_state"]:.6f}；增强组合={group.loc["Wave_enhanced_hybrid","S_state"]:.6f}。')
    if memory.get('measured'):lines.append(f'PyTorch显存峰值：allocated={memory["peak_allocated_gib"]:.3f}GiB；reserved={memory["peak_reserved_gib"]:.3f}GiB。')
    lines+=['本轮为独立交通波信息的特征消融；不声称Task3本地物理分数。详见comparison_methods.csv及缺口/逐日/拥堵、覆盖度和特征重要性文件。','']
    log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('a',encoding='utf-8',newline='') as stream:stream.write('\n'.join(lines)+'\n')
    status=dict(status='appended',log_path=str(log_path));save_json(run/'log_update.json',status);return status
