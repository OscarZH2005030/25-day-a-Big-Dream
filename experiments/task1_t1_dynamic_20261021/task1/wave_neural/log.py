from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

from ..data import save_json


def append_log(run,log_path):
    from .reference import CONTROL,METHODS
    run,log_path=Path(run).resolve(),Path(log_path).resolve()
    read=lambda name:json.loads((run/name).read_text(encoding='utf-8'))
    complete,cfg=read('COMPLETE.json'),read('run_config.json')
    if complete.get('experiment')!='wave_neural':raise ValueError('Completed five-method neural experiment required')
    status='skipped_smoke' if complete.get('smoke') or cfg.get('smoke') else 'skipped_by_option' if cfg.get('no_log') else None
    if status:
        result=dict(status=status);save_json(run/'log_update.json',result);return result
    if complete['training_rows']!=214000 or complete['evaluation_rows']!=550149 or len(cfg['actual_fit_dates'])!=214 or len(cfg['actual_eval_dates'])!=31:
        raise ValueError('Formal dates or target pool differ')
    if set(complete['methods'])!=set(METHODS) or not all(complete.get(k) for k in ['reference_passed','pairing_passed','integrity_passed','all_regimes_present']):
        raise ValueError('Incomplete formal comparison')
    if not all(read(name).get('passed') for name in ['reference_check.json','sample_check.json','integrity_check.json']):
        raise ValueError('Formal integrity checks missing')
    table=pd.read_csv(run/'comparison_methods.csv');detail=pd.read_csv(run/'metrics_by_panel_regime.csv')
    if set(table.method)!=set(METHODS) or table.method.duplicated().any() or not table.all_regimes_present.all():
        raise ValueError('Missing or duplicate method')
    if not np.isfinite(table[['S_state','rmse_speed_kmh','rmse_flow_per_lane']].to_numpy()).all():
        raise ValueError('Nonfinite formal metrics')
    for method in METHODS:
        group=detail.loc[detail.method.eq(method)]
        if len(group)!=3 or set(group.regime)!={'R1','R2','R3'} or group.n.sum()!=550149:
            raise ValueError('Regime/target coverage incomplete')
        row=table.loc[table.method.eq(method)].iloc[0]
        if max(abs(row.S_state-group.S_state.mean()),abs(row.rmse_speed_kmh-group.rmse_speed_kmh.mean()),
               abs(row.rmse_flow_per_lane-group.rmse_flow_per_lane.mean()))>1e-9:
            raise ValueError('Aggregate scores differ from regime detail')
        if not (run/(method+'_predictions.parquet')).is_file():raise ValueError('Missing saved predictions')
    usage=read('cache_usage.json');coverage=read('physical_coverage.json');memory=read('memory.json')
    if len(usage['models'])!=4 or {r['method'] for r in usage['models']}!=set(METHODS[1:]) or len({r['initial_weights_sha256'] for r in usage['models']})!=1:
        raise ValueError('Four model initializations/sources differ')
    marker=f'实验编号：{run.name}；结果目录：{run}'
    previous=log_path.read_text(encoding='utf-8-sig') if log_path.exists() else ''
    if marker in previous:
        status=dict(status='already_logged',log_path=str(log_path));save_json(run/'log_update.json',status);return status
    date=datetime.now(ZoneInfo('America/Los_Angeles')).date().isoformat()
    lines=['',f'【交通波＋神经修正＋物理约束消融｜{date}｜已完成】',marker,'方向：D12_I5_N',
        '训练：2030-06-01至2030-12-31；开发：2031完整一月；二月未读取。',
        '214000个原监督目标，每天1000，seed=20261016；直接复用GRIN日张量、scaler与既有波缓存。无额外随机遮挡/可见值监督；无旧模型自身训练预测输入。',
        '冻结对照为交通波增强LightGBM＋原GRIN组合；4个新网络均在全部缺口输出，不使用GRIN切换。',
        f'小型图时序残差网络：hidden={cfg["hidden"]}；layers={cfg["layers"]}；window={cfg["window"]}；stride={cfg["stride"]}；batch={cfg["batch_size"]}；相同初始化与批次顺序。',
        f'训练上限：{cfg["epochs"]}epochs，每轮{cfg["steps_per_epoch"]}steps，patience={cfg["patience"]}，AdamW lr={cfg["learning_rate"]}；checkpoint按一月S_state选择。实际早停预算见各组。',
        f'固定lambda_FD={cfg["lambda_fd"]}，lambda_LWR={cfg["lambda_lwr"]}，warmup={cfg["warmup_epochs"]}epochs。FD是带10%容差的可行域软约束，不强制基本图等式；无q=kv恒等式损失。',
        f'局部守恒代理覆盖{coverage["conservation_nodes"]}/{coverage["nodes"]}节点，其中有附着匝道的节点数={coverage["conservation_nodes_with_ramps"]}；未知connector/外部边界/分支跳过。',
        '匝道缺失或质量无效时跳过约束，仅无附着匝道才结构性记零。守恒使用总流量、dt=5/60小时、公开中心流量的中点界面代理；不是官方隐藏边界复现。',
        'physics_proxy_diagnostics是本地代理，不是官方S_physics；所有方法在未提交缺失格使用相同时间插值背景以便比较。',
        f'原模型未重训；训练日输入缓存命中={sum(r["hit"] for r in usage["training_days"])}/{len(usage["training_days"])}；开发日输入缓存命中={sum(r["hit"] for r in usage["evaluation_days"])}/{len(usage["evaluation_days"])}；神经模型缓存命中={sum(r["hit"] for r in usage["models"])}/4。',
        '评分复用task1.metrics；速度与每车道流量，regime→方向→走廊家族等权。']
    for row in table.itertuples(index=False):
        lines.append(f'{row.method}：S_state={row.S_state:.6f}；速度RMSE={row.rmse_speed_kmh:.6f}；每车道流量RMSE={row.rmse_flow_per_lane:.6f}；相对当前最佳ΔS={row.delta_S_state_vs_current_best:+.6f}。')
    for regime in ['R1','R2','R3']:
        group=detail.loc[detail.regime.eq(regime)].set_index('method')
        lines.append(regime+'：'+'；'.join(f'{method}={group.loc[method,"S_state"]:.6f}' for method in METHODS)+'。')
    for item in usage['models']:
        lines.append(f'{item["method"]}：best_epoch={item["best_epoch"]}；epochs_run={item["epochs_run"]}；training_steps={item["training_steps"]}；parameters={item["parameters"]}。')
    if memory.get('measured'):
        lines.append(f'PyTorch峰值allocated={memory["peak_allocated_gib"]:.3f}GiB；reserved={memory["peak_reserved_gib"]:.3f}GiB；设备={memory["device"]}。')
    lines+=['分项见comparison_methods.csv、metrics_by_panel_regime.csv、各组分日/缺口/拥堵误差、训练历史和physics_proxy_diagnostics.csv。','']
    log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('a',encoding='utf-8',newline='') as stream:stream.write('\n'.join(lines)+'\n')
    status=dict(status='appended',log_path=str(log_path));save_json(run/'log_update.json',status);return status
