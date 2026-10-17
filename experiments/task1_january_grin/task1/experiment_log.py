"""Append completed formal target comparisons to the shared plain-text log."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .data import save_json


def append_target_log(run: Path, log_path: Path):
    run, log_path = Path(run).resolve(), Path(log_path).resolve()
    completion = json.loads((run / 'COMPLETE.json').read_text(encoding='utf-8'))
    if completion.get('experiment') != 'target_comparison':
        raise ValueError('Only completed target-comparison runs can be logged here')
    cfg = json.loads((run / 'run_config.json').read_text(encoding='utf-8'))
    if cfg.get('max_fit_days') is not None or cfg.get('max_eval_days') is not None:
        status = dict(status='skipped_smoke', reason='Subset checks are not formal experimental conclusions')
        save_json(run / 'log_update.json', status)
        return status
    if cfg.get('no_log'):
        status = dict(status='skipped_by_option')
        save_json(run / 'log_update.json', status)
        return status
    if cfg.get('reference_run'):
        checks = json.loads((run / 'reference_check.json').read_text(encoding='utf-8'))
        if not checks or not all(x['passed'] for x in checks):
            raise ValueError('Cannot log a run whose residual-reference checks did not pass')
    table = pd.read_csv(run / 'comparison_target_modes.csv')
    marker = f'实验编号：{run.name}；结果目录：{run}'
    existing = log_path.read_text(encoding='utf-8-sig') if log_path.exists() else ''
    if marker in existing:
        status = dict(status='already_logged', log_path=str(log_path))
        save_json(run / 'log_update.json', status)
        return status
    date = datetime.now(ZoneInfo('America/Los_Angeles')).date().isoformat()
    lines = ['', f'【残差预测 vs 直接预测｜{date}｜已完成】', marker,
        '方向：' + '、'.join(completion['panels']),
        f'训练：{cfg["fit_start"]} 至 {cfg["fit_end"]}；评测：{cfg["eval_start"]} 至 {cfg["eval_end"]}。',
        f'每天抽样 {cfg["sample_per_day"]}；{cfg["rounds"]} 轮；seed={cfg["seed"]}；线程={cfg["threads"]}。',
        '每对模型使用相同特征、相同样本和参数；直接标签来自训练真值，流量按每车道。',
        '分数为本地开发结果；RMSE 按 regime、方向和走廊等权聚合。']
    for _, row in table.iterrows():
        lines.extend([
            f'{row.feature_variant}（{int(row.n_features)} 个特征）：',
            f'  残差：S_state={row.residual_S_state:.6f}；速度 RMSE={row.residual_speed_rmse:.6f}；流量 RMSE={row.residual_flow_rmse:.6f}。',
            f'  直接：S_state={row.direct_S_state:.6f}；速度 RMSE={row.direct_speed_rmse:.6f}；流量 RMSE={row.direct_flow_rmse:.6f}。',
            f'  直接减残差：{row.delta_direct_minus_residual:+.6f}；本次分数较高者：{row.winner}。'])
    lines += ['本条将此前“残差 vs 直接预测”的待做项更新为已完成；结论仅适用于本次方向和开发期。',
              '详细分项见 comparison_target_modes.csv、metrics_by_panel_regime.csv 和 evaluation 下的误差表。', '']
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8', newline='') as stream:
        stream.write('\n'.join(lines) + '\n')
    status = dict(status='appended', log_path=str(log_path))
    save_json(run / 'log_update.json', status)
    return status


def log_command(args):
    result = append_target_log(args.run, args.log_path)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def append_fusion_log(run: Path, log_path: Path):
    run, log_path = Path(run).resolve(), Path(log_path).resolve()
    completion = json.loads((run / 'COMPLETE.json').read_text(encoding='utf-8'))
    if completion.get('experiment') != 'multihop_fusion' or not completion.get('reference_passed'):
        raise ValueError('Only completed, reference-checked fusion runs can be logged')
    checks = json.loads((run / 'reference_check.json').read_text(encoding='utf-8'))
    required = {(p, m) for p in completion['panels'] for m in completion['controls']}
    if {(c['panel'], c['method']) for c in checks} != required or not all(c['passed'] for c in checks):
        raise ValueError('Fusion control reference checks did not pass for every panel/method')
    cfg = json.loads((run / 'run_config.json').read_text(encoding='utf-8'))
    if cfg.get('smoke') or cfg.get('max_fit_days') is not None or cfg.get('max_eval_days') is not None:
        status = dict(status='skipped_smoke', reason='Subset checks are not formal experimental conclusions')
    elif cfg.get('no_log'):
        status = dict(status='skipped_by_option')
    else:
        status = None
    if status:
        save_json(run / 'log_update.json', status)
        return status
    table = pd.read_csv(run / 'comparison_fusions.csv')
    required_methods = set(completion['controls']) | set(completion['rules'])
    if set(table.method) != required_methods or table.method.duplicated().any():
        raise ValueError('Incomplete fusion result table')
    marker = f'实验编号：{run.name}；结果目录：{run}'
    existing = log_path.read_text(encoding='utf-8-sig') if log_path.exists() else ''
    if marker in existing:
        status = dict(status='already_logged', log_path=str(log_path))
        save_json(run / 'log_update.json', status)
        return status
    date = datetime.now(ZoneInfo('America/Los_Angeles')).date().isoformat()
    lines = ['', f'【多跳特征组｜残差/直接预测融合｜{date}｜已完成】', marker,
        '来源实验：' + cfg['source_run'], '方向：' + '、'.join(completion['panels']),
        f'原训练：{cfg["fit_start"]} 至 {cfg["fit_end"]}；评测：{cfg["eval_start"]} 至 {cfg["eval_end"]}。',
        '复用已保存的多跳残差/直接预测及特征缓存；没有重新训练；两个原模型逐 regime 复现检查通过。',
        'Fusion_channels：速度用残差，流量用直接；Fusion_mean50：两个模型等权平均。',
        'Fusion_gap30：速度两侧可见观测间隔 ≤30 分钟用残差；>30 分钟或缺一侧用直接，两项输出共用该规则。',
        '切换只用可见观测信息；不使用评测真值选模型。规则来自一月开发分析，仍需跨走廊验证。',
        'S_state 越大越好，RMSE 越小越好；分数和描述性 RMSE 按 regime、方向、走廊等权聚合。']
    for row in table.itertuples(index=False):
        lines.append(f'{row.method}：S_state={row.S_state:.6f}；速度 RMSE={row.rmse_speed_kmh:.6f}；'
            f'流量 RMSE={row.rmse_flow_per_lane:.6f}；相对残差 ΔS={row.delta_S_state_vs_residual:+.6f}；'
            f'相对直接 ΔS={row.delta_S_state_vs_direct:+.6f}。')
    best = table.sort_values('S_state', ascending=False).iloc[0]
    lines += [f'本次开发集最高分：{best.method}（{best.S_state:.6f}）；不据此宣称跨数据稳定提升。',
              '明细见 comparison_fusions.csv、metrics_by_panel_regime.csv 和 evaluation 下的分日/缺口/拥堵误差。', '']
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8', newline='') as stream:
        stream.write('\n'.join(lines) + '\n')
    status = dict(status='appended', log_path=str(log_path))
    save_json(run / 'log_update.json', status)
    return status


def fusion_log_command(args):
    print(json.dumps(append_fusion_log(args.run, args.log_path), ensure_ascii=False), flush=True)


def append_corridor_log(run: Path, log_path: Path):
    run, log_path = Path(run).resolve(), Path(log_path).resolve()
    completion = json.loads((run / 'COMPLETE.json').read_text(encoding='utf-8'))
    if completion.get('experiment') != 'corridor_validation' or not completion.get('pairing_passed') or not completion.get('integrity_passed'):
        raise ValueError('Only completed, checked corridor validations can be logged')
    pairing = json.loads((run / 'pairing_check.json').read_text(encoding='utf-8'))
    integrity = json.loads((run / 'integrity_check.json').read_text(encoding='utf-8'))
    panels = set(completion['panels'])
    if not pairing.get('passed') or not integrity.get('passed') or not panels:
        raise ValueError('Corridor validation checks failed')
    for check in [pairing, integrity]:
        if {r['panel'] for r in check['records']} != panels or not all(r['passed'] for r in check['records']):
            raise ValueError('Missing or failed corridor panel checks')
    cfg = json.loads((run / 'run_config.json').read_text(encoding='utf-8'))
    if cfg.get('max_fit_days') is not None or cfg.get('max_eval_days') is not None:
        status = dict(status='skipped_smoke', reason='Subset checks are not formal experimental conclusions')
    elif cfg.get('no_log'):
        status = dict(status='skipped_by_option')
    else:
        status = None
    if status:
        save_json(run / 'log_update.json', status)
        return status
    table = pd.read_csv(run / 'comparison_corridor.csv')
    panel_table = pd.read_csv(run / 'comparison_by_panel.csv')
    if set(table.method) != set(completion['methods']) or table.method.duplicated().any() or not table.all_regimes_present.all():
        raise ValueError('Incomplete formal corridor result table or regimes')
    for panel in panels:
        local = panel_table.loc[panel_table.panel.eq(panel)]
        if set(local.method) != set(completion['methods']) or local.method.duplicated().any() or not local.all_regimes_present.all():
            raise ValueError('Incomplete per-panel corridor results')
    marker = f'实验编号：{run.name}；结果目录：{run}'
    existing = log_path.read_text(encoding='utf-8-sig') if log_path.exists() else ''
    if marker in existing:
        status = dict(status='already_logged', log_path=str(log_path))
        save_json(run / 'log_update.json', status)
        return status
    date = datetime.now(ZoneInfo('America/Los_Angeles')).date().isoformat()
    usage = json.loads((run / 'cache_usage.json').read_text(encoding='utf-8'))
    lines = ['', f'【另一条走廊验证｜多跳残差/直接/融合｜{date}｜已完成】', marker,
        '方向：' + '、'.join(completion['panels']),
        f'训练：{cfg["fit_start"]} 至 {cfg["fit_end"]}；评测：{cfg["eval_start"]} 至 {cfg["eval_end"]}。',
        f'每天抽样 {cfg["sample_per_day"]}；{cfg["rounds"]} 轮；seed={cfg["seed"]}；线程={cfg["threads"]}。',
        '在新走廊自身训练期独立训练；验证同一方案能否重复有效，不是旧走廊模型直接迁移。',
        '只训练多跳残差和直接两种目标（各包含速度/每车道流量回归器）；样本、特征和参数配对检查通过。',
        'Fusion_mean50：两者各 50%；Fusion_gap60mean：速度两侧可见观测间隔 >60 分钟或缺一侧用直接，其余等权平均，两项输出共用此规则。',
        '阈值与权重沿用 I5 开发结论；没有用新走廊真值调整规则；二月留出集未使用。',
        'S_state 越大越好；描述性 RMSE 按 regime、方向、走廊等权聚合。']
    for u in usage:
        lines.append(f'{u["panel"]} 缓存命中：训练特征={u["training_hit"]}；评测特征={u["evaluation_hit"]}；'
            f'直接标签={u["direct_labels_hit"]}；模型={u["model_pair_hit"]}；预测={u["prediction_hit"]}。')
    def result_line(row):
        return (f'{row.method}：S_state={row.S_state:.6f}；速度 RMSE={row.rmse_speed_kmh:.6f}；'
            f'流量 RMSE={row.rmse_flow_per_lane:.6f}；相对残差 ΔS={row.delta_S_state_vs_residual:+.6f}；'
            f'相对直接 ΔS={row.delta_S_state_vs_direct:+.6f}；相对等权 ΔS={row.delta_S_state_vs_mean50:+.6f}。')
    lines += [result_line(row) for row in table.itertuples(index=False)]
    if len(panels) > 1:
        for panel in completion['panels']:
            lines.append(f'方向明细：{panel}')
            lines += [result_line(row) for row in panel_table.loc[panel_table.panel.eq(panel)].itertuples(index=False)]
    best = table.sort_values('S_state', ascending=False).iloc[0]
    lines += [f'本次最高分：{best.method}（{best.S_state:.6f}）；仍需结合旧走廊结果评估稳定性。',
              '明细见 comparison_corridor.csv、comparison_by_panel.csv、metrics_by_panel_regime.csv 和 evaluation 下的误差表。', '']
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8', newline='') as stream:
        stream.write('\n'.join(lines) + '\n')
    status = dict(status='appended', log_path=str(log_path))
    save_json(run / 'log_update.json', status)
    return status


def corridor_log_command(args):
    print(json.dumps(append_corridor_log(args.run, args.log_path), ensure_ascii=False), flush=True)
