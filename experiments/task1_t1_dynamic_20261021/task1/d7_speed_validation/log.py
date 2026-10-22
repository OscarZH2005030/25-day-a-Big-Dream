from __future__ import annotations

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import numpy as np
import pandas as pd
from ..data import save_json,sha256
from ..speed_fusion.rules import CONTROL,NEURAL,WEIGHTS,PRIMARY,METHODS,RULE
from .reference import PANEL,TRAIN_ROWS,EVAL_ROWS,SPLIT,OLD_METHOD,NN_EXPECTED


def append_log(run,log_path):
    run,log_path=Path(run).resolve(),Path(log_path).resolve()
    read=lambda name:json.loads((run/name).read_text(encoding='utf-8'))
    complete,cfg=read('COMPLETE.json'),read('run_config.json')
    if complete.get('experiment')!='d7_speed_validation':raise ValueError('Completed D7 speed validation required')
    status='skipped_smoke' if complete.get('smoke') or cfg.get('smoke') else 'skipped_by_option' if cfg.get('no_log') else None
    if status:
        result=dict(status=status);save_json(run/'log_update.json',result);return result
    if cfg.get('panel')!=PANEL or complete.get('training_rows')!=TRAIN_ROWS or complete.get('evaluation_rows')!=EVAL_ROWS:
        raise ValueError('D7 formal target scope incomplete')
    if cfg['actual_fit_dates']!=pd.date_range('2030-06-01','2030-12-31').strftime('%Y-%m-%d').tolist() or cfg['actual_eval_dates']!=pd.date_range('2031-01-01','2031-01-31').strftime('%Y-%m-%d').tolist():
        raise ValueError('D7 formal date scope differs')
    if any(cfg.get(k)!=v for k,v in SPLIT.items()) or any(cfg.get(k)!=v for k,v in NN_EXPECTED.items()) or cfg.get('rounds')!=350:
        raise ValueError('Frozen data/neural/LightGBM recipe differs')
    if cfg.get('weights')!=WEIGHTS or cfg.get('primary_method')!=PRIMARY or cfg.get('rule')!=RULE or set(complete['methods'])!=set(METHODS) or len(complete['methods'])!=4:
        raise ValueError('Frozen gate/weights/methods differ')
    if not all(complete.get(k) for k in ['reference_passed','pairing_passed','integrity_passed','all_regimes_present']):raise ValueError('Formal validation incomplete')
    if sha256(run/'artifact_manifest.json')!=complete.get('artifact_manifest_sha256'):raise ValueError('Artifact manifest changed')
    names=set()
    for item in read('artifact_manifest.json')['artifacts']:
        path=Path(item['path']).resolve()
        if not path.is_relative_to(run) or not path.is_file() or sha256(path)!=item['sha256']:raise ValueError('Formal artifact changed/missing: '+str(path))
        names.add(str(path.relative_to(run)))
    required={'run_config.json','frozen_recipe.json','reference_check.json','sample_check.json','neural_checks.json','integrity_check.json',
        'preservation_checks.json','cache_usage.json','source_controls.csv','comparison_methods.csv','comparison_with_i5.csv','metrics_by_panel_regime.csv',
        'fusion_gate.parquet','gate_counts.csv','neural_selection.json','neural_training_history.csv'}
    required.update(m+'_predictions.parquet' for m in METHODS)
    if not required.issubset(names):raise ValueError('Missing protected validation artifact')
    for name in ['reference_check.json','sample_check.json','neural_checks.json','integrity_check.json','preservation_checks.json']:
        if not read(name).get('passed'):raise ValueError('Formal integrity failed: '+name)
    preserved=read('preservation_checks.json');eligible=complete['gated_rows']
    if preserved.get('reference')!=CONTROL or not 0<eligible<EVAL_ROWS or not preserved['new_reference_long_gap_equals_saved_GRIN']['passed']:
        raise ValueError('Reference/gate/GRIN preservation differs')
    for method,alpha in [(CONTROL,0.),*WEIGHTS.items()]:
        item=preserved['methods'][method]
        if not item.get('passed') or item['alpha']!=alpha or item['target_rows']!=EVAL_ROWS or item['eligible_rows']!=eligible or item['preserved_speed_rows']!=EVAL_ROWS-eligible:
            raise ValueError('Preservation target scope differs')
        if not all(item[k] for k in ['all_flow_values_equal','all_outside_speed_values_equal','all_inside_speed_values_match_formula']):raise ValueError('Flow/non-gated speed/formula preservation failed')
    usage=read('cache_usage.json');events=usage['events']
    if any(events[k] for k in ['original_baseline_retrained','original_GRIN_retrained','original_GRIN_inference']):raise ValueError('Original D7 model reuse failed')
    for key in ['wave_prediction','neural_prediction']:
        if usage[key]['metadata']['training_rows']!=TRAIN_ROWS or usage[key]['metadata']['evaluation_rows']!=EVAL_ROWS:
            raise ValueError('Source prediction scope differs')
    neural=read('neural_selection.json')
    if neural['available_training_rows']!=TRAIN_ROWS or neural['parameters']!=38722 or not 0<neural['best_checkpoint_coverage']['unique_targets_seen']<=TRAIN_ROWS:
        raise ValueError('Neural architecture/coverage differs')
    table=pd.read_csv(run/'comparison_methods.csv');detail=pd.read_csv(run/'metrics_by_panel_regime.csv')
    if set(table.method)!=set(METHODS) or table.method.duplicated().any() or not table.all_regimes_present.all():raise ValueError('Missing/duplicate methods')
    metrics=['S_state','rmse_speed_kmh','rmse_flow_per_lane']
    if not np.isfinite(table[metrics].to_numpy()).all():raise ValueError('Nonfinite formal scores')
    base=table.loc[table.method.eq(CONTROL)].iloc[0]
    for method in METHODS:
        row=table.loc[table.method.eq(method)].iloc[0];group=detail.loc[detail.method.eq(method)]
        if len(group)!=3 or set(group.regime)!={'R1','R2','R3'} or group.n.sum()!=EVAL_ROWS:raise ValueError('Regime coverage incomplete')
        if any(abs(row[k]-group[k].mean())>1e-9 for k in metrics):raise ValueError('Macro regime score differs')
        if abs(row.delta_S_state_vs_current_best-(row.S_state-base.S_state))>1e-12 or abs(row.rmse_flow_per_lane-base.rmse_flow_per_lane)>1e-12:
            raise ValueError('Delta/flow scores differ')
    cross=pd.read_csv(run/'comparison_with_i5.csv')
    if set(cross.method)!=set(METHODS) or cross.method.duplicated().any():raise ValueError('I5/D7 pairing incomplete')
    source=pd.read_csv(run/'source_controls.csv').set_index('method')
    if abs(source.loc[CONTROL,'S_state']-base.S_state)>1e-12:raise ValueError('Wave reference score differs')
    marker=f'实验编号：{run.name}；结果目录：{run}'
    previous=log_path.read_text(encoding='utf-8-sig') if log_path.exists() else ''
    if marker in previous:
        result=dict(status='already_logged',log_path=str(log_path));save_json(run/'log_update.json',result);return result
    date=datetime.now(ZoneInfo('America/Los_Angeles')).date().isoformat()
    lines=['',f'【D7_I405_N 冻结I5配方速度融合验证｜{date}｜已完成】',marker,
        'D7独立拟合；训练2030年6–12月214000原目标，开发2031完整一月819695目标；352节点。二月观测/真值未读。',
        '原D7 baseline/GRIN预测、日张量、scaler、特征和直接目标sidecar复用；原模型未训练或推理。补齐相同I5波增强LightGBM与长时间范围无波网络。',
        '本轮共同参照是I5冻结波增强配方在D7的复现；旧D7 baseline＋GRIN单独报告。融合流量和非门控速度的不变项均相对新参照。',
        '仅双侧有限正数且15<gap<=60分钟融合速度；10%/25%/50%预定权重，25%继续为主候选。无D7权重/阈值/模型超参搜索。',
        f'门控={eligible}/{EVAL_ROWS}；其余{EVAL_ROWS-eligible}行速度和全部{EVAL_ROWS}行流量逐值保留；>60分钟参照两通道等于已保存GRIN。',
        f'旧D7组合S={source.loc[OLD_METHOD,"S_state"]:.6f}；波增强参照S={base.S_state:.6f}；波增强相对旧组合ΔS={base.S_state-source.loc[OLD_METHOD,"S_state"]:+.6f}；无波网络S={source.loc[NEURAL,"S_state"]:.6f}。']
    for row in table.itertuples(index=False):lines.append(f'{row.method}：S_state={row.S_state:.6f}；速度RMSE={row.rmse_speed_kmh:.6f}；每车道流量RMSE={row.rmse_flow_per_lane:.6f}；相对D7波增强参照ΔS={row.delta_S_state_vs_current_best:+.6f}。')
    for regime in ['R1','R2','R3']:
        g=detail.loc[detail.regime.eq(regime)].set_index('method');lines.append(regime+'：'+'；'.join(f'{m}={g.loc[m,"S_state"]:.6f}' for m in METHODS)+'。')
    for row in cross.itertuples(index=False):lines.append(f'{row.method}：I5 ΔS={row.I5_delta_S:+.6f}；D7 ΔS={row.D7_delta_S:+.6f}；两北向走廊家族等权均分={row.two_north_corridor_family_mean_S_state:.6f}（非全赛道）。')
    lines += [f'无波网络best_epoch={neural["best_epoch"]}，实际epochs={neural["epochs_run"]}，steps={neural["training_steps"]}，最佳时访问唯一目标={neural["best_checkpoint_coverage"]["unique_targets_seen"]}/214000；同I5最多30×100步、batch2。',
        f'来源预测缓存命中：波增强={usage["wave_prediction"]["hit"]}，无波={usage["neural_prediction"]["hit"]}；本次新波模型训练={events["wave_models_trained"]}，无波训练={events["neural_model_trained"]}。',
        '相同开发规则在D7选checkpoint，属于跨走廊开发验证；融合规则固定，但仍需后续留出验证。未自动替换冻结基准。','']
    log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('a',encoding='utf-8',newline='') as stream:stream.write('\n'.join(lines)+'\n')
    result=dict(status='appended',log_path=str(log_path));save_json(run/'log_update.json',result);return result
