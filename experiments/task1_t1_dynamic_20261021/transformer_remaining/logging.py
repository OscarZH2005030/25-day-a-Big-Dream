from pathlib import Path
from task1.data import save_json
from task1.transformer_priority.common import upstream


def append_batch_log(out,project,cfg,table):
    if cfg['smoke']:
        result=dict(status='skipped_smoke')
    else:
        marker=f'实验编号：{out.name}；结果目录：{out}'
        log=project/'实验日志.txt'
        with upstream().lock(project/'.remaining_dynamic_log.lock'):
            old=log.read_text(encoding='utf-8-sig') if log.exists() else ''
            if marker in old:
                result=dict(status='already_logged',log_path=str(log))
            else:
                lines=['',f'【剩余方向T1动态缺失｜{" / ".join(cfg["panels"])}｜已完成】',marker,
                    '每方向T1/A两个主模型及T0基准；共享LightGBM；28/168/22/55日、原评分、60×1024上限/早停及原孤立回退。',
                    '连通节点两通道固定45%T0+45%候选+10%LightGBM；全部六个神经模型和内部产物冻结后才评估本批留出。']
                for r in table.itertuples(index=False):
                    lines.append(f'{r.method}：S={r.S_state:.9f}；对T1 ΔS={r.delta_vs_T1:+.9f}；对原固定融合 ΔS={r.delta_vs_current_fusion:+.9f}；速度宏RMSE={r.rmse_speed_kmh:.6f}；每车道流量宏RMSE={r.rmse_flow_per_lane:.6f}。')
                lines+=['本条仅是当前两方向批次，不冒充完整十路。逐Regime/日期/缺口/拥堵/孤立及来源、预算和断点记录见结果目录。','']
                with log.open('a',encoding='utf-8',newline='') as stream:
                    stream.write('\n'.join(lines)+'\n')
                result=dict(status='appended',log_path=str(log))
    save_json(out/'log_update.json',result)
    return result


def append_summary_log(out,project,table):
    log=project/'实验日志.txt'
    marker=f'十路汇总编号：{out.name}；结果目录：{out}'
    with upstream().lock(project/'.remaining_dynamic_log.lock'):
        old=log.read_text(encoding='utf-8-sig') if log.exists() else ''
        if marker in old:
            result=dict(status='already_logged',log_path=str(log))
        else:
            lines=['','【T1动态缺失全十路同口径汇总｜已完成】',marker,
                '复用已完成D7_I405、D12_I5四方向，加剩余三批六方向；五个走廊家族、十个方向、各三Regime，按原层级宏平均。',
                '训练/掩码/预算上限/评分/孤立回退和45%T0+45%候选+10%LightGBM均冻结；来源及完整范围检查通过。']
            for r in table.itertuples(index=False):
                lines.append(f'{r.method}：S={r.S_state:.9f}；对T0 ΔS={r.delta_vs_T0:+.9f}；对T1 ΔS={r.delta_vs_T1:+.9f}；对原固定融合 ΔS={r.delta_vs_current_fusion:+.9f}；相对历史0.936910312032 ΔS={r.delta_vs_historical:+.9f}。')
            lines+=['历史目标线不是官方测试成绩；同口径新T0/T1/原融合对照同时报告。单训练种子、已反复查看的本地留出；配对日期区间不包含训练/选择不确定性。','']
            with log.open('a',encoding='utf-8',newline='') as stream:
                stream.write('\n'.join(lines)+'\n')
            result=dict(status='appended',log_path=str(log))
    save_json(out/'log_update.json',result)
    return result
