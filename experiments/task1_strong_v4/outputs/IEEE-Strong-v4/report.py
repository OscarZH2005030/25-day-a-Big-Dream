"""Verify the deliverable against v3, then report measured larger-model results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *


def main():
    preserved=json.loads((ROOT/'preserved-best-sha256.json').read_text())
    assert all(sha(Path(p))==v for p,v in preserved.items())
    routes=json.loads((ROOT/'frozen-routes.json').read_text())['routes'];adopted={r['panel'] for r in routes if r['alpha']>0}
    old=V3/'candidate-submission/submission.csv';candidate=ROOT/'candidate-submission/submission.csv'
    changes=0;rows=0
    if adopted:
        keyfile=WORKSPACE/'work/trafficflowbench-data/kaggle_public/submission_key.csv'
        keys=pd.read_csv(keyfile,usecols=['submission_id','task','panel'],chunksize=250000)
        for a,b,k in zip(pd.read_csv(old,chunksize=250000),pd.read_csv(candidate,chunksize=250000),keys,strict=True):
            assert np.array_equal(a.submission_id,b.submission_id) and np.array_equal(b.submission_id,k.submission_id)
            for col in ['task','speed_kmh','queue_pred','path_flow']:assert np.array_equal(a[col].to_numpy(),b[col].to_numpy()),col
            changed=a.flow_vph.to_numpy()!=b.flow_vph.to_numpy();allowed=(k.task=='state')&k.panel.isin(adopted)
            assert not (changed&~allowed.to_numpy()).any();changes+=int(changed.sum());rows+=len(a)
        assert rows==6980503
        validation=json.loads((ROOT/'candidate-submission/submission-validation.json').read_text())
    else:
        candidate=old;rows=6980503;validation=json.loads((V3/'candidate-submission/submission-validation.json').read_text())
    save(ROOT/'delivery-check.json',dict(passed=True,rows=rows,previous_best_files_unchanged=len(preserved),
        speeds_identical=True,queue_identical=True,odme_identical=True,only_selected_road_flows_changed=True,changed_flow_rows=changes,
        selected_sha256={str((ROOT/r['panel']/'best-model.pt').relative_to(ROOT)):sha(ROOT/r['panel']/'best-model.pt') for r in routes if r['alpha']>0}))
    result=json.loads((ROOT/'evaluation-summary.json').read_text());monitor=json.loads((ROOT/'monitor-summary.json').read_text())
    before=result['baseline'];after=result['new'];table=pd.read_csv(ROOT/'comparison-by-panel.csv')
    table=table[table.panel.isin(PANELS)]
    table.to_csv(ROOT/'priority-directions.csv',index=False)
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for ax,key,title in [(axes[0],'score','Same 55-day local holdout score'),(axes[1],'flow_rmse','Flow RMSE (veh/hour/lane)')]:
        values=[before[key],after[key]];ax.plot(['Current v3','Larger v4'],values,'o-',color='#16806a',markersize=10)
        pad=max(abs(values[1]-values[0])*.5,.00005 if key=='score' else .05)
        ax.set_ylim(min(values)-pad,max(values)+pad*2)
        for i,v in enumerate(values):ax.annotate(f'{v:.7f}' if key=='score' else f'{v:.4f}',(i,v),xytext=(0,10),textcoords='offset points',ha='center')
        ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.margins(x=.3)
    fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
    text=f'''更大主模型实测：IEEE-Strong-v4

主模型从 11,388,974 参数扩为 43,855,662 参数，增加至 3.8507 倍。
宽度 256→512，注意力头 8→16，每头32维；10个图/SSM块和全天288时间点、全道路站点保持。
64个主模型输入特征不变，现有v3流量专家仍保留；本轮只考察增大主模型的价值。
通过复制通道并拆分输入权重继承已有训练结果，取消对称的小幅成对扰动让新增通道能够分别学习。
CPU和真实道路FP32输出继承检查通过；FP16由于矩阵计算次序不同，有小量初始舍入差异，各方向检查记录在expansion-parity.json。

训练与选择
三个重点方向各训练60次更新，20/40/60检查内部验证；训练当前v3 + (新大模型 - 冻结v2原始主模型)的流量误差。
部署采用同一表达式，混合系数从0、0.1、0.25、0.5、0.75、1中选；速度保持v3，其他任务不变。
每个方向必须在完整22天、前11天、后11天的汇总及遮挡等权流量RMSE均不恶化，且汇总改善至少0.01，才能采用。
训练期2030-06-29至12-13，内部验证2030-12-14至2031-01-04。
所有方向的选择在评估留出集前冻结。原有专家使用冻结训练特征，历史统计对训练日排除自己当天；专家训练误差为样本内误差。

相同55天本地留出集（6,636,705个缺失目标）
                    当前v3          更大v4候选
S_state             {before['score']:.9f}      {after['score']:.9f}
每车道流量RMSE      {before['flow_rmse']:.6f}         {after['flow_rmse']:.6f}
速度RMSE            {before['speed_rmse']:.6f}          {after['speed_rmse']:.6f}
流量单位为辆/小时/车道，速度单位为km/h。
{result['adopted']}/3 个方向采用更大主模型，{result['improved_directions']}/10 个方向的本地留出评分提高。
其余方向保留当前v3。

分方向（流量列为三种遮挡RMSE的等权平均）
{table[['panel','rmse_flow_per_lane_baseline','rmse_flow_per_lane_new','score_delta']].to_string(index=False)}

冻结选择
{pd.DataFrame(routes).to_string(index=False)}

资源
训练及内部验证共{monitor['seconds']/60:.2f}分钟；采样显存峰值{monitor['peak_vram_mib']/1024:.2f} GiB，包含批量探测阶段。
GPU功耗峰值{monitor['peak_power_w']:.2f}W，温度峰值{monitor['peak_temperature_c']:.1f}°C。功耗不是整机插座功耗。
批量由真实前向/反向和Adam状态探测；单批不足时通过梯度累积保留有效批量2，没有裁剪道路或缩短全天序列。
批量2若超过显存限制会回退批量1；探测结果保存在各方向capacity-plan.json。

交付
当前v3及其v2依赖共{len(preserved)}个已记录文件的SHA256核验未改变，完整保留旧最佳版本。
提交文件：{candidate.resolve()}
结构校验：{validation['status']}，{validation['rows']:,}行，SHA256 {validation['sha256']}。
速度、排队与ODME输出逐项核对一致，只有采用大模型的方向的流量可以改变。

限制
本轮是单种子、60次更新的有限扩容试验，不代表大模型已训练至收敛。
本轮没有等额追加训练的小模型对照，收益不能全部归因于参数量增加。
该本地留出集此前已用于多轮开发，当前结果属于探索性证据，不能保证Kaggle隐藏测试收益。
没有比较不可用的原Transformer，没有上传Kaggle。
'''
    (ROOT/'更大模型实测报告.txt').write_text(text,encoding='utf-8')
    save(ROOT/'delivery.json',dict(status='completed',baseline='IEEE-Strong-v3',report='更大模型实测报告.txt',
        submission=str(candidate.resolve()),validation=validation['status'],previous_best_preserved=True,
        parameters=43855662,adopted=result['adopted'],score=after['score'],flow_rmse=after['flow_rmse']))
    print(text,flush=True)


if __name__=='__main__':main()
