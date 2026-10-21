"""Finish inference only after all GPU training ends; no Kaggle upload."""
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parent;workspace=root.parent.parent
while not (root/'specialists-complete.json').exists():time.sleep(2)
with (root/'final-inference-verification.log').open('w',encoding='utf-8') as log:
    subprocess.run([sys.executable,'-X','utf8',str(root/'verify_specialist_inference.py')],
        stdout=log,stderr=subprocess.STDOUT,check=True)
out=root/'candidate-submission-experts'
if not (out/'submission-validation.json').exists():
    with (root/'submission-inference.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-X','utf8',str(root/'specialist_inference.py'),
            '--release',str(workspace/'work/trafficflowbench-data/kaggle_public')],
            stdout=log,stderr=subprocess.STDOUT,check=True)
verification=json.loads((out/'submission-validation.json').read_text())
parity=json.loads((root/'specialist-inference-parity.json').read_text())
assert verification['status']=='passed' and verification['rows']==6980503
assert parity['passed'] and len(parity['checks'])==10
import pandas as pd
count=0;minimum={'speed_kmh':float('inf'),'flow_vph':float('inf')}
maximum={'speed_kmh':0.,'flow_vph':0.}
for path in (out/'parts').glob('state-*.csv'):
    for frame in pd.read_csv(path,usecols=list(minimum),chunksize=250000):
        count+=len(frame)
        for field in minimum:
            minimum[field]=min(minimum[field],float(frame[field].min()))
            maximum[field]=max(maximum[field],float(frame[field].max()))
scope=json.loads((out/'candidate-scope.json').read_text())
assert count==scope['new_state_rows']
range_report=dict(rows=count,minimum=minimum,maximum=maximum,
    note='Descriptive prediction ranges only; no unvalidated upper clipping is applied.')
(out/'prediction-ranges.json').write_text(json.dumps(range_report,indent=2))
summary=json.loads((root/'specialists/summary.json').read_text())
appendix=f'''\n候选提交已生成并完成校验\n\ncandidate-submission-experts/submission.csv\n总行数：{verification['rows']:,}；Task 1 state 行数：{count:,}。\n精确匹配官方模板顺序，无缺失、重复 ID、非有限或负数输出。\nSHA256：{verification['sha256']}\nTask 1 使用本轮十方向模型；Task 2/4 沿用原基线；Task 3 由 state 预测参与官方评价。\n未向 Kaggle 上传，也没有主办方隐藏标签得分。\n十个方向原始输入推理均与保存的本地留出预测核对通过。\n本地分数来自前述同协议留出，不是这份 validation/private 提交的隐藏评分。\n'''
report=root/'最终模型结果.txt'
text=report.read_text(encoding='utf-8')
if '候选提交已生成并完成校验' not in text:report.write_text(text+appendix,encoding='utf-8')
deliverables=dict(status='completed',model_manifest='final-model.json',
    report='最终模型结果.txt',metrics='specialists/summary.json',chart='specialists/final-chart.png',
    comparisons='specialists/comparison-with-transformer.csv',
    submission='candidate-submission-experts/submission.csv',
    submission_validation='candidate-submission-experts/submission-validation.json',
    inference_verification='specialist-inference-parity.json',
    state_score=summary['score'],transformer_score=summary['transformer_score'],
    previous_combo_score=summary['existing_hybrid_score'],
    submission_rows=verification['rows'],sha256=verification['sha256'],
    completed_at=dt.datetime.now().astimezone().isoformat())
(root/'deliverables.json').write_text(json.dumps(deliverables,indent=2,ensure_ascii=False),encoding='utf-8')
(root/'delivery-complete.json').write_text(json.dumps(dict(status='completed',
    completed_at=deliverables['completed_at']),indent=2))
print(json.dumps(deliverables,ensure_ascii=False),flush=True)
