import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import json,hashlib,traceback,time
from pathlib import Path
import numpy as np
from probe_paired import ROOT,OUT,read,save,guard,progress,fit
from paired_data import build
def main():
    started=time.time();gates=read(ROOT/'training-gates.json');assert gates['passed_roads']>0 and read(ROOT/'early-verification.json')['passed'];before=guard('full-before');data,audit=build('full',progress);save('full-training-data-audit.json',audit);model=fit(data,full=True);q=model.predict(data['X'][:min(1024,len(data['y']))],num_threads=4);assert model.num_feature()==557 and model.current_iteration()==600 and model.num_trees()==3600;assert np.isfinite(q).all() and np.allclose(q.sum(1),1,atol=1e-5);assert guard('full-after')==before
    names=['conditional-first-full.txt','timing_core.py','gated_core.py','paired_data.py'];hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names};save('full-training-verification.json',dict(passed=True,trained_only_pending_development=True,rows=len(data['y']),features=557,boosting_rounds=600,trees=3600,hashes=hashes,early_gate_sha256=hashlib.sha256((ROOT/'training-gates.json').read_bytes()).hexdigest(),paired_index_sha256=hashlib.sha256((OUT/'task2-round39-paired-phase-calendar/paired-targets.json').read_bytes()).hexdigest(),Task1_92files_unchanged=True,probability_simplex_passed=True,feature_causality_16groups_from_early_same_source=True,official_S_queue=None,pending='new-directory comparison against frozen R38, December routes, January global-only calibration, frozen two-population reports, independent inference/future acceptance'))
    lines=['Task2 第四十轮：原conditional目标保留，追加来源中立三相位配对视角','尚未开发/校准/部署，没有新的官方或二月分数。','原始X/y逐行相同，原相对权重保留；各新增物理firstpositive观测组总质量1，原条件窗口总质量1，再整体归一。重复视角不等于独立事件，新的损失单位不是无偏宏IoU。',json.dumps(audit,ensure_ascii=False),'早期原/配对双人口门槛：',json.dumps(gates,ensure_ascii=False),'头为固定600轮/3600树、557维，Task1原92hash保护；未来metadata只用于标签/抽样。']
    (ROOT/'第四十轮训练说明.txt').write_text('\n'.join(lines)+'\n',encoding='utf8');progress('full_head_trained_pending_development',rows=len(data['y']),seconds=time.time()-started)
if __name__=='__main__':
    try:main()
    except Exception:save('full-training-failure.json',dict(traceback=traceback.format_exc()));progress('failed');raise
