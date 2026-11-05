from pathlib import Path
import argparse, hashlib, json
import pandas as pd
parser=argparse.ArgumentParser(description='Compare fixed OD reweighting rules on released weak-prior proxies.')
parser.add_argument('--results-root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args=parser.parse_args()
root=args.results_root
out=args.output
out.mkdir(parents=True, exist_ok=True)
inputs=[root/f'task4_pairing_prior_{tag}_weights/by_panel.csv' for tag in ('span','disagreement')]
pairs=[]
for path in inputs:
    data=pd.read_csv(path)
    keys=['panel','view']
    baseline=data.query("method=='uniform'").drop(columns='method')
    pair=data.query("method!='uniform'").merge(baseline,on=keys,validate='many_to_one',suffixes=('','_uniform'))
    pair['l1_delta']=pair.relative_l1-pair.relative_l1_uniform
    pair['squared_delta']=pair.relative_squared_error-pair.relative_squared_error_uniform
    pairs.append(pair[keys+['method','l1_delta','squared_delta']])
data=pd.concat(pairs,ignore_index=True)
assert len(data)==80
data['family']=data.panel.str.rsplit('_',n=1).str[0]
data['l1_win']=data.l1_delta< -1e-10
data['both_win']=data.l1_win & (data.squared_delta< -1e-10)
data.to_csv(out/'by_panel.csv',index=False)
data.groupby(['view','method'])[['l1_delta','squared_delta','l1_win','both_win']].mean().to_csv(out/'summary.csv')
leave=[]
for family in sorted(data.family.unique()):
    subset=data[data.family!=family]
    means=subset.groupby(['view','method'])[['l1_delta','squared_delta']].mean().reset_index()
    leave.append(means.assign(excluded_family=family))
pd.concat(leave,ignore_index=True).to_csv(out/'family_exclusion.csv',index=False)
(out/'protocol.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    input_sha256={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
    scope='Descriptive road pairs and family sensitivity on exposed weak-prior targets. No tuning, sampling or hidden labels.'),indent=2),encoding='utf-8')
(out/'failures.json').write_text('[]\n',encoding='utf-8')
