from pathlib import Path
import pandas as pd,json
O=Path('outputs/robust_v9');tr=pd.read_csv(O/'multi_error_trials.csv');cv=pd.read_csv(O/'actual_train_count_cv.csv').groupby(['panel','seed','candidate']).NMAE.mean();cal=tr[tr.phase=='calibration'].groupby(['panel','candidate'])[['S_od','S_dev','S_attr','synthetic_S_ODME']].mean();sc=tr[tr.phase=='calibration'].groupby(['panel','scenario','candidate']).synthetic_S_ODME.mean();policy={};checks=[]
for p in sorted(tr.panel.unique()):
 vals=cv.loc[p].unstack();baseline=vals.official;score=cal.loc[p];eligible=[]
 for n in ['official','mix25','mix50','mix75','scaled_kl']:
  imp=float(1-vals[n].mean()/baseline.mean());worst=float((vals[n]/baseline).max());g=sc.loc[p].unstack();scdiff=float((g[n]-g.official).min());ok=(n=='official') or (imp>=.01 and worst<=1.03 and score.loc[n,'S_dev']>=score.loc['official','S_dev'] and score.loc[n,'synthetic_S_ODME']>=score.loc['official','synthetic_S_ODME'] and scdiff>=-.01)
  checks.append({'panel':p,'candidate':n,'real_CV_relative_improvement':imp,'worst_seed_ratio':worst,'cal_Sdev_delta':float(score.loc[n,'S_dev']-score.loc['official','S_dev']),'worst_cal_error_family_total_delta':scdiff,'eligible':bool(ok)})
  if ok:eligible.append(n)
 pick=str(score.loc[eligible].synthetic_S_ODME.idxmax());policy[p]=pick
pd.DataFrame(checks).to_csv(O/'conservative_policy_checks.csv',index=False);(O/'conservative_policy.json').write_text(json.dumps({'policy':policy,'selection':'calibration path metrics + 3-seed real count CV guard; fresh seeds 20261106,20261107 reserved for audit','data_reuse':'same released train reused; not an independent path test'},indent=2),encoding='utf-8');print(policy)
