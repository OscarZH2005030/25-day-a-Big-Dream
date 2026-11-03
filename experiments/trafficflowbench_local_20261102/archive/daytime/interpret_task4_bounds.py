"""Interpret feasible endpoints in zone intervals, never fabricate missing km."""
import common as d
import numpy as np,pandas as pd
O=d.OUT/'task4_demand_bounds'
f=pd.read_csv(O/'bounds.csv');rows=[]
for r in f.to_dict('records'):
 if not r['min_success'] or not r['max_success']:continue
 panel,split=r['panel'],r['split'];paths=pd.read_csv(d.DATA/'corridors'/panel/'network/path_set.csv',dtype=str);oi=paths.origin_zone.str.extract(r'^Z(\d+)$',expand=False);di=paths.destination_zone.str.extract(r'^Z(\d+)$',expand=False);assert oi.notna().all() and di.notna().all();hops=di.astype(int).to_numpy()-oi.astype(int).to_numpy();assert (hops>0).all();witness=d.WORK/'task4_demand_bounds'/panel/split/'diagnostic_extremes_NOT_A_SUBMISSION.parquet';v=pd.read_parquet(witness).set_index('path_id').reindex(paths.path_id);low=v.min_total_extreme.to_numpy();high=v.max_total_extreme.to_numpy();lo_work=float(hops@low);hi_work=float(hops@high);rel=abs(lo_work-hi_work)/max(abs(lo_work),1e-9);assert rel<1e-8;rows.append(dict(panel=panel,split=split,min_total=r['min_total'],max_total=r['max_total'],total_ratio=r['max_min_ratio'],min_total_mean_zone_intervals=lo_work/low.sum(),max_total_mean_zone_intervals=hi_work/high.sum(),identical_zone_interval_work_relative_error=rel,scope='Zone intervals only, not kilometers; both endpoints fit released counts and remain nonnegative.'))
pd.DataFrame(rows).to_csv(O/'zone_span_interpretation.csv',index=False)
d.save(O/'summary.json',dict(successful_cases=len(rows),cases=len(f),min_total_ratio=float(f.max_min_ratio.min()),max_total_ratio=float(f.max_min_ratio.max()),maximum_count_NMAE=float(f[['min_count_NMAE','max_count_NMAE']].max().max()),missing_km_policy='Leave mean path km null when any path metadata length is missing; use validated positive zone index differences for interval-span interpretation.',source_sha256=d.sha(__file__)))
print(pd.DataFrame(rows).head(3).to_string(index=False))
