"""FD target-point diagnostic for Strong v4 submission.csv on public validation (all 10 panels).
Local diagnostic only, NOT the official Task 3 score."""
from pathlib import Path
import numpy as np, pandas as pd
R = Path(r"C:/Users/ROG/Desktop/IEEE Big data 2026")
KEY = R / "25-day-a-Big-Dream/data/kaggle_public/submission_key.csv"
SUB = R / "25-day-a-Big-Dream-main/experiments/task1_strong_v4/downloaded/IEEE-Best-v4-20261020/submission.csv"
REL = R / "25-day-a-Big-Dream/data/kaggle_public/corridors"
key = pd.read_csv(KEY, usecols=["submission_id","task","split","panel","link_id","mask_regime"])
key = key[(key.task == "state") & (key.split == "validation")]
sub = pd.read_csv(SUB, usecols=["submission_id","speed_kmh","flow_vph"])
d = key.merge(sub, on="submission_id", how="left")
assert d.speed_kmh.notna().all() and np.isfinite(d.flow_vph).all()
def fd(g, p):
    g = g.join(p, on="link_id"); lanes = g.lanes.to_numpy(float)
    q = g.flow_vph.to_numpy(float)/lanes; k = g.flow_vph.to_numpy(float)/np.maximum(g.speed_kmh.to_numpy(float),1.0)/lanes
    vf=g.free_speed_kmh.to_numpy(float); cap=g.capacity_vph.to_numpy(float)/lanes
    kc=g.critical_density.to_numpy(float)/lanes; kj=g.k_jam.to_numpy(float)/lanes
    qfd=np.where(k<=kc, vf*k, cap/np.maximum(kj-kc,1e-9)*np.maximum(kj-k,0.0))
    return max(0.0, 1-float(np.abs(q-qfd).sum()/(np.abs(q).sum()+1e-9)))
rows=[]
for panel,g in d.groupby("panel"):
    p = pd.read_csv(REL/panel/"network/fd_parameters.csv").set_index("link_id")
    for reg,h in g.groupby("mask_regime"):
        rows.append(dict(panel=panel,regime=reg,n=len(h),S_FD=fd(h,p)))
out=pd.DataFrame(rows); out.to_csv(Path(__file__).with_name("fd_diagnostic_strong_v4_10dir.csv"),index=False)
t=out.groupby("panel").S_FD.mean(); print(out.pivot(index="panel",columns="regime",values="S_FD").round(5)); print(t.round(5)); print("mean",t.mean().round(5), "rows",len(d))
