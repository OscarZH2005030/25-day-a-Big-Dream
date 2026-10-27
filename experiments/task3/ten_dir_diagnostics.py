"""Ten-direction T1-dynamic summary: S_state (5 families), FD target-point score, LWR proxy.
FD / LWR proxy are local diagnostics, NOT the official Task 3 score."""
from pathlib import Path
import numpy as np, pandas as pd
R = Path(r"C:/Users/ROG/Desktop/IEEE Big data 2026")
REL = R/"25-day-a-Big-Dream/data/kaggle_public/corridors"
T1 = R/"25-day-a-Big-Dream-main/experiments/task1_t1_dynamic_20261021"
RUNS = T1/"runs"; SUP = T1/"holdout_predictions_20261026"
SRC = {  # panel -> (holdout dir root, metrics csv)
 **{p: (SUP/"d7_i405", SUP/"d7_i405/holdout_metrics_by_panel_regime.csv") for p in ("D7_I405_N","D7_I405_S")},
 **{p: (SUP/"d12_i5", SUP/"d12_i5/holdout_metrics_by_panel_regime.csv") for p in ("D12_I5_N","D12_I5_S")},
 **{p: (SUP/"d7_i210", SUP/"d7_i210/holdout_metrics_by_panel_regime.csv") for p in ("D7_I210_E","D7_I210_W")},
 **{p: (RUNS/"oscar_handoff_i10_formal_20261022/batches/02_D7_I10", None) for p in ("D7_I10_E","D7_I10_W")},
 **{p: (RUNS/"oscar_handoff_i405_formal_20261022/batches/03_D12_I405", None) for p in ("D12_I405_N","D12_I405_S")}}
M = ["T1","A_dynamic_fixed","Current_fusion","LightGBM"]
out = Path(__file__).parent

# ---- S_state, 10 directions, 5 families (regime -> panel -> family equal weight)
mf = [SUP/f"{g}/holdout_metrics_by_panel_regime.csv" for g in ("d7_i405","d12_i5","d7_i210")] + \
     [RUNS/"oscar_handoff_i10_formal_20261022/batches/02_D7_I10/holdout_metrics_by_panel_regime.csv",
      RUNS/"oscar_handoff_i405_formal_20261022/batches/03_D12_I405/holdout_metrics_by_panel_regime.csv"]
d = pd.concat([pd.read_csv(f) for f in mf])
d = d[d.method.isin(["T0","T1","A_dynamic","LightGBM","Current_fusion","A_dynamic_fixed"])]
fam = d.groupby(["method","family","panel"]).S_state.mean().reset_index().groupby(["method","family"]).S_state.mean().unstack(0)
tot = fam.mean().rename("S_10dir_5family"); tot.to_csv(out/"ten_dir_S_state_total.csv"); fam.to_csv(out/"ten_dir_S_state_by_family.csv")
print("S_state by family\n", fam.round(5).to_string()); print("TOTAL\n", tot.round(6).to_string(), flush=True)

# ---- FD and LWR proxy
def fd(g):
    lanes=g.lanes.to_numpy(float); q=g.flow_vph.to_numpy(float)/lanes; k=g.flow_vph.to_numpy(float)/np.maximum(g.speed_kmh.to_numpy(float),1)/lanes
    vf=g.free_speed_kmh.to_numpy(float); cap=g.capacity_vph.to_numpy(float)/lanes; kc=g.critical_density.to_numpy(float)/lanes; kj=g.k_jam.to_numpy(float)/lanes
    qf=np.where(k<=kc, vf*k, cap/np.maximum(kj-kc,1e-9)*np.maximum(kj-k,0.0)); return max(0.0,1-float(np.abs(q-qf).sum()/(np.abs(q).sum()+1e-9)))
def prep(df):
    df=df.copy(); df["timestamp"]=pd.to_datetime(df.timestamp,utc=True)
    return df.groupby(["timestamp","link_id"],as_index=False)[["speed_kmh","flow_vph"]].mean()
def rho(pred,truth,length):
    m=pred.merge(truth,on=["timestamp","link_id"],suffixes=("","_t")).merge(length,on="link_id").sort_values(["link_id","timestamp"])
    m["N"]=m.flow_vph/np.maximum(m.speed_kmh,1)*m.length_km; m["Nt"]=m.flow_vph_t/np.maximum(m.speed_kmh_t,1)*m.length_km
    g=m.groupby("link_id"); ok=(g.timestamp.shift(-1)-m.timestamp)==pd.Timedelta(minutes=5)
    a=(g.N.shift(-1)-m.N)[ok]; b=(g.Nt.shift(-1)-m.Nt)[ok]; return float(np.abs(a-b).sum()/np.abs(b).sum()), int(ok.sum())
rows=[]
for panel,(root,_) in SRC.items():
    net=REL/panel/"network"; L=pd.read_csv(net/"links.csv",usecols=["link_id","length_km"]).drop_duplicates("link_id")
    fdp=pd.read_csv(net/"fd_parameters.csv").set_index("link_id")
    preds={m:pd.read_parquet(root/"holdout"/panel/f"{m}_predictions.parquet") for m in M}
    lo=min(pd.to_datetime(p.timestamp,utc=True).min() for p in preds.values()); hi=max(pd.to_datetime(p.timestamp,utc=True).max() for p in preds.values())
    T=[]
    for f in sorted((REL/panel/"train"/"mainline_states").glob("**/*.parquet")):
        x=pd.read_parquet(f,columns=["timestamp","station_id","link_id","speed_kmh","flow_vph"]); x["timestamp"]=pd.to_datetime(x.timestamp,utc=True)
        T.append(x[(x.timestamp>=lo)&(x.timestamp<=hi)])
    T=prep(pd.concat(T))
    for m,p in preds.items():
        for reg,g in p.groupby("mask_regime"):
            r,n=rho(prep(g),T,L); gg=g.join(fdp,on="link_id")
            rows.append(dict(panel=panel,method=m,regime=reg,n=len(g),pairs=n,S_FD=fd(gg),rho=r,S_LWR_proxy=max(0,1-r)))
    print(panel,"done",flush=True)
res=pd.DataFrame(rows); res.to_csv(out/"ten_dir_fd_lwr_proxy.csv",index=False)
pp=res.groupby(["panel","method"])[["S_FD","S_LWR_proxy"]].mean().unstack(1); print(pp.round(4).to_string())
print(res.groupby("method")[["S_FD","S_LWR_proxy"]].mean().round(5))
