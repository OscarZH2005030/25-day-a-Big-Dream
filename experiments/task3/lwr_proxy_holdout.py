"""LWR-proxy on T1-dynamic holdout predictions (target-target adjacent 5-min pairs).
Per the official spec the organizer boundary flux is projected on released observations, so the
truth state satisfies conservation exactly: residual ~= dN_pred - dN_true, rhs ~= dN_true.
proxy S_LWR = max(0, 1 - sum|dN_pred - dN_true| / sum|dN_true|).  Local proxy, NOT official."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path(r"C:/Users/ROG/Desktop/IEEE Big data 2026")
REL = R / "25-day-a-Big-Dream/data/kaggle_public/corridors"
RUNS = R / "25-day-a-Big-Dream-main/experiments/task1_t1_dynamic_20261021/runs"
B = {"D7_I10": RUNS/"oscar_handoff_i10_formal_20261022/batches/02_D7_I10", "D12_I405": RUNS/"oscar_handoff_i405_formal_20261022/batches/03_D12_I405"}
P = {"D7_I10": ["D7_I10_E","D7_I10_W"], "D12_I405": ["D12_I405_N","D12_I405_S"]}
M = ["T1","A_dynamic_fixed","Current_fusion","LightGBM"]

def truth_for(panel, keys):
    parts = []
    for f in sorted((REL/panel/"train"/"mainline_states").glob("**/*.parquet")):
        d = pd.read_parquet(f, columns=["timestamp","station_id","link_id","speed_kmh","flow_vph"])
        d = d[d.timestamp.isin(keys)] if False else d
        parts.append(d)
    return pd.concat(parts)

def prep(df):
    df = df.copy(); df["timestamp"] = pd.to_datetime(df.timestamp, utc=True)
    return df.groupby(["timestamp","link_id"], as_index=False)[["speed_kmh","flow_vph"]].mean()

def proxy(pred, truth, length, reg):
    m = pred.merge(truth, on=["timestamp","link_id"], suffixes=("","_t")).merge(length, on="link_id")
    m = m.sort_values(["link_id","timestamp"])
    m["N"] = m.flow_vph/np.maximum(m.speed_kmh,1)*m.length_km; m["Nt"] = m.flow_vph_t/np.maximum(m.speed_kmh_t,1)*m.length_km
    g = m.groupby("link_id")
    ok = (g.timestamp.shift(-1)-m.timestamp)==pd.Timedelta(minutes=5)
    dN = (g.N.shift(-1)-m.N)[ok]; dNt = (g.Nt.shift(-1)-m.Nt)[ok]
    return float(np.abs(dN-dNt).sum()/np.abs(dNt).sum()), len(dN)

rows = []
for fam,b in B.items():
    for panel in P[fam]:
        L = pd.read_csv(REL/panel/"network/links.csv", usecols=["link_id","length_km"]).drop_duplicates("link_id")
        T = None
        for m in M:
            p = pd.read_parquet(b/"holdout"/panel/f"{m}_predictions.parquet")
            p["timestamp"] = pd.to_datetime(p.timestamp, utc=True)
            if T is None:
                lo, hi = p.timestamp.min(), p.timestamp.max()
                T = truth_for(panel, None); T["timestamp"] = pd.to_datetime(T.timestamp, utc=True)
                T = T[(T.timestamp>=lo)&(T.timestamp<=hi)]; T = T.groupby(["timestamp","link_id"],as_index=False)[["speed_kmh","flow_vph"]].mean()
            for reg,g in p.groupby("mask_regime"):
                rho,n = proxy(prep(g), T, L, reg)
                rows.append(dict(panel=panel,method=m,regime=reg,pairs=n,rho=rho,S_LWR_proxy=max(0,1-rho)))
        print(panel,"done",flush=True)
out = pd.DataFrame(rows); out.to_csv(Path(__file__).with_name("lwr_proxy_holdout_4dir.csv"),index=False)
print(out.groupby(["panel","method"])[["rho","S_LWR_proxy"]].mean().round(3).to_string())
