"""Test temporal smoothing post-processing on one panel: effect on S_state and LWR-proxy rho."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import numpy as np, pandas as pd
from pathlib import Path
import importlib.util
spec = importlib.util.spec_from_file_location("lp", Path(__file__).with_name("lwr_proxy_holdout.py"))
src = open(spec.origin, encoding="utf-8").read().split("rows = []")[0]
exec(src)
panel, fam, method = (sys.argv[1], sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("D7_I10_E", "D7_I10", "A_dynamic_fixed")
b = B[fam]
L = pd.read_csv(REL/panel/"network/links.csv", usecols=["link_id","length_km"]).drop_duplicates("link_id")
p = pd.read_parquet(b/"holdout"/panel/f"{method}_predictions.parquet"); p["timestamp"] = pd.to_datetime(p.timestamp, utc=True)
T = truth_for(panel, None); T["timestamp"] = pd.to_datetime(T.timestamp, utc=True)
T = T[(T.timestamp>=p.timestamp.min())&(T.timestamp<=p.timestamp.max())].groupby(["timestamp","link_id"],as_index=False)[["speed_kmh","flow_vph"]].mean()
lanes = pd.read_csv(REL/panel/"network/fd_parameters.csv").groupby("link_id").lanes.mean()

def smooth(df, w, alpha):
    d = df.sort_values(["link_id","timestamp"]).copy()
    run = (d.groupby("link_id").timestamp.diff() != pd.Timedelta(minutes=5)).cumsum()
    for c in ("speed_kmh","flow_vph"):
        s = d.groupby(run)[c].transform(lambda x: x.rolling(w, center=True, min_periods=1).mean())
        d[c] = (1-alpha)*d[c] + alpha*s
    return d

def s_state(pred, truth, reg):
    m = pred.merge(truth, on=["timestamp","link_id"], suffixes=("","_t")); ln = m.link_id.map(lanes)
    rs = np.sqrt(((m.speed_kmh-m.speed_kmh_t)**2).mean()); rf = np.sqrt((((m.flow_vph-m.flow_vph_t)/ln)**2).mean())
    return .54*max(0,1-rs/25)+.46*max(0,1-rf/600)

res = []
for w, a in [(1,0),(3,1.0),(3,0.5),(5,1.0),(5,0.5),(7,0.5)]:
    rr, ss = [], []
    for reg, g in p.groupby("mask_regime"):
        q = prep(g); q = smooth(q, w, a) if w > 1 else q
        rho, n = proxy(q, T, L, reg); rr.append(rho); ss.append(s_state(q, T, reg))
    res.append(dict(panel=panel, method=method, window=w, alpha=a, rho=np.mean(rr), S_LWR_proxy=max(0,1-np.mean(rr)), S_state=np.mean(ss)))
    print(res[-1], flush=True)
pd.DataFrame(res).to_csv(Path(__file__).with_name(f"pp_smooth_test_{panel}_{method}.csv"), index=False)
