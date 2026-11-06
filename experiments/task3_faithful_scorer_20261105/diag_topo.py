import sys, numpy as np, pandas as pd
import faithful_lwr_scorer as F
S=F.S
panel=sys.argv[1]
pdir=F.REL/"corridors"/panel; params=S.network_parameters(pdir).set_index("link_id"); L=params.length_km
truth=F.load_truth(pdir,F.HOLD_LO,F.HOLD_HI,panel)
ramps=S.load_ramp_flows(pdir,"train")
r="R1"
tr=truth[truth.regime==r].groupby(["timestamp","link_id"],as_index=False)[["speed_kmh","flow_vph"]].mean()
tr["length_km"]=tr.link_id.map(L); tr=tr.dropna()
tr["N"]=tr.flow_vph/tr.speed_kmh.clip(lower=1)*tr.length_km
bf=S.topology_boundary_flows(tr,pdir)               # topology flux from TRUTH flows
t=tr.merge(bf,on=["timestamp","link_id"],how="left").merge(ramps,on=["timestamp","link_id"],how="left").sort_values(["link_id","timestamp"])
for c in ("on_ramp_flow_vph","off_ramp_flow_vph"): t[c]=t[c].fillna(0.0)
g=t.groupby("link_id"); t["dN"]=g.N.shift(-1)-t.N; t["dt"]=(g.timestamp.shift(-1)-t.timestamp).dt.total_seconds()/3600
t=t[(t.dt>0)&(t.dt<=5.1/60)]
rhs=t.dt*(t.inflow_vph+t.on_ramp_flow_vph-t.outflow_vph-t.off_ramp_flow_vph)
print(panel,"truth-topology flux vs truth dN: E =",float((t.dN-rhs).abs().sum()/t.dN.abs().sum()), " sum|dN|/sum N=",float(t.dN.abs().sum()/t.N.sum()))
topo=pd.read_csv(pdir/"network/lwr_mainline_topology.csv",dtype=str); print(topo.head(3).to_string()); print(len(topo),tr.link_id.nunique())
