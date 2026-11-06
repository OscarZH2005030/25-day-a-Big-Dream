import sys, numpy as np, pandas as pd
import faithful_lwr_scorer as F
from postproc_test import s_state_proxy
S=F.S
rows=[]
for panel in sys.argv[1:]:
    pdir=F.REL/"corridors"/panel; params=S.network_parameters(pdir).set_index("link_id")
    env={"params":params,"order":S.link_order(pdir,"train"),"ramps":S.load_ramp_flows(pdir,"train")}
    truth=F.load_truth(pdir,F.HOLD_LO,F.HOLD_HI,panel)
    M=["T1","A_dynamic_fixed","Current_fusion","LightGBM"]
    P={m:pd.read_parquet(F.ROOT[panel]/"holdout"/panel/f"{m}_predictions.parquet").set_index(F.KEYS+["mask_regime"]).sort_index() for m in M}
    base=P["Current_fusion"]
    def mix(w):
        x=base.copy()
        for c in ("speed_kmh","flow_vph"):
            x[c]=sum(w[m]*P[m].reindex(x.index)[c].to_numpy() for m in w)/sum(w.values())
        return x.reset_index()
    V={"Current_fusion":{"Current_fusion":1},"mean4":{m:1 for m in M},"T1+A+LGBM":{"T1":1,"A_dynamic_fixed":1,"LightGBM":1},
       "fusion+LGBM":{"Current_fusion":1,"LightGBM":1},"A+LGBM":{"A_dynamic_fixed":1,"LightGBM":1}}
    for name,w in V.items():
        p=mix(w); res=F.score_predictions(panel,p,truth,env)
        rows.append(dict(panel=panel,variant=name,S_LWR=res.S_LWR.mean(),S_physics=res.S_physics.mean(),S_LWR_R3=res.S_LWR[res.regime=="R3"].iloc[0],S_state=s_state_proxy(p,truth,params.lanes)))
        print(rows[-1],flush=True)
d=pd.DataFrame(rows); d.to_csv("ens_scores.csv",index=False); print(d.groupby("variant")[["S_LWR","S_physics","S_state"]].mean().round(4))
