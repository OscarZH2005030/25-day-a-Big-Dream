import sys, numpy as np, pandas as pd
import faithful_lwr_scorer as F
S=F.S
panel=sys.argv[1]; method=sys.argv[2] if len(sys.argv)>2 else "Current_fusion"
pdir=F.REL/"corridors"/panel; params=S.network_parameters(pdir).set_index("link_id"); L=params.length_km
truth=F.load_truth(pdir,F.HOLD_LO,F.HOLD_HI,panel)
pred=pd.read_parquet(F.ROOT[panel]/"holdout"/panel/f"{method}_predictions.parquet")
hs=F.hybrid_state(truth)
for r in S.REGIMES:
    p=pred[pred.mask_regime==r]
    full,_=hs(panel,None,None,r,p)
    tr=truth[truth.regime==r].groupby(["timestamp","link_id"],as_index=False)[["speed_kmh","flow_vph"]].mean()
    grid=pd.date_range(full.timestamp.min(),full.timestamp.max(),freq="5min")
    def pv(df,c): return df.pivot(index="timestamp",columns="link_id",values=c).reindex(grid)
    cols=pv(full,"flow_vph").columns; Ls=L.reindex(cols).to_numpy(float)
    Nh=(pv(full,"flow_vph")/pv(full,"speed_kmh").clip(lower=1)*Ls)
    Nt=(pv(tr,"flow_vph").reindex(columns=cols)/pv(tr,"speed_kmh").reindex(columns=cols).clip(lower=1)*Ls)
    tg=pv(full,"is_target").fillna(0).astype(bool)
    Nobs=Nh.where(~tg)
    Ni=Nobs.interpolate(limit=12,limit_area="inside")   # linear in time between observed neighbours
    def E(N, mask_pairs=True):
        dN=N.shift(-1)-N; dT=Nt.shift(-1)-Nt
        touch=(tg|tg.shift(-1,fill_value=False))
        ok=dN.notna()&dT.notna()
        return (dN-dT)[ok].abs().sum().sum()/dT[ok].abs().sum().sum(), (dN-dT)[ok&touch].abs().sum().sum()/dT[ok].abs().sum().sum()
    # run lengths
    e_m=(Nh-Nt)[tg]; e_i=(Ni-Nt)[tg]
    print(f"\n{panel} {method} {r}: targets {int(tg.sum().sum()):,}  share of all cells {tg.sum().sum()/Nt.notna().sum().sum():.2f}")
    print(" median |e|/N  model %.4f  interp %.4f"%((e_m.abs()/Nt[tg]).stack().median(),(e_i.abs()/Nt[tg]).stack().median()))
    print(" mean |e| model %.3f interp %.3f ; corr(e_m,e_i) %.2f"%(e_m.abs().stack().mean(),e_i.abs().stack().mean(), np.corrcoef(e_m.stack().align(e_i.stack(),join='inner')[0],e_i.stack().align(e_m.stack(),join='inner')[0])[0,1]))
    em_,tot_=E(Nh); ei_,toti=E(Ni.where(tg,Nh))
    print(" E_LWR(all pairs) model %.3f | interp-at-targets %.3f"%(em_,ei_))
    for lam in (0.25,0.5,0.75):
        Nb=Nh.where(~tg,(1-lam)*Nh+lam*Ni)
        print("  blend lam=%.2f E=%.3f"%(lam,E(Nb)[0]))
    # max run length per cell approx: share of targets in runs of length 1 (isolated)
    iso=(tg&~tg.shift(1,fill_value=False)&~tg.shift(-1,fill_value=False)).sum().sum()
    print(" isolated single-cell targets share %.2f"%(iso/tg.sum().sum()))
