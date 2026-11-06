"""Post-processing screen on the faithful Task 3 scorer.

Each variant rewrites the submitted (speed, flow) cells of a prediction file using only
information a participant has: the predictions themselves and the observed (unmasked)
neighbours. Truth at target cells is never read. Variants are scored with the faithful
Task 3 scorer (S_physics) and a Task 1 proxy (S_state, link-level lanes).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import faithful_lwr_scorer as F

S = F.S


def smooth_predictions(panel, pred, truth, params, w, lam, domain):
    """Blend each submitted cell toward a centred moving average of the hybrid series.

    Hybrid series = observed values + the submitted values (per link, 5-minute grid).
    domain 'qv' smooths flow and speed; 'N' smooths flow and accumulation N=q/v*L and
    recovers speed from them (clipped to the range seen in the predictions).
    """
    out = []
    hs = F.hybrid_state(truth)
    L = params.length_km
    for r in S.REGIMES:
        p = pred[pred.mask_regime == r].copy()
        if p.empty:
            continue
        full, _ = hs(panel, None, None, r, p)
        grid = pd.date_range(full.timestamp.min(), full.timestamp.max(), freq="5min")
        piv_q = full.pivot(index="timestamp", columns="link_id", values="flow_vph").reindex(grid)
        piv_v = full.pivot(index="timestamp", columns="link_id", values="speed_kmh").reindex(grid)
        cols = piv_q.columns
        ti = grid.get_indexer(pd.to_datetime(p.timestamp, utc=True))
        li = cols.get_indexer(p.link_id.astype(str))
        ok = (ti >= 0) & (li >= 0)
        q0, v0 = p.flow_vph.to_numpy(float), p.speed_kmh.to_numpy(float)
        qs = piv_q.rolling(w, center=True, min_periods=1).mean().to_numpy()[ti.clip(0), li.clip(0)]
        if domain == "qv":
            vs = piv_v.rolling(w, center=True, min_periods=1).mean().to_numpy()[ti.clip(0), li.clip(0)]
        else:
            Ls = L.reindex(cols).to_numpy(float)
            piv_N = piv_q / piv_v.clip(lower=1.0) * Ls
            Ns = piv_N.rolling(w, center=True, min_periods=1).mean().to_numpy()[ti.clip(0), li.clip(0)]
            vs = qs * Ls[li.clip(0)] / np.maximum(Ns, 1e-6)
            vs = np.clip(vs, max(1.0, np.nanmin(v0)), np.nanmax(v0))
        q1 = np.where(ok, (1 - lam) * q0 + lam * qs, q0)
        v1 = np.where(ok, (1 - lam) * v0 + lam * vs, v0)
        p["flow_vph"], p["speed_kmh"] = q1, v1
        out.append(p)
    return pd.concat(out, ignore_index=True)


def s_state_proxy(pred, truth, lanes):
    m = pred.assign(timestamp=pd.to_datetime(pred.timestamp, utc=True)).merge(
        truth[F.KEYS + ["speed_kmh", "flow_vph"]], on=F.KEYS, suffixes=("", "_t"))
    ln = m.link_id.map(lanes).fillna(1.0).clip(lower=1.0)
    res = {}
    for r, g in m.groupby("mask_regime"):
        l = ln.loc[g.index]
        rs = np.sqrt(((g.speed_kmh - g.speed_kmh_t) ** 2).mean())
        rf = np.sqrt((((g.flow_vph - g.flow_vph_t) / l) ** 2).mean())
        res[r] = 0.54 * max(0, 1 - rs / 25) + 0.46 * max(0, 1 - rf / 600)
    return float(np.mean(list(res.values())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append", required=True)
    ap.add_argument("--method", default="Current_fusion")
    ap.add_argument("--variants", default="base;qv,3,1;qv,5,1;qv,9,1;qv,5,0.5;N,3,1;N,5,1")
    ap.add_argument("--output", type=Path, default=Path(__file__).parent / "postproc_scores.csv")
    a = ap.parse_args()
    rows = []
    for panel in a.panel:
        pdir = F.REL / "corridors" / panel
        params = S.network_parameters(pdir).set_index("link_id")
        env = {"params": params, "order": S.link_order(pdir, "train"), "ramps": S.load_ramp_flows(pdir, "train")}
        truth = F.load_truth(pdir, F.HOLD_LO, F.HOLD_HI, panel)
        pred = pd.read_parquet(F.ROOT[panel] / "holdout" / panel / f"{a.method}_predictions.parquet")
        for v in a.variants.split(";"):
            if v == "base":
                p2, name = pred, "base"
            else:
                dom, w, lam = v.split(","); w, lam = int(w), float(lam)
                p2 = smooth_predictions(panel, pred, truth, params, w, lam, dom); name = v
            res = F.score_predictions(panel, p2, truth, env)
            rows.append(dict(panel=panel, variant=name, S_FD=res.S_FD.mean(), S_LWR=res.S_LWR.mean(),
                             S_physics=res.S_physics.mean(), S_LWR_R1=res.S_LWR[res.regime == "R1"].iloc[0],
                             S_LWR_R3=res.S_LWR[res.regime == "R3"].iloc[0],
                             S_state=s_state_proxy(p2, truth, params.lanes)))
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(a.output, index=False)
    d = pd.DataFrame(rows)
    base = d[d.variant == "base"].set_index("panel")
    d["dS_phys"] = d.S_physics.values - base.S_physics.reindex(d.panel).values
    d["dS_state"] = d.S_state.values - base.S_state.reindex(d.panel).values
    print(d.groupby("variant")[["S_LWR", "S_physics", "S_state", "dS_phys", "dS_state"]].mean().round(4))


if __name__ == "__main__":
    main()
