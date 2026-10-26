"""FD (fundamental-diagram) target-point diagnostic on T1-dynamic holdout predictions.

Applies the public FD formula to masked target cells of each method's holdout
predictions. Local diagnostic only: it is NOT the official Task 3 score
(S_LWR needs organizer boundary flows, unavailable publicly).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REL = Path(r"C:/Users/ROG/Desktop/IEEE Big data 2026/25-day-a-Big-Dream/data/kaggle_public/corridors")
RUNS = Path(r"C:/Users/ROG/Desktop/IEEE Big data 2026/25-day-a-Big-Dream-main/experiments/task1_t1_dynamic_20261021/runs")
BATCHES = {"D7_I10": RUNS / "oscar_handoff_i10_formal_20261022/batches/02_D7_I10",
           "D12_I405": RUNS / "oscar_handoff_i405_formal_20261022/batches/03_D12_I405"}
PANELS = {"D7_I10": ["D7_I10_E", "D7_I10_W"], "D12_I405": ["D12_I405_N", "D12_I405_S"]}
METHODS = ["T0", "T1", "A_dynamic", "LightGBM", "Current_fusion", "A_dynamic_fixed"]


def fd_score(speed, flow, p):
    lanes = p.lanes.to_numpy(float)
    q = flow / lanes
    k = flow / np.maximum(speed, 1.0) / lanes
    vf = p.free_speed_kmh.to_numpy(float); cap = p.capacity_vph.to_numpy(float) / lanes
    kc = p.critical_density.to_numpy(float) / lanes; kj = p.k_jam.to_numpy(float) / lanes
    q_fd = np.where(k <= kc, vf * k, cap / np.maximum(kj - kc, 1e-9) * np.maximum(kj - k, 0.0))
    return max(0.0, 1.0 - float(np.abs(q - q_fd).sum() / (np.abs(q).sum() + 1e-9)))


rows = []
for fam, b in BATCHES.items():
    for panel in PANELS[fam]:
        fd = pd.read_csv(REL / panel / "network/fd_parameters.csv").set_index("link_id")
        for m in METHODS:
            f = b / "holdout" / panel / f"{m}_predictions.parquet"
            if not f.exists():
                continue
            d = pd.read_parquet(f)
            d = d.join(fd, on="link_id")
            for reg, g in list(d.groupby("mask_regime")) + [("ALL", d)]:
                rows.append(dict(family=fam, panel=panel, method=m, regime=reg, n=len(g),
                                 S_FD=fd_score(g.speed_kmh.to_numpy(float), g.flow_vph.to_numpy(float), g)))
out = pd.DataFrame(rows)
out.to_csv(Path(__file__).with_name("fd_diagnostic_t1dynamic_4dir.csv"), index=False)
piv = out[out.regime != "ALL"].groupby(["panel", "method"]).S_FD.mean().unstack()
print(piv[METHODS].round(5).to_string()); print(piv[METHODS].mean().round(5))
