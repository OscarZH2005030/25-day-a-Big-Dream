"""Faithful local Task 3 scorer on a train-split holdout.

The leaderboard checks conservation against organizer boundary flows that are never
published, so the public evaluator cannot be run as is. On the train split the
observation layer is complete, so boundary flows can be derived from the truth: the
truth then satisfies the conservation identity exactly (rhs == dN_true).

Everything else is the official code: this script imports score_task3 and calls its
network_parameters, load_ramp_flows, link_order, derive_physics and score_panel, so
the FD score, the observable-transition filter, the ramp-validity modes and the
E_LWR formula are the organizer's. The only substitutions are
  * the hybrid grid: truth on every eligible cell, the submitted value on the cells a
    prediction file covers (state_for_regime, patched to read a holdout window), and
  * the boundary flux, built from the truth.
The organizer's flux carries measurement noise, so a perfect answer scores 0.96 there
and about 1.0 here. Compare methods with each other, not with the leaderboard.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

OFFICIAL = Path(r"C:/Users/ROG/Desktop/IEEE Big data 2026/trafficflowbench-public")
REL = Path(r"C:/Users/ROG/Desktop/IEEE Big data 2026/25-day-a-Big-Dream/data/kaggle_public")
R = Path(r"C:/Users/ROG/Desktop/IEEE Big data 2026")
sys.path.insert(0, str(OFFICIAL / "src"))
import task3.score_task3 as S  # noqa: E402
from task1.baseline_task1_historical_mean import regime_of_dates  # noqa: E402

CFG = json.loads((OFFICIAL / "config" / "task3_lwr_modes.json").read_text(encoding="utf-8"))
HOLD_LO, HOLD_HI = "2031-01-05", "2031-02-28"
T1 = R / "25-day-a-Big-Dream-main/experiments/task1_t1_dynamic_20261021"
RUNS, SUP = T1 / "runs", T1 / "holdout_predictions_20261026"
ROOT = {
    **{p: SUP / "d7_i405" for p in ("D7_I405_N", "D7_I405_S")},
    **{p: SUP / "d12_i5" for p in ("D12_I5_N", "D12_I5_S")},
    **{p: SUP / "d7_i210" for p in ("D7_I210_E", "D7_I210_W")},
    **{p: RUNS / "oscar_handoff_i10_formal_20261022/batches/02_D7_I10" for p in ("D7_I10_E", "D7_I10_W")},
    **{p: RUNS / "oscar_handoff_i405_formal_20261022/batches/03_D12_I405" for p in ("D12_I405_N", "D12_I405_S")},
}
KEYS = ["timestamp", "station_id", "link_id"]


def load_truth(panel_dir: Path, lo: str, hi: str, panel: str) -> pd.DataFrame:
    """Eligible train cells for the dates in [lo, hi], tagged with their regime."""
    parts = []
    for f in sorted((panel_dir / "train" / "mainline_states").glob("**/*.parquet")):
        d = f.stem[-10:].replace("_", "-")
        if not (lo <= d <= hi):
            continue
        x = pd.read_parquet(f, columns=["date", "timestamp", "station_id", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"])
        x = x[x.is_score_eligible.astype(bool)].copy()
        x["regime"] = regime_of_dates(panel, x.date.astype(str))
        x["timestamp"] = pd.to_datetime(x.timestamp, utc=True)
        x["station_id"] = x.station_id.astype(str); x["link_id"] = x.link_id.astype(str)
        parts.append(x.drop(columns=["date", "is_score_eligible"]))
    return pd.concat(parts, ignore_index=True)


def hybrid_state(truth: pd.DataFrame):
    """state_for_regime replacement: truth, overwritten by the submitted target cells."""
    def f(panel, panel_dir, split, regime, state_panel):
        sr = state_panel[state_panel.mask_regime == regime][KEYS + ["speed_kmh", "flow_vph"]].copy()
        sr["timestamp"] = pd.to_datetime(sr.timestamp, utc=True)
        sr["station_id"] = sr.station_id.astype(str); sr["link_id"] = sr.link_id.astype(str)
        base = truth[truth.regime == regime][KEYS + ["speed_kmh", "flow_vph"]].rename(
            columns={"speed_kmh": "true_speed", "flow_vph": "true_flow"})
        sr["is_target"] = True
        base = base.merge(sr, on=KEYS, how="left", validate="one_to_one")
        it = base.is_target.astype("boolean").fillna(False).to_numpy(dtype=bool)
        missing = int(len(sr) - it.sum())
        base["speed_kmh"] = np.where(it, base.speed_kmh, base.true_speed)
        base["flow_vph"] = np.where(it, base.flow_vph, base.true_flow)
        base["is_target"] = it.astype(float)
        full = base[["timestamp", "link_id", "speed_kmh", "flow_vph", "is_target"]]
        return full.groupby(["timestamp", "link_id"], as_index=False).mean(numeric_only=True), missing
    return f


def truth_boundary(truth_r: pd.DataFrame, params: pd.DataFrame, ramps: pd.DataFrame) -> pd.DataFrame:
    """Boundary flows that make the truth satisfy rhs == dN exactly (outflow 0, inflow absorbs it)."""
    t = truth_r.groupby(["timestamp", "link_id"], as_index=False)[["speed_kmh", "flow_vph"]].mean()
    t = t.merge(params.reset_index(), on="link_id", how="left")
    t = t.dropna(subset=["speed_kmh", "flow_vph", "length_km", "capacity_vph", "critical_density", "k_jam"])
    t["N"] = t.flow_vph / t.speed_kmh.clip(lower=1.0) * t.length_km
    t = t.sort_values(["link_id", "timestamp"])
    g = t.groupby("link_id")
    t["dN"] = g.N.shift(-1) - t.N
    t["dt"] = (g.timestamp.shift(-1) - t.timestamp).dt.total_seconds() / 3600.0
    if not ramps.empty:
        t = t.merge(ramps[["timestamp", "link_id", "on_ramp_flow_vph", "off_ramp_flow_vph"]], on=["timestamp", "link_id"], how="left")
    for c in ("on_ramp_flow_vph", "off_ramp_flow_vph"):
        if c not in t:
            t[c] = 0.0
        t[c] = t[c].fillna(0.0)
    t["outflow_vph"] = 0.0
    t["inflow_vph"] = (t.dN / t.dt.where(t.dt > 0)) - t.on_ramp_flow_vph + t.off_ramp_flow_vph
    return t[["timestamp", "link_id", "inflow_vph", "outflow_vph"]].dropna()


def score_predictions(panel: str, pred: pd.DataFrame, truth: pd.DataFrame, env: dict) -> pd.DataFrame:
    S.state_for_regime = hybrid_state(truth)
    rows = []
    for r in S.REGIMES:
        flux = truth_boundary(truth[truth.regime == r], env["params"], env["ramps"])
        f, missing = S.derive_physics(panel, REL, "train", pred, r, env["order"], env["params"], env["ramps"], flux)
        res = S.score_panel(panel, f, REL, CFG, r)
        res["missing_targets"] = missing
        rows.append(res)
    return pd.DataFrame(rows)


def naive_predictions(panel_dir: Path, panel: str, keys: pd.DataFrame) -> pd.DataFrame:
    """Per (link, time-of-day) mean of the train days before the holdout, applied to the target keys."""
    parts = []
    for f in sorted((panel_dir / "train" / "mainline_states").glob("**/*.parquet")):
        if f.stem[-10:].replace("_", "-") >= HOLD_LO:
            continue
        x = pd.read_parquet(f, columns=["timestamp", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"])
        parts.append(x[x.is_score_eligible.astype(bool)])
    h = pd.concat(parts, ignore_index=True)
    h["tod"] = pd.to_datetime(h.timestamp, utc=True).dt.strftime("%H:%M")
    h["link_id"] = h.link_id.astype(str)
    m = h.groupby(["link_id", "tod"], as_index=False)[["speed_kmh", "flow_vph"]].mean()
    k = keys.copy()
    k["tod"] = pd.to_datetime(k.timestamp, utc=True).dt.strftime("%H:%M"); k["link_id"] = k.link_id.astype(str)
    k = k.drop(columns=["speed_kmh", "flow_vph"]).merge(m, on=["link_id", "tod"], how="left")
    k["speed_kmh"] = k.speed_kmh.fillna(h.speed_kmh.mean()); k["flow_vph"] = k.flow_vph.fillna(h.flow_vph.mean())
    return k.drop(columns="tod")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append", required=True)
    ap.add_argument("--methods", nargs="*", default=["T1", "A_dynamic_fixed", "Current_fusion", "LightGBM"])
    ap.add_argument("--controls", action="store_true", help="also score truth-as-prediction and a time-of-day mean")
    ap.add_argument("--output", type=Path, default=Path(__file__).parent / "faithful_lwr_scores.csv")
    a = ap.parse_args()
    out = []
    for panel in a.panel:
        pdir = REL / "corridors" / panel
        env = {"params": S.network_parameters(pdir).set_index("link_id"), "order": S.link_order(pdir, "train"),
               "ramps": S.load_ramp_flows(pdir, "train")}
        truth = load_truth(pdir, HOLD_LO, HOLD_HI, panel)
        print(f"[{panel}] truth rows {len(truth):,}", flush=True)
        preds = {m: pd.read_parquet(ROOT[panel] / "holdout" / panel / f"{m}_predictions.parquet") for m in a.methods}
        if a.controls:
            ref = next(iter(preds.values()))
            t = truth.assign(timestamp=truth.timestamp.dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
            tk = ref[["panel", "timestamp", "station_id", "link_id", "mask_regime"]].merge(
                t[["timestamp", "station_id", "link_id", "speed_kmh", "flow_vph"]], on=KEYS, how="left")
            preds["TRUTH"] = tk.dropna(subset=["speed_kmh", "flow_vph"])
            preds["TOD_MEAN"] = naive_predictions(pdir, panel, ref.assign(speed_kmh=np.nan, flow_vph=np.nan))
        for m, p in preds.items():
            res = score_predictions(panel, p, truth, env); res.insert(1, "method", m)
            out.append(res)
            print(res[["panel", "method", "regime", "n_rows", "S_FD", "S_LWR", "E_LWR", "S_physics"]].round(4).to_string(index=False), flush=True)
        pd.concat(out, ignore_index=True).to_csv(a.output, index=False)
    res = pd.concat(out, ignore_index=True)
    print(res.groupby("method")[["S_FD", "S_LWR", "S_physics"]].mean().round(4))


if __name__ == "__main__":
    main()
