"""Local Task 3 diagnostic using released validation state and public formulas.

This is deliberately labelled a proxy: the organizer's boundary flux is not
released, so topology-derived boundary flow cannot reproduce the official LWR
score. The FD calculation and state reconstruction follow the current public
scorer; the LWR component uses its public topology fallback.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "data" / "kaggle_public"
OUT = ROOT / "outputs" / "task3_local_proxy"
PANELS = [
    "D12_I405_N", "D12_I405_S", "D12_I5_N", "D12_I5_S",
    "D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W",
    "D7_I405_N", "D7_I405_S",
]
REGIMES = ("R1", "R2", "R3")
KEYS = ["timestamp", "station_id", "link_id"]


def model_predictions(path: Path, panel: str, source: str, historical: pd.DataFrame) -> pd.DataFrame:
    if source == "historical_mean":
        f = historical.loc[historical.panel == panel, [*KEYS, "speed_kmh", "flow_vph"]].copy()
    else:
        f = pd.read_parquet(path, columns=[*KEYS, "pred_speed_kmh", "pred_flow_vph"])
        f = f.rename(columns={"pred_speed_kmh": "speed_kmh", "pred_flow_vph": "flow_vph"})
    f["timestamp"] = f.timestamp.astype(str)
    f["station_id"] = f.station_id.astype(str)
    f["link_id"] = f.link_id.astype(str)
    if f.duplicated(KEYS).any():
        raise ValueError(f"{panel}/{model}: duplicate prediction keys")
    return f


def network_parameters(panel_dir: Path) -> pd.DataFrame:
    links = pd.read_csv(panel_dir / "network" / "links.csv", dtype={"link_id": str})
    out = links[["link_id", "length_km", "free_speed_kmh", "capacity_vph", "lanes"]].drop_duplicates("link_id")
    fd_path = panel_dir / "network" / "fd_parameters.csv"
    if fd_path.exists():
        fd = pd.read_csv(fd_path, dtype={"link_id": str})
        cols = [c for c in ("free_speed_kmh", "capacity_vph", "lanes", "k_jam") if c in fd.columns]
        fd = fd.groupby("link_id", as_index=False)[cols].mean()
        out = out.merge(fd, on="link_id", how="left", suffixes=("", "_fd"))
        for c in ("free_speed_kmh", "capacity_vph", "lanes"):
            if f"{c}_fd" in out:
                out[c] = out[f"{c}_fd"].fillna(out[c])
                out = out.drop(columns=[f"{c}_fd"])
    else:
        out["k_jam"] = np.nan
    out["critical_density"] = out.capacity_vph / out.free_speed_kmh.clip(lower=1)
    out["k_jam"] = out.k_jam.fillna(100.0 * out.lanes.clip(lower=1.0))
    out["k_jam"] = np.maximum(out.k_jam, out.critical_density * 1.05)
    return out


def load_ramps(panel_dir: Path, allowed_timestamps: set[str]) -> pd.DataFrame:
    amap_path = panel_dir / "network" / "ramp_attachment_map.csv"
    root = panel_dir / "validation" / "ramp_states"
    cols = ["timestamp", "link_id", "on_ramp_flow_vph", "off_ramp_flow_vph", "on_ramp_valid", "off_ramp_valid"]
    if not amap_path.exists() or not root.exists():
        return pd.DataFrame(columns=cols)
    amap = pd.read_csv(amap_path, dtype=str).rename(columns={"nearest_mainline_link_id": "link_id"})
    pieces = []
    for path in root.rglob("*.parquet"):
        f = pd.read_parquet(path, columns=["timestamp", "ramp_link_id", "ramp_type", "flow_vph", "pct_observed", "is_score_eligible"])
        f["timestamp"] = f.timestamp.astype(str)
        f = f[f.timestamp.isin(allowed_timestamps)]
        f["flow_vph"] = pd.to_numeric(f.flow_vph, errors="coerce")
        f["pct_observed"] = pd.to_numeric(f.pct_observed, errors="coerce")
        f = f[f.is_score_eligible.astype(bool) & (f.pct_observed >= 75) & f.flow_vph.notna()]
        if f.empty:
            continue
        f = f.merge(amap[["ramp_link_id", "link_id"]], on="ramp_link_id", how="inner")
        is_on = f.ramp_type.astype(str).str.upper().isin(["OR", "ON"])
        is_off = f.ramp_type.astype(str).str.upper().isin(["FR", "OFF"])
        f["on_ramp_flow_vph"] = np.where(is_on, f.flow_vph, 0.0)
        f["off_ramp_flow_vph"] = np.where(is_off, f.flow_vph, 0.0)
        f["on_ramp_valid"] = is_on.astype(int)
        f["off_ramp_valid"] = is_off.astype(int)
        pieces.append(f[cols])
    if not pieces:
        return pd.DataFrame(columns=cols)
    return pd.concat(pieces, ignore_index=True).groupby(["timestamp", "link_id"], as_index=False).agg(
        on_ramp_flow_vph=("on_ramp_flow_vph", "sum"),
        off_ramp_flow_vph=("off_ramp_flow_vph", "sum"),
        on_ramp_valid=("on_ramp_valid", "max"),
        off_ramp_valid=("off_ramp_valid", "max"),
    )


def topology_boundaries(state: pd.DataFrame, panel_dir: Path) -> pd.DataFrame:
    topo_path = panel_dir / "network" / "lwr_mainline_topology.csv"
    cols = ["timestamp", "link_id", "inflow_vph", "outflow_vph"]
    if not topo_path.exists():
        return pd.DataFrame(columns=cols)
    topo = pd.read_csv(topo_path, dtype=str).drop_duplicates("link_id").set_index("link_id")
    pivot = state.pivot_table(index="timestamp", columns="link_id", values="flow_vph", aggfunc="mean")
    known = set(pivot.columns.astype(str))
    rows = []
    for link in state.link_id.astype(str).drop_duplicates():
        if link not in pivot.columns:
            continue
        if link in topo.index:
            incoming = [v for v in str(topo.loc[link].get("incoming_link_ids", "")).split(";") if v in known]
            outgoing = [v for v in str(topo.loc[link].get("outgoing_link_ids", "")).split(";") if v in known]
        else:
            incoming, outgoing = [], []
        inflow = pivot[incoming].sum(axis=1) if incoming else pivot[link]
        outflow = pivot[outgoing].sum(axis=1) if outgoing else pivot[link]
        rows.append(pd.DataFrame({"timestamp": pivot.index, "link_id": link,
                                  "inflow_vph": inflow.to_numpy(), "outflow_vph": outflow.to_numpy()}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)


def score_regime(panel: str, regime: str, panel_dir: Path, predictions: pd.DataFrame) -> dict:
    template = pd.read_csv(panel_dir.parent.parent / "task1" / panel / "validation" / "sample_submission_state.csv",
                           usecols=[*KEYS, "mask_regime"], dtype=str)
    template = template[template.mask_regime == regime].copy()
    template["timestamp"] = template.timestamp.astype(str)
    template["station_id"] = template.station_id.astype(str)
    template["link_id"] = template.link_id.astype(str)
    target = template[[*KEYS]].drop_duplicates()
    target_pred = target.merge(predictions, on=KEYS, how="left", validate="one_to_one")
    missing = int(target_pred[["speed_kmh", "flow_vph"]].isna().any(axis=1).sum())
    if missing:
        raise ValueError(f"{panel}/{regime}: {missing:,} target cells lack model predictions")

    pieces = []
    masked_root = panel_dir / "validation" / "mainline_states_masked" / f"mask_regime={regime}"
    for path in sorted(masked_root.rglob("*.parquet")):
        f = pd.read_parquet(path, columns=["timestamp", "station_id", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"])
        f["timestamp"] = f.timestamp.astype(str)
        f["station_id"] = f.station_id.astype(str)
        f["link_id"] = f.link_id.astype(str)
        f = f[f.is_score_eligible.astype(bool)].copy()
        if f.empty:
            continue
        f = f.merge(target_pred.assign(is_target=1), on=KEYS, how="left", validate="one_to_one")
        is_target = f.is_target.notna().to_numpy()
        f["speed_kmh"] = np.where(is_target, f.speed_kmh_y, f.speed_kmh_x)
        f["flow_vph"] = np.where(is_target, f.flow_vph_y, f.flow_vph_x)
        f["is_target"] = is_target.astype(float)
        f["timestamp"] = pd.to_datetime(f.timestamp, utc=True)
        pieces.append(f[["timestamp", "link_id", "speed_kmh", "flow_vph", "is_target"]])
    if not pieces:
        raise ValueError(f"{panel}/{regime}: no eligible masked state files")
    state = pd.concat(pieces, ignore_index=True).groupby(["timestamp", "link_id"], as_index=False).mean(numeric_only=True)
    state["link_id"] = state.link_id.astype(str)

    params = network_parameters(panel_dir)
    x = state.merge(params, on="link_id", how="left")
    x = x.dropna(subset=["speed_kmh", "flow_vph", "length_km", "free_speed_kmh", "capacity_vph", "critical_density", "k_jam"])
    x["density_vpkm"] = x.flow_vph / x.speed_kmh.clip(lower=1.0)
    lanes = x.lanes.clip(lower=1.0)
    q_lane = x.flow_vph / lanes
    k_lane = x.density_vpkm / lanes
    cap_lane = x.capacity_vph / lanes
    kcrit_lane = x.critical_density / lanes
    kjam_lane = x.k_jam / lanes
    q_fd = np.where(k_lane <= kcrit_lane,
                    x.free_speed_kmh * k_lane,
                    cap_lane / np.maximum(kjam_lane - kcrit_lane, 1e-9) * np.maximum(kjam_lane - k_lane, 0))
    s_fd = float(max(0.0, 1.0 - np.abs(q_lane - q_fd).sum() / (np.abs(q_lane).sum() + 1e-9)))
    submitted = x.is_target.to_numpy() > 0.5
    if submitted.any() and float((x.flow_vph.to_numpy()[submitted] < 50.0).mean()) > 0.20:
        s_fd = 0.0

    x["accumulation_N"] = x.density_vpkm * x.length_km
    boundary = topology_boundaries(x, panel_dir)
    x = x.merge(boundary, on=["timestamp", "link_id"], how="left")
    x["inflow_vph"] = x.inflow_vph.fillna(x.flow_vph)
    x["outflow_vph"] = x.outflow_vph.fillna(x.flow_vph)
    ramps = load_ramps(panel_dir, set(x.timestamp.astype(str)))
    x = x.merge(ramps, on=["timestamp", "link_id"], how="left")
    for c in ("on_ramp_flow_vph", "off_ramp_flow_vph", "on_ramp_valid", "off_ramp_valid"):
        if c not in x:
            x[c] = 0.0
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
    topo = pd.read_csv(panel_dir / "network" / "lwr_mainline_topology.csv", dtype=str)
    on_attached = set(topo.loc[topo.on_ramp_link_ids.fillna("").astype(str).str.len() > 0, "link_id"].astype(str))
    off_attached = set(topo.loc[topo.off_ramp_link_ids.fillna("").astype(str).str.len() > 0, "link_id"].astype(str))
    x.loc[~x.link_id.isin(on_attached), "on_ramp_valid"] = 1.0
    x.loc[~x.link_id.isin(off_attached), "off_ramp_valid"] = 1.0
    x = x.sort_values(["link_id", "timestamp"]).copy()
    x["dN"] = x.groupby("link_id").accumulation_N.shift(-1) - x.accumulation_N
    x["dt"] = (x.groupby("link_id").timestamp.shift(-1) - x.timestamp).dt.total_seconds() / 3600.0
    x = x[(x.dt > 0) & (x.dt <= 5.1 / 60.0) & (x.on_ramp_valid >= 0.5) & (x.off_ramp_valid >= 0.5)].copy()
    rhs = x.dt * (x.inflow_vph + x.on_ramp_flow_vph - x.outflow_vph - x.off_ramp_flow_vph)
    if x.empty:
        e_lwr, s_lwr = 1.0, 0.0
    else:
        resid = (x.dN - rhs).abs()
        e_lwr = float(resid.sum() / (rhs.abs().sum() + 1e-9))
        s_lwr = max(0.0, 1.0 - min(1.0, e_lwr))
    return {
        "panel": panel, "regime": regime, "target_cells": len(target),
        "missing_predictions": missing, "eligible_state_rows": int(len(state)),
        "lwr_intervals": int(len(x)), "S_FD": s_fd,
        "S_LWR_topology_proxy": s_lwr, "E_LWR_topology_proxy": e_lwr,
        "S_physics_topology_proxy": (s_fd + 2.0 * s_lwr) / 3.0,
    }


def evaluate_artifact(name: str, historical: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for panel in PANELS:
        print(f"[{name}] {panel}", flush=True)
        panel_dir = RELEASE / "corridors" / panel
        if historical is None:
            pred_path = OUT.parent / "task1_cv_all" / panel / "validation_predictions.parquet"
            # This artifact was checked against historical_mean_validation.csv
            # and is numerically identical on all ten panels. It is not a
            # Transformer prediction despite residing in task1_cv_all.
            predictions = model_predictions(pred_path, panel, "task1_cv_all_artifact", pd.DataFrame())
        else:
            predictions = model_predictions(Path(), panel, "historical_mean", historical)
        for regime in REGIMES:
            rows.append(score_regime(panel, regime, panel_dir, predictions))
    detail = pd.DataFrame(rows)
    panel_means = detail.groupby("panel", as_index=False)[["S_FD", "S_LWR_topology_proxy", "S_physics_topology_proxy"]].mean()
    summary = {
        "panel": "ALL_EQUAL_PANEL_MEAN", "regime": "macro_mean",
        "target_cells": int(detail.target_cells.sum()), "missing_predictions": int(detail.missing_predictions.sum()),
        "eligible_state_rows": int(detail.eligible_state_rows.sum()), "lwr_intervals": int(detail.lwr_intervals.sum()),
        "S_FD": float(panel_means.S_FD.mean()),
        "S_LWR_topology_proxy": float(panel_means.S_LWR_topology_proxy.mean()),
        "E_LWR_topology_proxy": float("nan"),
        "S_physics_topology_proxy": float(panel_means.S_physics_topology_proxy.mean()),
    }
    detail = pd.concat([detail, pd.DataFrame([summary])], ignore_index=True)
    return detail


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hist_path = OUT.parent / "task1_historical_mean_validation.csv"
    historical = pd.read_csv(hist_path, dtype={"panel": str, "station_id": str, "link_id": str})
    transformer = evaluate_artifact("task1_cv_all_artifact")
    historical_result = evaluate_artifact("historical_mean", historical)
    transformer.insert(0, "model", "task1_cv_all_artifact")
    historical_result.insert(0, "model", "historical_mean")
    all_results = pd.concat([transformer, historical_result], ignore_index=True)
    all_results.to_csv(OUT / "task3_local_proxy_by_panel_regime.csv", index=False)
    overall = all_results[all_results.panel == "ALL_EQUAL_PANEL_MEAN"].copy()
    overall.to_csv(OUT / "task3_local_proxy_summary.csv", index=False)
    (OUT / "task3_local_proxy_provenance.json").write_text(json.dumps({
        "split": "validation (2031-03-01 through 2031-03-31)",
        "model_predictions": "outputs/task1_cv_all/<panel>/validation_predictions.parquet; verified numerically identical to historical mean artifacts for all ten panels, so not labelled as a Transformer",
        "baseline_predictions": "outputs/task1_historical_mean_validation.csv",
        "scorer": "local reimplementation of public FD/LWR formulas; not organizer scorer",
        "boundary_flux": "not released; LWR uses topology-derived submitted-flow proxy",
        "mode": "A for all panels per current public config/task3_lwr_modes.json",
        "aggregation": "mean R1/R2/R3 per panel, then equal panel mean",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSummary (local proxy; not official Task 3):")
    print(overall.to_string(index=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
