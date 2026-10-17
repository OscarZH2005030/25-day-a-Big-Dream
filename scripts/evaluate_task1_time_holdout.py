"""Evaluate the official Task 1 historical-mean baseline on a time holdout.

The final train month is held out. Profiles are fitted only on earlier train
days, then scored on the official Task 1 target rows for the held-out month.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def slots(timestamp: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(timestamp, utc=True)
    return (ts.dt.weekday.to_numpy(dtype=np.int16) * 288
            + ts.dt.hour.to_numpy(dtype=np.int16) * 12
            + (ts.dt.minute.to_numpy(dtype=np.int16) // 5))


def panel_files(root: Path, panel: str) -> list[Path]:
    return sorted((root / "corridors" / panel / "train" / "mainline_states").glob("**/*.parquet"))


def score_panel(root: Path, panel: str, cutoff: str) -> list[dict]:
    panel_root = root / "corridors" / panel
    paths = panel_files(root, panel)
    if not paths:
        raise FileNotFoundError(f"No unmasked train files for {panel}")

    first = pd.read_parquet(paths[0], columns=["link_id"])
    link_ids = sorted(first["link_id"].astype(str).unique())
    link_index = {v: i for i, v in enumerate(link_ids)}
    shape = (len(link_ids), 7 * 288)
    speed_sum = np.zeros(shape, dtype=np.float64)
    flow_sum = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.int64)

    read_cols = ["date", "timestamp", "station_id", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"]
    for path in paths:
        frame = pd.read_parquet(path, columns=read_cols)
        frame["link_id"] = frame["link_id"].astype(str)
        before = frame["date"].astype(str) < cutoff
        link = frame["link_id"].map(link_index).to_numpy(dtype=np.int64)
        slot = slots(frame["timestamp"])
        speed = pd.to_numeric(frame["speed_kmh"], errors="coerce").to_numpy(dtype=float)
        flow = pd.to_numeric(frame["flow_vph"], errors="coerce").to_numpy(dtype=float)
        eligible = (frame["is_score_eligible"].astype(bool).to_numpy() & before.to_numpy()
                    & np.isfinite(speed) & np.isfinite(flow))
        np.add.at(speed_sum, (link[eligible], slot[eligible]), speed[eligible])
        np.add.at(flow_sum, (link[eligible], slot[eligible]), flow[eligible])
        np.add.at(count, (link[eligible], slot[eligible]), 1)

    speed_profile = np.divide(speed_sum, np.maximum(count, 1))
    flow_profile = np.divide(flow_sum, np.maximum(count, 1))
    speed_fallback = speed_sum.sum(axis=1) / np.maximum(count.sum(axis=1), 1)
    flow_fallback = flow_sum.sum(axis=1) / np.maximum(count.sum(axis=1), 1)

    template_path = root / "task1" / panel / "train" / "sample_submission_state.csv"
    template = pd.read_csv(template_path, dtype={"timestamp": str, "station_id": str, "link_id": str})
    target = template.loc[template["timestamp"].str[:10] >= cutoff,
                          ["panel", "timestamp", "station_id", "link_id", "mask_regime"]].copy()
    if target.empty:
        raise ValueError(f"No Task 1 targets on or after {cutoff} for {panel}")

    target["link_id"] = target["link_id"].astype(str)
    target_link = target["link_id"].map(link_index).to_numpy(dtype=np.int64)
    target_slot = slots(target["timestamp"])
    target["pred_speed_kmh"] = speed_profile[target_link, target_slot]
    target["pred_flow_vph"] = flow_profile[target_link, target_slot]
    no_profile = count[target_link, target_slot] == 0
    target.loc[no_profile, "pred_speed_kmh"] = speed_fallback[target_link[no_profile]]
    target.loc[no_profile, "pred_flow_vph"] = flow_fallback[target_link[no_profile]]
    target[["pred_speed_kmh", "pred_flow_vph"]] = target[["pred_speed_kmh", "pred_flow_vph"]].clip(lower=0)

    holdout_dates = set(target["timestamp"].str[:10])
    truth_parts = []
    for path in paths:
        date_text = path.stem.removeprefix("synthetic_mainline_").replace("_", "-")
        if date_text in holdout_dates:
            truth_parts.append(pd.read_parquet(path, columns=["timestamp", "station_id", "link_id", "speed_kmh", "flow_vph"]))
    truth = pd.concat(truth_parts, ignore_index=True)
    truth["station_id"] = truth["station_id"].astype(str)
    truth["link_id"] = truth["link_id"].astype(str)
    scored = target.merge(truth, on=["timestamp", "station_id", "link_id"], how="left", validate="one_to_one")
    if scored[["speed_kmh", "flow_vph"]].isna().any().any():
        raise ValueError(f"Missing train truth for some Task 1 template rows in {panel}")

    links = pd.read_csv(panel_root / "network" / "links.csv", dtype={"link_id": str})
    lanes = links.set_index("link_id")["lanes"].astype(float).to_dict()
    scored["lanes"] = scored["link_id"].map(lanes).fillna(1.0).clip(lower=1).to_numpy(dtype=float)
    result = []
    for regime in ("R1", "R2", "R3"):
        g = scored[scored["mask_regime"].astype(str) == regime]
        ds = g["pred_speed_kmh"].to_numpy(dtype=float) - g["speed_kmh"].to_numpy(dtype=float)
        dq = ((g["pred_flow_vph"].to_numpy(dtype=float) - g["flow_vph"].to_numpy(dtype=float))
              / g["lanes"].to_numpy(dtype=float))
        rmse_speed = float(np.sqrt(np.mean(ds * ds))) if len(g) else float("nan")
        rmse_flow = float(np.sqrt(np.mean(dq * dq))) if len(g) else float("nan")
        score_speed = max(0.0, 1.0 - rmse_speed / 25.0) if np.isfinite(rmse_speed) else float("nan")
        score_flow = max(0.0, 1.0 - rmse_flow / 600.0) if np.isfinite(rmse_flow) else float("nan")
        result.append({
            "panel": panel, "family": panel.rsplit("_", 1)[0], "regime": regime,
            "n_targets": len(g), "rmse_speed_kmh": rmse_speed,
            "rmse_flow_vph_per_lane": rmse_flow, "S_speed": score_speed,
            "S_flow": score_flow,
            "S_state": 0.54 * score_speed + 0.46 * score_flow
            if np.isfinite(score_speed) and np.isfinite(score_flow) else float("nan"),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "kaggle_public")
    parser.add_argument("--cutoff", default="2031-02-01", help="first held-out date; default is the final train month")
    parser.add_argument("--panel", action="append", help="optional panel filter")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "task1_time_holdout.csv")
    args = parser.parse_args()
    panels = sorted(path.name for path in (args.release_root / "task1").iterdir() if path.is_dir())
    if args.panel:
        unknown = sorted(set(args.panel) - set(panels))
        if unknown:
            parser.error(f"Unknown panel(s): {unknown}")
        panels = [p for p in panels if p in set(args.panel)]
    rows = []
    for panel in panels:
        print(f"[Task1 time holdout] {panel}", flush=True)
        rows.extend(score_panel(args.release_root.resolve(), panel, args.cutoff))
    detail = pd.DataFrame(rows)
    panel_scores = detail.groupby(["panel", "family"], as_index=False)["S_state"].mean()
    corridor_scores = panel_scores.groupby("family", as_index=False)["S_state"].mean()
    total = float(corridor_scores["S_state"].mean())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output, index=False)
    corridor_scores.to_csv(args.output.with_name("task1_time_holdout_by_corridor.csv"), index=False)
    print(detail.to_string(index=False))
    print("\nMean by corridor family:")
    print(corridor_scores.to_string(index=False))
    print(f"\nTask 1 time-holdout S_state: {total:.6f}")
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
