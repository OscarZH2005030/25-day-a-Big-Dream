"""Score the Task 2 persistence baseline against derived train queue labels.

The public release does not ship queue labels. For train-only diagnostics this
script derives them from the complete unmasked train speed records using the
published 0.60 * free-flow-speed rule and the published eligibility flag.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_parts(root: Path, panel: str, split: str, filename: str) -> pd.DataFrame:
    path = root / "task2" / panel / split / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"window_id": str, "timestamp": str, "link_id": str})


def state_paths(panel_root: Path, dates: set[str]) -> list[Path]:
    return [p for p in (panel_root / "train" / "mainline_states").glob("**/*.parquet")
            if p.stem.removeprefix("synthetic_mainline_").replace("_", "-") in dates]


def main() -> None:
    parser = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--release-root", type=Path, default=repo / "data" / "kaggle_public")
    parser.add_argument("--submission", type=Path, default=repo / "outputs" / "task2_persistence_train.csv")
    parser.add_argument("--output", type=Path, default=repo / "outputs" / "task2_persistence_train_proxy.csv")
    args = parser.parse_args()
    root = args.release_root.resolve()

    panels = sorted(p.name for p in (root / "task2").iterdir() if p.is_dir())
    indices, templates = [], []
    for panel in panels:
        idx = read_parts(root, panel, "train", "window_index.csv")
        tmp = read_parts(root, panel, "train", "sample_submission_queue.csv")
        if not idx.empty:
            indices.append(idx)
            templates.append(tmp)
    index = pd.concat(indices, ignore_index=True)
    template = pd.concat(templates, ignore_index=True)
    prediction = pd.read_csv(args.submission, dtype={"window_id": str, "timestamp": str, "link_id": str})
    for frame in (template, prediction):
        frame["timestamp_key"] = pd.to_datetime(frame["timestamp"], utc=True)
    keys = ["window_id", "timestamp_key", "link_id"]
    if template.duplicated(keys).any() or prediction.duplicated(keys).any():
        raise ValueError("Duplicate Task 2 submission keys")
    if set(map(tuple, template[keys].itertuples(index=False, name=None))) != set(map(tuple, prediction[keys].itertuples(index=False, name=None))):
        raise ValueError("Train baseline rows do not exactly match the Task 2 template")

    index["window_id"] = index["window_id"].astype(str)
    index["panel"] = index["panel"].astype(str)
    index["condition"] = index["condition"].astype(str)
    template = template.merge(index[["window_id", "panel", "condition"]], on="window_id", how="left", validate="many_to_one")
    prediction = prediction.rename(columns={"queue_pred": "pred_queue"})
    merged = template.merge(prediction[keys + ["pred_queue"]], on=keys, how="left", validate="one_to_one")
    if merged["pred_queue"].isna().any():
        raise ValueError("Missing persistence predictions for some train target rows")

    truth_parts = []
    for panel in panels:
        panel_rows = merged[merged["panel"] == panel]
        if panel_rows.empty:
            continue
        panel_root = root / "corridors" / panel
        target_dates = set(panel_rows["timestamp_key"].dt.strftime("%Y-%m-%d"))
        target_times = set(panel_rows["timestamp_key"])
        target_links = set(panel_rows["link_id"].astype(str))
        paths = state_paths(panel_root, target_dates)
        if not paths:
            raise FileNotFoundError(f"No train state files found for Task 2 targets in {panel}")
        chunks = []
        for path in paths:
            state = pd.read_parquet(path, columns=["timestamp", "link_id", "speed_kmh", "is_score_eligible"])
            state["link_id"] = state["link_id"].astype(str)
            state["timestamp_key"] = pd.to_datetime(state["timestamp"], utc=True)
            state = state[state["timestamp_key"].isin(target_times) & state["link_id"].isin(target_links)]
            if not state.empty:
                chunks.append(state[["timestamp_key", "link_id", "speed_kmh", "is_score_eligible"]])
        states = pd.concat(chunks, ignore_index=True)
        links = pd.read_csv(panel_root / "network" / "links.csv", dtype={"link_id": str})
        threshold = (links.set_index("link_id")["free_speed_kmh"].astype(float) * 0.60).to_dict()
        panel_rows = merged[merged["panel"] == panel].merge(
            states, on=["timestamp_key", "link_id"], how="left", validate="many_to_one")
        panel_rows["eligible"] = (panel_rows["is_score_eligible"].fillna(0).astype(bool)
                                   & pd.to_numeric(panel_rows["speed_kmh"], errors="coerce").notna())
        panel_rows["true_queue"] = (pd.to_numeric(panel_rows["speed_kmh"], errors="coerce")
                                     <= panel_rows["link_id"].map(threshold))
        truth_parts.append(panel_rows)

    scored = pd.concat(truth_parts, ignore_index=True)
    scored = scored[scored["eligible"]].copy()
    scored["pred_queue"] = scored["pred_queue"].astype(int).astype(bool)
    scored["true_queue"] = scored["true_queue"].astype(bool)
    window_rows = []
    for window_id, group in scored.groupby("window_id", sort=True):
        union = group["pred_queue"] | group["true_queue"]
        intersection = group["pred_queue"] & group["true_queue"]
        iou = float(intersection.sum() / union.sum()) if union.any() else 1.0
        meta = group.iloc[0]
        window_rows.append({"window_id": window_id, "panel": meta["panel"],
                            "condition": meta["condition"], "n_eligible": len(group), "IoU": iou})
    result = pd.DataFrame(window_rows)
    if result["window_id"].nunique() != index["window_id"].nunique():
        raise ValueError("Some train windows did not receive an IoU score")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print("By condition:")
    print(result.groupby("condition")["IoU"].agg(["count", "mean"]).to_string())
    print("\nBy panel:")
    print(result.groupby("panel")["IoU"].agg(["count", "mean"]).to_string())
    print(f"\nTrain proxy S_queue: {result['IoU'].mean():.6f} over {len(result)} windows")
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
