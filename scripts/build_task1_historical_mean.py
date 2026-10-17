"""Build Task 1 predictions with a weekday-by-time-of-day historical mean."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def slots(timestamp: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(timestamp, utc=True)
    return (ts.dt.weekday.to_numpy(dtype=np.int16) * 288
            + ts.dt.hour.to_numpy(dtype=np.int16) * 12
            + ts.dt.minute.to_numpy(dtype=np.int16) // 5)


def train_files(root: Path, panel: str) -> list[Path]:
    return sorted((root / "corridors" / panel / "train" / "mainline_states").glob("**/*.parquet"))


def build_profile(root: Path, panel: str) -> tuple[dict[str, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths = train_files(root, panel)
    if not paths:
        raise FileNotFoundError(f"No unmasked train parquet files for {panel}")
    first = pd.read_parquet(paths[0], columns=["link_id"])
    link_ids = sorted(first["link_id"].astype(str).unique())
    link_index = {v: i for i, v in enumerate(link_ids)}
    shape = (len(link_ids), 7 * 288)
    speed_sum = np.zeros(shape, dtype=np.float64)
    flow_sum = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.int64)
    for path in paths:
        frame = pd.read_parquet(path, columns=["timestamp", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"])
        frame["link_id"] = frame["link_id"].astype(str)
        link = frame["link_id"].map(link_index).to_numpy(dtype=np.int64)
        slot = slots(frame["timestamp"])
        speed = pd.to_numeric(frame["speed_kmh"], errors="coerce").to_numpy(dtype=float)
        flow = pd.to_numeric(frame["flow_vph"], errors="coerce").to_numpy(dtype=float)
        valid = (frame["is_score_eligible"].astype(bool).to_numpy()
                 & np.isfinite(speed) & np.isfinite(flow))
        np.add.at(speed_sum, (link[valid], slot[valid]), speed[valid])
        np.add.at(flow_sum, (link[valid], slot[valid]), flow[valid])
        np.add.at(count, (link[valid], slot[valid]), 1)
    speed_profile = np.divide(speed_sum, np.maximum(count, 1))
    flow_profile = np.divide(flow_sum, np.maximum(count, 1))
    speed_fallback = speed_sum.sum(axis=1) / np.maximum(count.sum(axis=1), 1)
    flow_fallback = flow_sum.sum(axis=1) / np.maximum(count.sum(axis=1), 1)
    return link_index, speed_profile, flow_profile, speed_fallback, flow_fallback, count


def predict_template(root: Path, panel: str, split: str, output: Path,
                     profile: tuple[dict[str, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                     write_header: bool) -> tuple[int, bool]:
    template_path = root / "task1" / panel / split / "sample_submission_state.csv"
    link_index, speed_profile, flow_profile, speed_fallback, flow_fallback, count = profile
    columns = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]
    rows = 0
    for chunk in pd.read_csv(template_path, usecols=columns, dtype={
        "panel": str, "timestamp": str, "station_id": str, "link_id": str, "mask_regime": str,
    }, chunksize=250_000):
        link_text = chunk["link_id"].astype(str)
        link = link_text.map(link_index)
        if link.isna().any():
            raise ValueError(f"Unknown link_id in {template_path}")
        li = link.to_numpy(dtype=np.int64)
        slot = slots(chunk["timestamp"])
        speed = speed_profile[li, slot]
        flow = flow_profile[li, slot]
        no_profile = count[li, slot] == 0
        speed[no_profile] = speed_fallback[li[no_profile]]
        flow[no_profile] = flow_fallback[li[no_profile]]
        chunk["speed_kmh"] = np.maximum(speed, 0.0)
        chunk["flow_vph"] = np.maximum(flow, 0.0)
        chunk.to_csv(output, mode="w" if write_header else "a", header=write_header, index=False)
        write_header = False
        rows += len(chunk)
    return rows, write_header


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=repo / "data" / "kaggle_public")
    parser.add_argument("--split", choices=["validation", "private", "both"], default="both")
    parser.add_argument("--output-dir", type=Path, default=repo / "outputs")
    parser.add_argument("--panel", action="append", help="optional panel filter")
    args = parser.parse_args()
    root = args.release_root.resolve()
    splits = ["validation", "private"] if args.split == "both" else [args.split]
    panels = sorted(path.name for path in (root / "task1").iterdir() if path.is_dir())
    if args.panel:
        unknown = sorted(set(args.panel) - set(panels))
        if unknown:
            parser.error(f"Unknown panel(s): {unknown}")
        panels = [p for p in panels if p in set(args.panel)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {split: args.output_dir / f"task1_historical_mean_{split}.csv" for split in splits}
    for output in outputs.values():
        if output.exists():
            output.unlink()
    totals = {split: 0 for split in splits}
    headers = {split: True for split in splits}
    for panel in panels:
        print(f"[Task1 profile] {panel}", flush=True)
        profile = build_profile(root, panel)
        for split in splits:
            count, headers[split] = predict_template(
                root, panel, split, outputs[split], profile, headers[split])
            totals[split] += count
            print(f"  {split}: wrote {count:,} rows", flush=True)
    for split in splits:
        print(f"Wrote {totals[split]:,} rows to {outputs[split].resolve()}")


if __name__ == "__main__":
    main()
