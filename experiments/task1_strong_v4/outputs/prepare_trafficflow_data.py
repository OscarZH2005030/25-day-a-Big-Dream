"""Prepare the user-downloaded TrafficFlowBench release for Task 1.

Features use ONLY the published masked observation layer and public geometry.
Training targets come from the official CSV templates, not arbitrary NaNs.
February is excluded from optimization: Feb 1-21 validation, Feb 22-28 test.
"""
import argparse
import datetime as dt
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

FEATURES = ["interpolated_speed/100", "interpolated_per_lane_flow/2000",
            "observed_speed/100", "observed_per_lane_flow/2000", "observed",
            "pct_observed/100", "free_speed/120", "capacity_per_lane/2400",
            "lanes/8", "length_km/5", "time_sin", "time_cos",
            "weekday_sin", "weekday_cos", "nearest_observation_gap/24"]


def grid(frame, links, columns):
    # A full daily grid is required; validate before reshaping.
    frame = frame.sort_values(["timestamp", "link_id"])
    assert len(frame) == 288 * len(links)
    assert not frame.duplicated(["timestamp", "link_id"]).any()
    assert frame.groupby("timestamp").size().eq(len(links)).all()
    order = pd.Index(sorted(links)).get_indexer(links)
    return [frame[c].to_numpy().reshape(288, len(links))[:, order] for c in columns]


def interpolate(values, fallback):
    out = np.empty_like(values, dtype=np.float32)
    gap = np.empty_like(out)
    steps = np.arange(288)
    for j in range(values.shape[1]):
        ok = np.isfinite(values[:, j])
        if ok.any():
            positions = steps[ok]
            out[:, j] = np.interp(steps, positions, values[ok, j])
            gap[:, j] = np.minimum(np.abs(steps[:, None] - positions).min(axis=1), 24) / 24
        else:
            out[:, j] = fallback[j]
            gap[:, j] = 1
    return out, gap


def prepare(release, cache):
    start = time.time()
    manifest = json.loads((release / "config/corridors.json").read_text())
    cache.mkdir(parents=True, exist_ok=True)
    metadata = {"features": FEATURES, "train_dates": ["2030-06-01", "2031-01-31"],
                "validation_dates": ["2031-02-01", "2031-02-21"],
                "test_dates": ["2031-02-22", "2031-02-28"], "panels": []}
    total_targets = 0
    for spec in manifest["panels"]:
        panel = spec["corridor_id"]
        root = release / "corridors" / panel
        dest = cache / panel
        dest.mkdir(exist_ok=True)
        topology = pd.read_csv(root / "network/lwr_mainline_topology.csv").sort_values("order_index")
        links = topology.mainline_link_id.astype(str).tolist()
        static = pd.read_csv(root / "network/fd_parameters.csv").set_index("link_id").loc[links]
        lanes = static.lanes.to_numpy(dtype=np.float32)
        free = static.free_speed_kmh.to_numpy(dtype=np.float32)
        capacity_lane = static.capacity_vph.to_numpy(dtype=np.float32) / lanes
        paths = sorted((root / "train/mainline_states").rglob("*.parquet"))
        dates = [p.stem[-10:].replace("_", "-") for p in paths]
        assert len(paths) == 273 and dates[0] == "2030-06-01" and dates[-1] == "2031-02-28"
        masked_by_name = {p.name: p for p in (root / "train/mainline_states_masked").rglob("*.parquet")}
        shape = (len(paths), 288, len(links))
        x = np.lib.format.open_memmap(dest / "x.npy", mode="w+", dtype=np.float16,
                                      shape=(len(paths), len(FEATURES), 288, len(links)))
        y = np.lib.format.open_memmap(dest / "y.npy", mode="w+", dtype=np.float32,
                                      shape=(len(paths), 2, 288, len(links)))
        targets = np.lib.format.open_memmap(dest / "target.npy", mode="w+", dtype=bool, shape=shape)
        targets[:] = False
        date_index = {d: i for i, d in enumerate(dates)}
        link_index = {v: i for i, v in enumerate(links)}
        count = 0
        for chunk in pd.read_csv(release / "task1" / panel / "train/sample_submission_state.csv",
                                 usecols=["timestamp", "link_id"], chunksize=300000):
            ts = pd.to_datetime(chunk.timestamp, utc=True)
            di = ts.dt.strftime("%Y-%m-%d").map(date_index).to_numpy()
            ti = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()
            li = chunk.link_id.map(link_index).to_numpy()
            assert not (pd.isna(di).any() or pd.isna(li).any())
            targets[di.astype(int), ti.astype(int), li.astype(int)] = True
            count += len(chunk)
        assert int(targets.sum()) == count, "Duplicate or missing template keys"
        regimes = []
        for d, path in enumerate(paths):
            masked = pd.read_parquet(masked_by_name[path.name])
            truth = pd.read_parquet(path)
            columns = ["speed_kmh", "flow_vph", "pct_observed"]
            speed, flow, quality = grid(masked, links, columns)
            true_s, true_q, eligible = grid(truth, links, ["speed_kmh", "flow_vph", "is_score_eligible"])
            valid = np.isfinite(speed) & np.isfinite(flow)
            assert not valid[targets[d]].any(), "Target leakage in masked features"
            assert eligible[targets[d]].all()
            assert np.isfinite(true_s[targets[d]]).all() and np.isfinite(true_q[targets[d]]).all()
            filled_s, gap = interpolate(speed, free)
            filled_q, _ = interpolate(flow / lanes, capacity_lane * .2)
            weekday = dt.date.fromisoformat(dates[d]).weekday()
            tod = np.arange(288, dtype=np.float32)[:, None] * (2 * np.pi / 288)
            channels = [filled_s / 100, filled_q / 2000,
                        np.nan_to_num(speed) / 100, np.nan_to_num(flow / lanes) / 2000,
                        valid, quality / 100, free / 120, capacity_lane / 2400, lanes / 8,
                        static.length_km.to_numpy() / 5, np.sin(tod), np.cos(tod),
                        np.sin(weekday * 2 * np.pi / 7), np.cos(weekday * 2 * np.pi / 7), gap]
            for c, values in enumerate(channels):
                x[d, c] = np.broadcast_to(values, shape[1:])
            y[d, 0] = np.nan_to_num(true_s) / 25
            y[d, 1] = np.nan_to_num(true_q / lanes) / 600
            regimes.append(str(masked.mask_regime.iloc[0]))
            if d % 90 == 0:
                print(json.dumps({"panel": panel, "day": d, "targets": count}), flush=True)
        assert np.isfinite(x).all()
        x.flush(); y.flush(); targets.flush()
        details = dict(panel=panel, family=spec["family_id"], dates=dates, links=links,
                       regimes=regimes, lanes=lanes.tolist(), shape=list(shape),
                       target_count=count, train_days=245, validation_days=21, test_days=7)
        (dest / "metadata.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
        metadata["panels"].append(details)
        total_targets += count
        print(json.dumps({"prepared": panel, "seconds": round(time.time() - start, 1)}), flush=True)
    metadata.update(target_count=total_targets, preparation_seconds=time.time()-start,
                    source_release=str(release.resolve()))
    (cache / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"completed": True, "targets": total_targets, "seconds": time.time()-start}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.release, args.cache)
