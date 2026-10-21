"""Build a complete TrafficFlowBench submission for BOTH scored months.

Task 1: trained CNN. Task 2: official persistence baseline.
Task 3: reuses Task 1. Task 4: nonnegative regularized least squares.
All joins are strict: missing predictions fail rather than silently becoming zero.
"""
import argparse
from collections import Counter
import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prepare_trafficflow_data import FEATURES, grid, interpolate
from gpu_stress_test import save, telemetry
from trafficflow_model import Model

KEYS = {"state": ["panel", "timestamp", "station_id", "link_id", "mask_regime"],
        "queue": ["window_id", "timestamp", "link_id"],
        "odme": ["panel", "departure_time", "path_id"]}
VALUES = {"state": ["speed_kmh", "flow_vph"], "queue": ["queue_pred"], "odme": ["path_flow"]}
COLS = ["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]


def emit(out, phase, **kw):
    data = dict(phase=phase, timestamp=dt.datetime.now().astimezone().isoformat(), **kw)
    save(out / "progress.json", data)
    print(json.dumps(data), flush=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8*1024**2), b""):
            h.update(part)
    return h.hexdigest()


def daily_features(frame, links, static, day):
    speed, flow, quality = grid(frame, links, ["speed_kmh", "flow_vph", "pct_observed"])
    lanes = static.lanes.to_numpy(dtype=np.float32)
    free = static.free_speed_kmh.to_numpy(dtype=np.float32)
    capacity = static.capacity_vph.to_numpy(dtype=np.float32) / lanes
    filled_s, gap = interpolate(speed, free)
    filled_q, _ = interpolate(flow / lanes, capacity * .2)
    tod = np.arange(288, dtype=np.float32)[:, None] * (2*np.pi/288)
    weekday = dt.date.fromisoformat(day).weekday()
    channels = [filled_s/100, filled_q/2000, np.nan_to_num(speed)/100,
                np.nan_to_num(flow/lanes)/2000, np.isfinite(speed)&np.isfinite(flow),
                quality/100, free/120, capacity/2400, lanes/8,
                static.length_km.to_numpy()/5, np.sin(tod), np.cos(tod),
                np.sin(weekday*2*np.pi/7), np.cos(weekday*2*np.pi/7), gap]
    x = np.stack([np.broadcast_to(c, speed.shape) for c in channels]).astype(np.float16)
    assert np.isfinite(x).all()
    return x


def build_state(args, panels):
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.89)
    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location="cuda")
    assert checkpoint["features"] == FEATURES
    model = Model().cuda().to(memory_format=torch.channels_last).eval()
    model.load_state_dict(checkpoint["state_dict"])
    model_hash = digest(args.checkpoint)
    parts = args.out / "parts"
    parts.mkdir(exist_ok=True)
    for panel in panels:
        root = args.release / "corridors" / panel
        topology = pd.read_csv(root / "network/lwr_mainline_topology.csv").sort_values("order_index")
        links = topology.mainline_link_id.astype(str).tolist()
        li_map = {v:i for i,v in enumerate(links)}
        static = pd.read_csv(root / "network/fd_parameters.csv").set_index("link_id").loc[links]
        lanes = static.lanes.to_numpy(dtype=np.float32)
        for split in ["validation", "private"]:
            target = parts / f"state-{panel}-{split}.csv"
            meta_path = target.with_suffix(".json")
            if target.exists() and meta_path.exists():
                meta = json.loads(meta_path.read_text())
                if meta.get("checkpoint_sha256") == model_hash and meta.get("csv_sha256") == digest(target):
                    emit(args.out, "state_cached", panel=panel, split=split)
                    continue
            paths = sorted((root / split / "mainline_states_masked").rglob("*.parquet"),
                           key=lambda p: p.name)
            dates = [p.stem[-10:].replace("_", "-") for p in paths]
            assert len(paths) == (31 if split == "validation" else 30)
            xs, regimes, station_by_link = [], [], None
            for path, day in zip(paths, dates):
                frame = pd.read_parquet(path)
                station_map = frame[["link_id", "station_id"]].drop_duplicates().set_index("link_id").station_id.to_dict()
                assert len(station_map) == len(links)
                if station_by_link is None:
                    station_by_link = station_map
                assert station_map == station_by_link
                xs.append(daily_features(frame, links, static, day))
                regimes.append(str(frame.mask_regime.iloc[0]))
            inputs = np.stack(xs)
            predictions = []
            for begin in range(0, len(xs), 16):
                emit(args.out, "state_inference", panel=panel, split=split, day=begin)
                x = torch.from_numpy(inputs[begin:begin+16].astype(np.float32)).cuda().to(memory_format=torch.channels_last)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    pred = model(x).cpu().numpy()
                assert np.isfinite(pred).all() and (pred >= 0).all()
                predictions.append(pred)
                del x
            predictions = np.concatenate(predictions)
            template = pd.read_csv(args.release / "task1" / panel / split / "sample_submission_state.csv",
                                   dtype={k:str for k in KEYS["state"]})
            assert not template.duplicated(KEYS["state"]).any()
            ts = pd.to_datetime(template.timestamp, utc=True)
            di = ts.dt.strftime("%Y-%m-%d").map({d:i for i,d in enumerate(dates)}).to_numpy()
            ti = (ts.dt.hour*12+ts.dt.minute//5).to_numpy()
            si = template.link_id.map(li_map).to_numpy()
            assert not pd.isna(di).any() and not pd.isna(si).any()
            di, ti, si = di.astype(int), ti.astype(int), si.astype(int)
            assert (template.panel == panel).all()
            assert (template.station_id.to_numpy() == template.link_id.map(station_by_link).to_numpy()).all()
            assert (template.mask_regime.to_numpy() == np.asarray(regimes)[di]).all()
            assert not inputs[di, 4, ti, si].any(), "A target was visible in model inputs"
            assert (inputs[di, 5, ti, si] >= .75).all()
            template["speed_kmh"] = predictions[di, 0, ti, si]*25
            template["flow_vph"] = predictions[di, 1, ti, si]*600*lanes[si]
            temp = target.with_suffix(".partial")
            template.to_csv(temp, index=False, float_format="%.6f")
            temp.replace(target)
            save(meta_path, dict(panel=panel, split=split, rows=len(template),
                                checkpoint_sha256=model_hash, csv_sha256=digest(target),
                                input_source="masked observations and public network only"))
            emit(args.out, "state_completed", panel=panel, split=split, rows=len(template))
    del model
    torch.cuda.empty_cache()


def solve_odme(A, counts, prior, progress=None, maximum=5000):
    """ADMM solve of .5||Af-c||² + .5*lambda||f-b||², f>=0.

    Woodbury reduces each linear solve to measured-link space. Float64 keeps
    the normal equations stable; no hidden flow reference is used.
    """
    dtype = torch.float64
    a = torch.as_tensor(A, dtype=dtype, device="cuda")
    c = torch.as_tensor(counts, dtype=dtype, device="cuda")
    b = torch.as_tensor(prior, dtype=dtype, device="cuda")
    lam, rho = .05, .5
    d = lam+rho
    factor = torch.linalg.cholesky(a@a.T + d*torch.eye(a.shape[0], device="cuda", dtype=dtype))
    g = a.T@c + lam*b
    z, u = b.clamp_min(0).clone(), torch.zeros_like(b)
    converged = False
    for iteration in range(1, maximum+1):
        h = g+rho*(z-u)
        sol = torch.cholesky_solve((a@h)[:, None], factor)[:, 0]
        x = (h-a.T@sol)/d
        old = z
        z = (x+u).clamp_min(0)
        u = u+x-z
        if iteration % 100 == 0:
            primal = float(torch.linalg.vector_norm(x-z)/(1+torch.linalg.vector_norm(z)))
            dual = float(rho*torch.linalg.vector_norm(z-old)/(1+torch.linalg.vector_norm(g)))
            grad = a.T@(a@z-c)+lam*(z-b)
            projected = float(torch.linalg.vector_norm(z-(z-grad).clamp_min(0))/(1+torch.linalg.vector_norm(z)))
            if progress:
                progress(iteration=iteration, primal_relative=primal, projected_gradient_relative=projected)
            if primal < 1e-7 and projected < 1e-5:
                converged = True
                break
    flow = z.cpu().numpy()
    assert np.isfinite(flow).all() and (flow >= 0).all()
    metrics = dict(iterations=iteration, converged=converged, primal_relative=primal,
                   dual_relative=dual, projected_gradient_relative=projected, regularization=lam,
                   objective=float(.5*torch.sum((a@z-c)**2)+.5*lam*torch.sum((z-b)**2)))
    if not converged:
        raise RuntimeError(f"ODME solver has not converged: {metrics}")
    return flow, metrics


def build_auxiliary(args, panels):
    sys.path.insert(0, str(args.official / "src"))
    from task4.build_task4_odme_artifacts import load_operator
    records = []
    for split in ["validation", "private"]:
        emit(args.out, "queue", split=split)
        subprocess.run([sys.executable, str(args.official / "src/task2/build_task2_persistence_submission.py"),
                        "--release-root", str(args.release), "--split", split,
                        "--output", str(args.out / "parts" / f"queue-{split}.csv")], check=True)
        queue_path = args.out / "parts" / f"queue-{split}.csv"
        queue = pd.read_csv(queue_path, dtype={k:str for k in KEYS["queue"]})
        queue["timestamp"] = pd.to_datetime(queue.timestamp, utc=True, errors="raise").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        assert not queue.duplicated(KEYS["queue"]).any()
        queue.to_csv(queue_path, index=False)
        for panel in panels:
            target = args.out / "parts" / f"odme-{panel}-{split}.csv"
            paths, links, A, path_frame = load_operator(args.release / "corridors" / panel / "network")
            root = args.release / "task4" / panel / split
            count = pd.read_csv(root / "synthetic_link_counts.csv").set_index("link_id")
            prior = pd.read_csv(root / "synthetic_weak_prior.csv").set_index("path_id").loc[paths]
            measured = np.array([v in count.index for v in links])
            assert measured.any()
            observed = count.loc[np.asarray(links)[measured], "count"].to_numpy(dtype=np.float64)
            base = prior.path_flow.to_numpy(dtype=np.float64)
            flow, metrics = solve_odme(A[measured], observed, base,
                lambda **kw: emit(args.out, "odme_solve", panel=panel, split=split, **kw))
            table = pd.read_csv(root / "sample_submission_path_flow.csv", dtype=str)
            assert set(table.path_id) == set(paths)
            assert (table.departure_time == str(prior.departure_time.iloc[0])).all()
            table["path_flow"] = table.path_id.map(dict(zip(paths, flow)))
            assert table.path_flow.notna().all()
            table.to_csv(target, index=False, float_format="%.6f")
            loaded = A[measured]@flow
            link_score = max(0., 1-float(np.abs(loaded-observed).sum())/float(observed.sum()))
            prior_link_score = max(0., 1-float(np.abs(A[measured]@base-observed).sum())/float(observed.sum()))
            record = dict(panel=panel, split=split, paths=len(paths), measured_links=int(measured.sum()),
                          S_link=link_score, prior_S_link=prior_link_score, **metrics)
            records.append(record)
            save(args.out / "odme-solver-metrics.json", records)
            emit(args.out, "odme_completed", **record)
    torch.cuda.empty_cache()


def merge(args):
    tables = {}
    for task in KEYS:
        parts = sorted((args.out / "parts").glob(f"{task}-*.csv"))
        assert parts, f"No {task} predictions"
        table = pd.concat([pd.read_csv(p, dtype={k:str for k in KEYS[task]})[
            KEYS[task]+VALUES[task]] for p in parts], ignore_index=True)
        if "timestamp" in KEYS[task]:
            table["timestamp"] = pd.to_datetime(table.timestamp, utc=True, errors="raise").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        assert not table.duplicated(KEYS[task]).any(), f"Duplicate {task} keys"
        assert np.isfinite(table[VALUES[task]].to_numpy()).all()
        tables[task] = table.set_index(KEYS[task])
    target = args.out / "submission.csv"
    temporary = args.out / "submission.partial.csv"
    counts, missing = Counter(), Counter()
    written = 0
    for key in pd.read_csv(args.release / "submission_key.csv", chunksize=250000, dtype=str, keep_default_na=False):
        assert set(key.task) <= set(tables)
        out = pd.DataFrame({"submission_id": key.submission_id.astype("int64"), "task": key.task})
        for col in COLS[2:]:
            out[col] = 0.
        for task, table in tables.items():
            selected = key.task == task
            if not selected.any():
                continue
            wanted = pd.MultiIndex.from_frame(key.loc[selected, KEYS[task]])
            positions = table.index.get_indexer(wanted)
            if (positions < 0).any():
                raise ValueError(f"{task}: {(positions < 0).sum()} missing prediction keys")
            values = table.iloc[positions][VALUES[task]].to_numpy()
            assert np.isfinite(values).all() and (values >= 0).all()
            out.loc[selected, VALUES[task]] = values
        out[COLS].to_csv(temporary, mode="w" if written == 0 else "a", header=written == 0,
                        index=False, float_format="%.6f")
        written += len(key)
        for (task, split), n in key.groupby(["task", "split"]).size().items():
            counts[f"{task}/{split}"] += int(n)
        emit(args.out, "merge", rows=written)
    assert written == sum(len(t) for t in tables.values()), "Unexpected extra predictions"
    temporary.replace(target)
    save(args.out / "merge-report.json", dict(rows=written, counts=dict(counts), missing_predictions=0))


def verify(args):
    submission = args.out / "submission.csv"
    a = pd.read_csv(submission, chunksize=250000)
    b = pd.read_csv(args.release / "sample_submission.csv", chunksize=250000)
    last_id, rows, counts = None, 0, Counter()
    for actual, expected in itertools.zip_longest(a, b):
        assert actual is not None and expected is not None
        assert list(actual) == COLS and len(actual) == len(expected)
        assert np.array_equal(actual.submission_id, expected.submission_id)
        assert np.array_equal(actual.task, expected.task)
        ids = actual.submission_id.to_numpy()
        assert (np.diff(ids) > 0).all() and (last_id is None or ids[0] > last_id)
        last_id = int(ids[-1])
        values = actual[COLS[2:]].to_numpy()
        assert np.isfinite(values).all() and (values >= 0).all()
        for task in KEYS:
            mask = actual.task == task
            unused = [c for c in COLS[2:] if c not in VALUES[task]]
            assert (actual.loc[mask, unused].to_numpy() == 0).all()
            counts[task] += int(mask.sum())
        assert actual.loc[actual.task == "queue", "queue_pred"].isin([0, 1]).all()
        rows += len(actual)
        emit(args.out, "verify", rows=rows)
    report = dict(status="passed", rows=rows, tasks=dict(counts), columns=COLS,
                  exact_template_order=True, missing_predictions=0, duplicate_ids=0,
                  nonfinite_values=0, negative_values=0, queue_binary=True, unused_fields_zero=True,
                  sha256=digest(submission), bytes=submission.stat().st_size,
                  note="Local structural validation only. No Kaggle upload or hidden-label score.")
    save(args.out / "submission-validation.json", report)
    emit(args.out, "complete", **report)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--official", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stage", choices=["all", "state", "aux", "merge", "verify"], default="all")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "parts").mkdir(exist_ok=True)
    torch.set_num_threads(4)
    panels = [v["corridor_id"] for v in json.loads((args.release / "config/corridors.json").read_text())["panels"]]
    if args.stage in ["all", "state"]:
        build_state(args, panels)
    if args.stage in ["all", "aux"]:
        build_auxiliary(args, panels)
    if args.stage in ["all", "merge"]:
        merge(args)
    if args.stage in ["all", "verify"]:
        verify(args)


if __name__ == "__main__":
    main()
