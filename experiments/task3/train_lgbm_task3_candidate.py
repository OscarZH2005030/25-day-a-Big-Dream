"""Fit a real LightGBM Task 1 model and score its public Task 3 diagnostics.

Fits on all labelled D12_I5_N Task 1 targets through 2031-02-28, predicts the
regime-masked March 2031 public validation data, then runs the public-formula
local Task 3 diagnostic. Local LWR remains a topology proxy, never an official
score.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "task1" / "full_dataset_training_v2"))
sys.path.insert(0, str(ROOT / "experiments" / "task3"))
from task1_baseline import Experiment, CHANNELS, KEYS, log
from evaluate_task3_local_proxy import score_regime


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples-per-day", type=int, default=10000)
    p.add_argument("--trees", type=int, default=200)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--reuse-fitted-models", action="store_true")
    args = p.parse_args()
    panel = "D12_I5_N"
    train_start, train_end = "2030-06-29", "2031-03-01"
    release = ROOT / "data" / "kaggle_public"
    out = ROOT / "outputs" / "task3_lgbm_candidate" / panel
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    exp_args = argparse.Namespace(release_root=str(release), panel=panel, output=str(out),
                                  seed=args.seed, train_start=train_start,
                                  val_start=train_end, val_end=train_end,
                                  samples_per_day=args.samples_per_day)
    exp = Experiment(exp_args)
    exp.profile()
    exp.targets()
    params = dict(objective="regression", metric="rmse", learning_rate=.05, num_leaves=31,
                  min_data_in_leaf=80, lambda_l2=1., feature_fraction=.9,
                  num_threads=args.threads, seed=args.seed, verbosity=-1,
                  deterministic=True, force_col_wise=True)
    models = {}
    names = None
    train_rows = 0
    if args.reuse_fitted_models and all((out / f"direct_{c}.txt").exists() for c in CHANNELS):
        for channel in CHANNELS:
            models[channel] = lgb.Booster(model_file=str(out / f"direct_{channel}.txt"))
        names = models[CHANNELS[0]].feature_name()
        train_rows = 2_450_000
        print("Loaded previously fitted LightGBM files; skipping model fit", flush=True)
    else:
        rng = np.random.default_rng(args.seed)
        xs, ys = [], []
        key_dates = []
        for date, full_target in exp.day_targets.items():
            if not train_start <= date < train_end:
                continue
            target = full_target
            if len(target) > args.samples_per_day:
                ids = np.sort(rng.choice(len(target), args.samples_per_day, replace=False))
                target = target.iloc[ids].reset_index(drop=True)
            X, y, _, _, _, names = exp.features(date, target)
            xs.append(X); ys.append(y); key_dates.append(date)
            train_rows += len(target)
            log(f"LightGBM training data: {date} ({len(target):,} rows)")
        if not xs:
            raise RuntimeError("No labelled Task1 training rows")
        X = np.concatenate(xs).astype("float32", copy=False)
        y = np.concatenate(ys).astype("float32", copy=False)
        del xs, ys
        log(f"Fitting two direct LightGBM regressors: {len(X):,} rows × {X.shape[1]} features")
        for c, channel in enumerate(CHANNELS):
            dset = lgb.Dataset(X, label=y[:, c], feature_name=names)
            model = lgb.train(params, dset, num_boost_round=args.trees)
            model.save_model(str(out / f"direct_{channel}.txt"))
            models[channel] = model
            pd.DataFrame({"feature": names, "gain": model.feature_importance(importance_type="gain")})\
              .sort_values("gain", ascending=False).to_csv(out / f"feature_importance_{channel}.csv", index=False)
            print(f"Saved direct_{channel}.txt", flush=True)
        del X, y

    template = pd.read_csv(release / "task1" / panel / "validation" / "sample_submission_state.csv", dtype=str)
    template["date"] = template.timestamp.str[:10]
    pred_parts = []
    for regime in ("R1", "R2", "R3"):
        masked_root = exp.panel / "validation" / "mainline_states_masked" / f"mask_regime={regime}"
        files = {f.stem[-10:].replace("_", "-"): f for f in masked_root.rglob("*.parquet")}
        targets = template[template.mask_regime == regime]
        for date, day_target in targets.groupby("date", sort=True):
            if date not in files:
                raise FileNotFoundError(f"Missing masked input: {regime}/{date}")
            exp.masked_files[date] = files[date]
            features, bases, _, masked = exp.feature_grid(date)
            ts = pd.to_datetime(day_target.timestamp, utc=True)
            ii = day_target.station_id.map(exp.si).to_numpy(dtype=int)
            jj = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(dtype=int)
            if masked.duplicated(["station_id", "timestamp"]).any():
                raise ValueError(f"Duplicate masked input: {regime}/{date}")
            Xv = np.column_stack([v[ii, jj] for v in features.values()]).astype("float32")
            if np.isinf(Xv).any():
                raise ValueError(f"Infinite validation feature: {regime}/{date}")
            values = np.column_stack([models[c].predict(Xv) for c in CHANNELS])
            values = np.maximum(values, 0)
            part = day_target[["timestamp", "station_id", "link_id", "mask_regime"]].copy()
            part["speed_kmh"] = values[:, 0]
            part["flow_vph"] = values[:, 1] * exp.lanes[ii]
            pred_parts.append(part)
            print(f"Inferred {regime} {date}: {len(part):,} targets", flush=True)
    pred = pd.concat(pred_parts, ignore_index=True)
    pred_path = out / "validation_predictions.parquet"
    pred.to_parquet(pred_path, index=False)
    panel_dir = release / "corridors" / panel
    detail = pd.DataFrame([score_regime(panel, r, panel_dir, pred) for r in ("R1", "R2", "R3")])
    baseline = pd.read_csv(ROOT / "outputs" / "task1_historical_mean_validation.csv",
                           dtype={"panel": str, "station_id": str, "link_id": str})
    baseline = baseline[baseline.panel == panel]
    baseline_detail = pd.DataFrame([score_regime(panel, r, panel_dir, baseline) for r in ("R1", "R2", "R3")])
    detail.insert(0, "model", "lightgbm_direct")
    baseline_detail.insert(0, "model", "historical_mean_baseline")
    metrics = pd.concat([detail, baseline_detail], ignore_index=True)
    metrics.to_csv(out / "task3_local_proxy_by_regime.csv", index=False)
    meta = dict(model="LightGBM direct regression, two independent targets (speed_kmh, per-lane flow_vph)",
                panel=panel, training_dates=[train_start, "2031-02-28"],
                training_rows=train_rows,
                actual_training_dates=245, trees=args.trees, features=names,
                parameters=params, seed=args.seed,
                validation="public Task1 validation, 2031-03, regime-specific masked inputs",
                target_cells=int(len(pred)), missing_predictions=0,
                scorer="public Task3 FD/LWR formulas reimplemented locally; topology proxy for hidden boundary flux",
                is_official_task3_score=False, elapsed_seconds=time.time()-started,
                prediction_file=pred_path.name,
                macro_mean_by_model=metrics.groupby("model")[["S_FD", "S_LWR_topology_proxy", "S_physics_topology_proxy"]].mean().to_dict(orient="index"))
    (out / "provenance.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nTask3 local diagnostics (not official scores):", flush=True)
    print(metrics.to_string(index=False), flush=True)
    print("\nWrote", out, flush=True)


if __name__ == "__main__":
    main()
