"""Train a window-grouped LightGBM queue forecast candidate for Task 2.

Training labels are a proxy derived from the released unmasked train speeds.
The organizer's queue truth is latent and is not released, so internal proxy
scores must never be reported as Kaggle scores.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "data" / "kaggle_public"
OUT = ROOT / "outputs" / "task2_lgbm_candidate"
BASELINE = ROOT / "outputs" / "task2_persistence_train.csv"
PANELS = sorted(p.name for p in (RELEASE / "task2").iterdir() if p.is_dir())
FEATURE_KEYS = ["window_id", "link_id"]
STATIC = ["lanes", "length_km", "free_speed_kmh", "capacity_vph", "order_index",
          "critical_density", "k_jam", "on_ramp_attached", "off_ramp_attached"]


def read_split(split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indices, templates, histories = [], [], []
    for panel in PANELS:
        base = RELEASE / "task2" / panel / split
        if not (base / "window_index.csv").exists():
            continue
        idx = pd.read_csv(base / "window_index.csv", dtype={"window_id": str})
        tmp = pd.read_csv(base / "sample_submission_queue.csv",
                          dtype={"window_id": str, "link_id": str})
        hist = pd.read_parquet(base / "window_history.parquet")
        idx["panel"] = panel
        tmp["panel"] = panel
        hist["panel"] = panel
        indices.append(idx)
        templates.append(tmp)
        histories.append(hist)
    if not indices:
        raise FileNotFoundError(f"No Task 2 files for {split}")
    return tuple(pd.concat(items, ignore_index=True) for items in (indices, templates, histories))


def static_by_panel(panel: str) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, list[str]]]:
    panel_root = RELEASE / "corridors" / panel
    links = pd.read_csv(panel_root / "network" / "links.csv", dtype={"link_id": str})
    fd = pd.read_csv(panel_root / "network" / "fd_parameters.csv", dtype={"link_id": str})
    keep_fd = [c for c in ("link_id", "critical_density", "k_jam", "v_cut") if c in fd.columns]
    static = links.merge(fd[keep_fd], on="link_id", how="left", suffixes=("", "_fd"))
    static["critical_density"] = static.get("critical_density", pd.Series(np.nan, index=static.index))
    static["k_jam"] = static.get("k_jam", pd.Series(np.nan, index=static.index))
    topology_path = panel_root / "network" / "lwr_mainline_topology.csv"
    topo = pd.read_csv(topology_path, dtype=str)
    mainline = set(static.link_id.astype(str))
    up: dict[str, list[str]] = {}
    down: dict[str, list[str]] = {}
    for row in topo.itertuples(index=False):
        link = str(getattr(row, "mainline_link_id", getattr(row, "link_id", "")))
        incoming = [x for x in str(row.incoming_link_ids or "").split(";") if x in mainline]
        outgoing = [x for x in str(row.outgoing_link_ids or "").split(";") if x in mainline]
        if link in mainline:
            up[link] = incoming
            down[link] = outgoing
    topo["order_index"] = pd.to_numeric(topo.order_index, errors="coerce")
    order = topo.drop_duplicates("mainline_link_id").set_index("mainline_link_id").order_index
    static["order_index"] = static.link_id.map(order).fillna(-1)
    ramps = topo.drop_duplicates("mainline_link_id").set_index(topo.drop_duplicates("mainline_link_id").mainline_link_id.astype(str))
    static["on_ramp_attached"] = static.link_id.map(ramps.on_ramp_link_ids.fillna("").astype(str).str.len().gt(0)).fillna(False).astype(int)
    static["off_ramp_attached"] = static.link_id.map(ramps.off_ramp_link_ids.fillna("").astype(str).str.len().gt(0)).fillna(False).astype(int)
    return static, up, down


def history_features(histories: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    h = histories.copy()
    h["window_id"] = h.window_id.astype(str)
    h["link_id"] = h.link_id.astype(str)
    h["timestamp"] = pd.to_datetime(h.timestamp, utc=True)
    origin = index.set_index("window_id").forecast_origin
    origin = pd.to_datetime(origin, utc=True)
    h["origin"] = h.window_id.map(origin)
    h["lag"] = ((h.origin - h.timestamp).dt.total_seconds() / 300).round().astype(int)
    h = h[h.lag.between(1, 12)].copy()
    h["is_score_eligible"] = h.is_score_eligible.fillna(0).astype(bool)
    h["speed_ratio"] = pd.to_numeric(h.speed_kmh, errors="coerce") / h.free_speed_kmh
    h.loc[~h.is_score_eligible, "speed_ratio"] = np.nan
    h["queue"] = (h.speed_ratio <= 0.60).astype(float)
    h.loc[h.speed_ratio.isna(), "queue"] = np.nan
    h["flow_util"] = pd.to_numeric(h.flow_vph, errors="coerce") / h.capacity_vph
    h.loc[~h.is_score_eligible, "flow_util"] = np.nan
    h["occupancy"] = pd.to_numeric(h.occupancy, errors="coerce")
    h.loc[~h.is_score_eligible, "occupancy"] = np.nan
    h["eligible"] = h.is_score_eligible.astype(float)
    frames = []
    for value in ("speed_ratio", "flow_util", "occupancy", "queue", "eligible"):
        wide = h.pivot(index=FEATURE_KEYS, columns="lag", values=value)
        wide.columns = [f"{value}_lag_{int(c)}" for c in wide.columns]
        frames.append(wide)
    result = pd.concat(frames, axis=1).reset_index()
    # Add regular columns even if one lag is absent in a corridor file.
    for value in ("speed_ratio", "flow_util", "occupancy", "queue", "eligible"):
        for lag in range(1, 13):
            col = f"{value}_lag_{lag}"
            if col not in result:
                result[col] = np.nan
    speed_cols = [f"speed_ratio_lag_{i}" for i in range(1, 13)]
    flow_cols = [f"flow_util_lag_{i}" for i in range(1, 13)]
    queue_cols = [f"queue_lag_{i}" for i in range(1, 13)]
    eligible_cols = [f"eligible_lag_{i}" for i in range(1, 13)]
    for window in (3, 6, 12):
        result[f"speed_mean_{window}"] = result[speed_cols[:window]].mean(axis=1)
        result[f"speed_min_{window}"] = result[speed_cols[:window]].min(axis=1)
        result[f"flow_mean_{window}"] = result[flow_cols[:window]].mean(axis=1)
        result[f"queue_count_{window}"] = result[queue_cols[:window]].sum(axis=1)
        result[f"eligible_count_{window}"] = result[eligible_cols[:window]].sum(axis=1)
    result["speed_change_15m"] = result.speed_ratio_lag_1 - result.speed_ratio_lag_3
    result["speed_change_30m"] = result.speed_ratio_lag_1 - result.speed_ratio_lag_6
    result["speed_change_60m"] = result.speed_ratio_lag_1 - result.speed_ratio_lag_12
    result["queue_lag1"] = result.queue_lag_1
    return result


def neighbor_features(link_features: pd.DataFrame, panel: str, upstream: dict[str, list[str]],
                      downstream: dict[str, list[str]]) -> pd.DataFrame:
    use = ["window_id", "link_id", "speed_ratio_lag_1", "speed_mean_3", "speed_mean_6",
           "queue_lag1", "queue_count_3", "queue_count_6"]
    feature_values = link_features[use]
    pieces = []
    for direction, mapping in (("up", upstream), ("down", downstream)):
        pairs = [(window_id, target, other) for window_id in link_features.window_id.unique()
                 for target, others in mapping.items() for other in others]
        if not pairs:
            continue
        edge = pd.DataFrame(pairs, columns=["window_id", "link_id", "neighbor_link_id"])
        values = feature_values.rename(columns={"link_id": "neighbor_link_id"})
        merged = edge.merge(values, on=["window_id", "neighbor_link_id"], how="left")
        agg_cols = use[2:]
        agg = merged.groupby(["window_id", "link_id"], as_index=False)[agg_cols].mean()
        agg = agg.rename(columns={c: f"{direction}_{c}" for c in agg_cols})
        pieces.append(agg)
    if not pieces:
        return link_features[FEATURE_KEYS].drop_duplicates()
    out = pieces[0]
    for p in pieces[1:]:
        out = out.merge(p, on=FEATURE_KEYS, how="outer")
    return out


def build_split(split: str, with_labels: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    index, template, history = read_split(split)
    index["window_id"] = index.window_id.astype(str)
    template["window_id"] = template.window_id.astype(str)
    template["link_id"] = template.link_id.astype(str)
    history["window_id"] = history.window_id.astype(str)
    all_features, static_by_panel_cache, neighbors_by_panel_cache = [], {}, []
    for panel in sorted(index.panel.unique()):
        idx = index[index.panel == panel].copy()
        h = history[history.panel == panel].copy()
        static, up, down = static_by_panel(panel)
        static_by_panel_cache[panel] = static
        h = h.merge(static[["link_id", "free_speed_kmh", "capacity_vph"]], on="link_id", how="left", validate="many_to_one")
        hist = history_features(h, idx)
        neigh = neighbor_features(hist, panel, up, down)
        targets = template[template.panel == panel].copy()
        targets = targets.merge(idx[["window_id", "date", "condition", "forecast_origin", "forecast_start"]],
                                on="window_id", how="left", validate="many_to_one")
        targets = targets.merge(static, on="link_id", how="left", validate="many_to_one")
        targets = targets.merge(hist, on=FEATURE_KEYS, how="left", validate="many_to_one")
        targets = targets.merge(neigh, on=FEATURE_KEYS, how="left", validate="many_to_one")
        targets["timestamp_dt"] = pd.to_datetime(targets.timestamp, utc=True)
        targets["forecast_origin_dt"] = pd.to_datetime(targets.forecast_origin, utc=True)
        targets["forecast_start_dt"] = pd.to_datetime(targets.forecast_start, utc=True)
        targets["horizon_step"] = ((targets.timestamp_dt - targets.forecast_start_dt).dt.total_seconds() / 300).round().astype(int) + 1
        targets["condition_code"] = targets.condition.map({"queue_onset": 0, "queue_ongoing": 1}).astype(int)
        targets["origin_hour"] = targets.forecast_origin_dt.dt.hour
        targets["origin_weekday"] = targets.forecast_origin_dt.dt.dayofweek
        targets["origin_hour_sin"] = np.sin(2 * np.pi * targets.origin_hour / 24)
        targets["origin_hour_cos"] = np.cos(2 * np.pi * targets.origin_hour / 24)
        targets["target_speed_cut"] = 0.60 * targets.free_speed_kmh
        all_features.append(targets)
    frame = pd.concat(all_features, ignore_index=True)
    for col in frame.select_dtypes(include=["bool"]).columns:
        frame[col] = frame[col].astype(int)
    frame = frame.replace([np.inf, -np.inf], np.nan)

    # Add real training-set proxy labels only from the unmasked train mainline.
    if with_labels:
        labels = []
        for panel in sorted(frame.panel.unique()):
            rows = frame[frame.panel == panel]
            times = set(rows.timestamp.astype(str))
            links = set(rows.link_id.astype(str))
            dates = {t[:10] for t in times}
            panel_root = RELEASE / "corridors" / panel
            paths = [p for p in (panel_root / "train" / "mainline_states").rglob("*.parquet")
                     if p.stem.removeprefix("synthetic_mainline_").replace("_", "-") in dates]
            chunks = []
            for path in paths:
                state = pd.read_parquet(path, columns=["timestamp", "link_id", "speed_kmh", "is_score_eligible"])
                state["link_id"] = state.link_id.astype(str)
                state = state[state.timestamp.isin(times) & state.link_id.isin(links)]
                if not state.empty:
                    chunks.append(state)
            state = pd.concat(chunks, ignore_index=True)
            if state.duplicated(["timestamp", "link_id"]).any():
                raise ValueError(f"Multiple state labels per link/time for {panel}")
            state["panel"] = panel
            labels.append(state)
        truth = pd.concat(labels, ignore_index=True)
        frame = frame.merge(truth, on=["panel", "timestamp", "link_id"], how="left", validate="many_to_one")
        frame["label_eligible"] = frame.is_score_eligible.fillna(0).astype(bool) & frame.speed_kmh.notna()
        frame["label_queue"] = (frame.speed_kmh <= frame.target_speed_cut).astype(int)
    return frame, template


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"window_id", "panel", "family_id", "split", "date", "forecast_origin", "history_start",
                "history_end", "forecast_start", "forecast_end", "condition", "history_coverage",
                "future_coverage", "timestamp", "link_id", "queue_pred", "timestamp_dt", "forecast_origin_dt",
                "forecast_start_dt", "speed_kmh", "flow_vph", "occupancy", "pct_observed", "is_score_eligible",
                "label_eligible", "label_queue", "target_speed_cut", "on_ramp_link_ids", "off_ramp_link_ids",
                "incoming_link_ids", "outgoing_link_ids", "incoming_has_sensor", "outgoing_has_sensor",
                "from_node", "to_node", "detector_id", "v_cut"}
    return [c for c in frame.columns if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])]


def fit_models(train: pd.DataFrame, features: list[str], rounds: int = 250) -> dict[int, lgb.Booster]:
    params = dict(objective="binary", metric="binary_logloss", learning_rate=0.035, num_leaves=15,
                  min_data_in_leaf=80, feature_fraction=0.9, lambda_l2=2.0, verbosity=-1,
                  num_threads=4, seed=2026, deterministic=True, force_col_wise=True)
    models = {}
    for condition in (0, 1):
        part = train[(train.condition_code == condition) & train.label_eligible]
        if part.label_queue.nunique() < 2:
            raise ValueError(f"Condition {condition} training labels do not contain both classes")
        dset = lgb.Dataset(part[features].fillna(-1), label=part.label_queue.astype(int), feature_name=features)
        models[condition] = lgb.train(params, dset, num_boost_round=rounds)
    return models


def predict(models: dict[int, lgb.Booster], frame: pd.DataFrame, features: list[str], threshold: float = 0.5) -> pd.DataFrame:
    probs = np.zeros(len(frame), dtype=float)
    for condition in (0, 1):
        mask = frame.condition_code.to_numpy() == condition
        probs[mask] = models[condition].predict(frame.loc[mask, features].fillna(-1), num_threads=4)
    out = frame[["window_id", "timestamp", "link_id", "panel", "condition"]].copy()
    out["queue_probability"] = probs
    out["queue_pred"] = (probs >= threshold).astype(int)
    return out


def score_windows(frame: pd.DataFrame, pred: pd.DataFrame, eligible_only: bool = True) -> pd.DataFrame:
    scored = frame[["window_id", "panel", "condition", "label_queue", "label_eligible"]].copy()
    scored["pred_queue"] = pred.queue_pred.to_numpy()
    if eligible_only:
        scored = scored[scored.label_eligible].copy()
    out = []
    for wid, group in scored.groupby("window_id", sort=True):
        truth = group.label_queue.astype(bool).to_numpy()
        predicted = group.pred_queue.astype(bool).to_numpy()
        union = np.logical_or(truth, predicted).sum()
        iou = float(np.logical_and(truth, predicted).sum() / union) if union else 1.0
        out.append({"window_id": wid, "panel": group.panel.iloc[0], "condition": group.condition.iloc[0],
                    "eligible_cells": len(group), "IoU": iou})
    return pd.DataFrame(out)


def persistence_prediction(frame: pd.DataFrame) -> pd.DataFrame:
    pred = np.zeros(len(frame), dtype=int)
    # The last released history row is T-5 (lag 1).
    for idx, row in frame.iterrows():
        q = row.get("queue_lag1", np.nan)
        pred[idx] = int(pd.notna(q) and q >= 0.5)
    return pd.DataFrame({"queue_pred": pred})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Building train examples and train-only proxy labels...", flush=True)
    train, _ = build_split("train", with_labels=True)
    train["date_dt"] = pd.to_datetime(train.date)
    # Hold out every window on the latest date for each panel/condition; this
    # prevents a second same-day window of the same condition from leaking into fit.
    holdout_ids = set()
    for _, group in train[["window_id", "panel", "condition", "date_dt"]].drop_duplicates().groupby(["panel", "condition"]):
        latest_date = group.date_dt.max()
        holdout_ids.update(group.loc[group.date_dt.eq(latest_date), "window_id"].tolist())
    fit_rows = train[~train.window_id.isin(holdout_ids)]
    holdout = train[train.window_id.isin(holdout_ids)].copy()
    features = feature_columns(train)
    print(f"Train: {train.window_id.nunique()} windows / {len(train):,} target cells; "
          f"holdout: {len(holdout_ids)} windows; features: {len(features)}", flush=True)
    cv_models = fit_models(fit_rows, features)
    cv = predict(cv_models, holdout, features, threshold=0.5)
    cv_score = score_windows(holdout, cv)
    # Generate the identical persistence output for these held-out windows.
    base_path = BASELINE
    if not base_path.exists():
        raise FileNotFoundError(f"Run the published persistence builder first: {base_path}")
    base = pd.read_csv(base_path, dtype={"window_id": str, "link_id": str, "timestamp": str})
    holdout_keys = holdout[["window_id", "timestamp", "link_id"]].copy()
    holdout_keys["timestamp_key"] = pd.to_datetime(holdout_keys.timestamp, utc=True)
    base["timestamp_key"] = pd.to_datetime(base.timestamp, utc=True)
    base = holdout_keys.merge(base.drop(columns="timestamp"), on=["window_id", "timestamp_key", "link_id"],
                              how="left", validate="one_to_one")
    if base.queue_pred.isna().any():
        raise ValueError("Persistence baseline does not cover internal holdout windows")
    base_score = score_windows(holdout, base)
    cv_score["model"] = "LightGBM"
    base_score["model"] = "Persistence"
    cv_detail = pd.concat([base_score, cv_score], ignore_index=True)
    cv_detail.to_csv(OUT / "internal_train_proxy_by_window.csv", index=False, float_format="%.8f")
    summary = cv_detail.groupby(["model", "condition"], as_index=False).IoU.mean()
    summary.to_csv(OUT / "internal_train_proxy_summary.csv", index=False, float_format="%.8f")
    macro = summary.groupby("model").IoU.mean().to_dict()

    # Fit final candidates on all train windows; generate full validation/private files.
    print("Refitting on all train windows and predicting validation/private...", flush=True)
    models = fit_models(train, features)
    for condition, model in models.items():
        model.save_model(str(OUT / f"lightgbm_condition_{'ongoing' if condition else 'onset'}.txt"))
    importance = pd.concat([
        pd.DataFrame({"condition": "queue_ongoing" if cond else "queue_onset",
                      "feature": features, "gain": model.feature_importance(importance_type="gain")})
        for cond, model in models.items()
    ], ignore_index=True)
    importance.to_csv(OUT / "feature_importance.csv", index=False)
    for split in ("validation", "private"):
        print(f"Building {split} features...", flush=True)
        frame, _ = build_split(split, with_labels=False)
        pred = predict(models, frame, features, threshold=0.5)
        submission = pred[["window_id", "timestamp", "link_id", "queue_pred"]]
        submission.to_csv(OUT / f"queue_lgbm_{split}.csv", index=False)
        pred.to_parquet(OUT / f"queue_lgbm_{split}_probabilities.parquet", index=False)
        print(f"{split}: {len(submission):,} rows; positive rate {submission.queue_pred.mean():.4f}", flush=True)

    meta = {
        "model": "Two native LightGBM binary classifiers, separate queue_onset and queue_ongoing",
        "training_split": "train windows only; proxy labels derived from unmasked measured speed <= 0.60*free_speed",
        "training_windows": int(train.window_id.nunique()),
        "internal_holdout": "all windows on the latest date for each panel and condition",
        "internal_holdout_windows": int(len(holdout_ids)),
        "feature_count": int(len(features)), "features": features,
        "threshold": 0.5,
        "label_caveat": "observed-speed labels are not the organizer's latent-state queue_true labels",
        "metric_caveat": "internal IoU is a train-only proxy, not Kaggle leaderboard score",
        "official_validation_predictions": "candidate files only; must be scored by Kaggle leaderboard",
    }
    (OUT / "provenance.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ["# Task2 LightGBM 初版训练报告", "", "## 内部时间留出代理结果", "",
              f"使用 80 个公开 train 窗口训练（87,000 个预测单元）；按面板与窗口条件留出日期最新的全部窗口，共 {len(holdout_ids)} 个（{int((cv_score.condition == 'queue_onset').sum())} onset、{int((cv_score.condition == 'queue_ongoing').sum())} ongoing）。训练标签由未遮挡速度按 `speed <= 0.60 × free_speed` 派生；不等于组织方隐藏的潜在状态标签。指标是内部代理 IoU，不是 Kaggle 官方成绩。", "",
              f"按 onset/ongoing 等权平均，LightGBM 代理 IoU 为 {macro['LightGBM']:.6f}，Persistence 为 {macro['Persistence']:.6f}。", "",
              "| 方法 | 条件 | 窗口 IoU 均值 |", "|---|---|---:|"]
    for row in summary.itertuples(index=False):
        report.append(f"| {row.model} | {row.condition} | {row.IoU:.6f} |")
    report += ["", "## 解释与限制", "",
               "- 两个 LightGBM 分类器分别学习 onset 与 ongoing；特征包括 60 分钟速度/流量历史、队列状态、质量标记、上下游状态、道路属性、预测步长和起报时间。",
               "- 内部验证按完整窗口切分，避免同一个窗口的未来样本同时出现在训练与验证中。",
               "- 窗口数少且代理标签来自有测量噪声的观测速度，holdout IoU 只用于初筛。最终判断看 Kaggle validation leaderboard。",
               "- LightGBM 阈值固定 0.5；后续若调阈值，应只用训练窗口内部交叉验证，不能看 official validation 标签。",
               "", "验证与 private 候选预测保存在 `queue_lgbm_validation.csv` 和 `queue_lgbm_private.csv`。"]
    (OUT / "Task2_LightGBM_初版报告_CN.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("Internal proxy summary:")
    print(summary.to_string(index=False))
    print(f"Wrote candidates to {OUT}")


if __name__ == "__main__":
    main()
