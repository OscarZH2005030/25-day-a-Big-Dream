"""Compare Task 1 accuracy and state-physics diagnostics on identical targets.

This is a chronological D12_I5_N train holdout analysis. Its FD metric applies
the public FD formula only to masked target cells; Delta-N compares adjacent
five-minute target pairs. Neither is the official Task 3 score (which needs
organizer boundary flows).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "task1" / "full_dataset_training_v2"))
from task1_baseline import Experiment, KEYS  # noqa: E402


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x - y))))


def fd_score(speed: np.ndarray, flow: np.ndarray, params: pd.DataFrame) -> tuple[float, float]:
    lanes = params.lanes.to_numpy(float)
    q = flow / lanes
    k = flow / np.maximum(speed, 1.0) / lanes
    vf = params.free_speed_kmh.to_numpy(float)
    cap = params.capacity_vph.to_numpy(float) / lanes
    kc = params.critical_density.to_numpy(float) / lanes
    kj = params.k_jam.to_numpy(float) / lanes
    q_fd = np.where(k <= kc, vf * k,
                    cap / np.maximum(kj - kc, 1e-9) * np.maximum(kj - k, 0.0))
    error_ratio = float(np.abs(q - q_fd).sum() / (np.abs(q).sum() + 1e-9))
    return max(0.0, 1.0 - error_ratio), float(np.abs(q - q_fd).mean())


def delta_n_rmse(frame: pd.DataFrame, model: str, pairs: pd.DataFrame) -> float:
    n = frame[model].to_numpy(float)
    next_n = pd.Series(n).groupby([frame.station_id, frame.link_id], sort=False).shift(-1).to_numpy()
    d_n = next_n - n
    values = []
    for regime in ("R1", "R2", "R3"):
        valid = pairs.pair_ok.to_numpy() & (pairs.mask_regime.to_numpy() == regime)
        values.append(rmse(d_n[valid], pairs.truth_delta_n.to_numpy()[valid]))
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=ROOT / "data" / "kaggle_public")
    parser.add_argument("--prediction-root", type=Path, default=ROOT / "outputs" / "task1_lgbm_d12_i5_n_native")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "task1_state_physics_comparison")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    exp_args = argparse.Namespace(
        release_root=str(args.release_root), panel="D12_I5_N", output=str(args.output_root / "working"),
        seed=2026, train_start="2030-06-29", val_start="2031-01-05", val_end="2031-03-01",
        samples_per_day=10000,
    )
    exp = Experiment(exp_args)
    exp.profile()
    exp.targets()
    (valid, keys) = exp.collect(validation=True)
    _, truth, interpolation, historical, lanes, regimes = valid
    keys = keys.reset_index(drop=True)
    expected_keys = keys[KEYS].astype(str).reset_index(drop=True)
    predictions: dict[str, np.ndarray] = {
        "历史星期/时刻均值": np.column_stack([historical[:, 0], historical[:, 1] * lanes]),
        "时间插值": np.column_stack([interpolation[:, 0], interpolation[:, 1] * lanes]),
    }

    keycols = KEYS
    for stem, name in (("direct", "LightGBM直接"), ("residual", "LightGBM残差")):
        f = pd.read_parquet(args.prediction_root / f"{stem}_holdout_predictions.parquet")
        got = f[keycols].astype(str).reset_index(drop=True)
        aligned = f.merge(keys[keycols].assign(_order=np.arange(len(keys))), on=keycols,
                          how="inner", validate="one_to_one").sort_values("_order")
        if len(aligned) != len(keys) or not aligned[keycols].astype(str).reset_index(drop=True).equals(expected_keys):
            raise ValueError(f"{name}: prediction keys do not exactly match the Task 1 holdout")
        predictions[name] = aligned[["speed_kmh", "flow_vph"]].to_numpy(float)

    if len(keys) != len(truth) or not np.array_equal(regimes, keys.mask_regime.to_numpy()):
        raise ValueError("Holdout labels/key alignment check failed")
    true_speed = truth[:, 0]
    true_flow = truth[:, 1] * lanes
    params = exp.static.reset_index().rename(columns={"index": "link_id"})
    if "link_id" not in params.columns:
        params = params.rename(columns={params.columns[0]: "link_id"})
    metrics = []
    point = keys[["timestamp", "station_id", "link_id", "mask_regime"]].copy()
    point["timestamp"] = pd.to_datetime(point.timestamp, utc=True)
    point["truth_speed"] = true_speed
    point["truth_flow"] = true_flow
    point = point.merge(params[["link_id", "length_km", "lanes", "free_speed_kmh", "capacity_vph", "critical_density", "k_jam"]],
                        on="link_id", how="left", validate="many_to_one")
    if point[["length_km", "lanes", "free_speed_kmh", "capacity_vph", "critical_density", "k_jam"]].isna().any().any():
        raise ValueError("Missing FD/link parameters")
    point["truth_N"] = point.truth_flow / np.maximum(point.truth_speed, 1.0) * point.length_km

    # Each model is scored on the same row set. N is q/v times link length.
    for name, pred in predictions.items():
        speed, flow = pred[:, 0], pred[:, 1]
        pred_flow_per_lane = flow / lanes
        true_flow_per_lane = truth[:, 1]
        fd_by_regime = [fd_score(speed[regimes == r], flow[regimes == r], point.loc[regimes == r])
                        for r in ("R1", "R2", "R3")]
        model_col = f"N_{name}"
        point[model_col] = flow / np.maximum(speed, 1.0) * point.length_km.to_numpy()
        metrics.append({
            "model": name,
            "n_Task1_targets": int(len(keys)),
            "speed_RMSE_km_h": float(np.mean([rmse(speed[regimes == r], true_speed[regimes == r]) for r in ("R1", "R2", "R3")])),
            "flow_RMSE_vph_lane": float(np.mean([rmse(pred_flow_per_lane[regimes == r], true_flow_per_lane[regimes == r]) for r in ("R1", "R2", "R3")])),
            "S_state": float(np.mean([
                .54 * max(0.0, 1.0 - rmse(speed[regimes == r], true_speed[regimes == r]) / 25.0)
                + .46 * max(0.0, 1.0 - rmse(pred_flow_per_lane[regimes == r], true_flow_per_lane[regimes == r]) / 600.0)
                for r in ("R1", "R2", "R3")
            ])),
            "FD_score_on_targets": float(np.mean([x[0] for x in fd_by_regime])),
            "FD_MAE_vph_lane": float(np.mean([x[1] for x in fd_by_regime])),
        })

    # Restrict every method to target-to-target adjacent five-minute pairs.
    point = point.sort_values(["station_id", "link_id", "timestamp"]).reset_index(drop=True)
    g = point.groupby(["station_id", "link_id"], sort=False)
    point["next_timestamp"] = g.timestamp.shift(-1)
    point["next_regime"] = g.mask_regime.shift(-1)
    point["truth_delta_n"] = g.truth_N.shift(-1) - point.truth_N
    point["pair_ok"] = (((point.next_timestamp - point.timestamp) == pd.Timedelta(minutes=5))
                        & (point.next_regime == point.mask_regime))
    pair_count = int(point.pair_ok.sum())
    if pair_count == 0:
        raise ValueError("No adjacent target pairs found")
    for row in metrics:
        model = row["model"]
        row["deltaN_RMSE_veh_per_5min"] = delta_n_rmse(point, f"N_{model}", point)
        row["n_adjacent_pairs"] = pair_count

    result = pd.DataFrame(metrics)[[
        "model", "n_Task1_targets", "speed_RMSE_km_h", "flow_RMSE_vph_lane", "S_state",
        "deltaN_RMSE_veh_per_5min", "n_adjacent_pairs", "FD_score_on_targets", "FD_MAE_vph_lane",
    ]]
    result.to_csv(args.output_root / "same_sample_state_physics_metrics.csv", index=False, float_format="%.8f")

    lines = [
        "# 同一留出样本上的状态精度与物理诊断",
        "",
        "## 实验口径",
        "",
        "- 走廊：D12_I5_N；时间留出：2031-01-05 至 2031-02-28。",
        f"- 所有方法使用完全相同的 {len(keys):,} 个 Task 1 模板目标；车辆积累变化比较仅使用相邻 5 分钟、两端都属于该目标集的 {pair_count:,} 对。",
        "- `N = (flow_vph / speed_kmh) × length_km`；`ΔN` 为相邻 5 分钟 N 的差值，误差单位为 veh/5 min。",
        "- FD 使用公开三角基本图公式，只在同一批预测目标点上算诊断分；不是官方 Task3 分数。Task3 官方守恒项仍需组织方边界流量。",
        "- 流量误差按车道数归一化；Task1 RMSE、ΔN RMSE 和 FD 分数都先按 R1/R2/R3 分别算，再等权平均。",
        "",
        "## 结果",
        "",
        "| 方法 | 目标数 | 速度 RMSE (km/h) ↓ | 流量 RMSE (veh/h/lane) ↓ | S_state ↑ | ΔN RMSE (veh/5 min) ↓ | 相邻对数 | FD 目标点分 ↑ | FD MAE (veh/h/lane) ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in result.iterrows():
        lines.append(
            f"| {r.model} | {int(r.n_Task1_targets):,} | {r.speed_RMSE_km_h:.3f} | {r.flow_RMSE_vph_lane:.3f} | "
            f"{r.S_state:.6f} | {r.deltaN_RMSE_veh_per_5min:.3f} | {int(r.n_adjacent_pairs):,} | "
            f"{r.FD_score_on_targets:.6f} | {r.FD_MAE_vph_lane:.3f} |"
        )
    lines += [
        "",
        "## 如何解读",
        "",
        "- 速度/流量 RMSE 和 `S_state` 衡量重建精度；`ΔN RMSE` 衡量相邻时刻车辆积累变化是否跟真实变化接近。",
        "- `FD_score_on_targets` 衡量预测点贴近基本图的程度。FD 更好不必然意味着速度/流量更接近真实值；需结合三类误差一起判断。",
        "- 这里的 `ΔN` 仅是状态差分误差，未包含入流、出流及匝道流量，不能称作 LWR 守恒误差或官方 `S_LWR`。",
        "- 本结果仅代表 D12_I5_N 这一走廊和该时间切分，不是十走廊竞赛总分。",
        "",
        "逐项 CSV：`same_sample_state_physics_metrics.csv`。",
    ]
    (args.output_root / "同批样本_Task1与物理一致性对照_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(result.to_string(index=False))
    print(f"Wrote {args.output_root}")


if __name__ == "__main__":
    main()
