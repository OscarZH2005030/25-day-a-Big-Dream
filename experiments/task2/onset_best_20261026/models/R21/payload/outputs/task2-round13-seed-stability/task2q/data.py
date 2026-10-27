"""Causal, same-day proxy windows from public train observations.

Public training observations are noisy measurements, *not* official queue_true.
The feature allowlist excludes future coverage and any future measurement. The
first twelve frames use released train observations; optional T uses the masked
release when available, matching the official inference input boundary.
"""
from __future__ import annotations

from datetime import date as Date
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .cache import atomic_json, content_key, read_arrays, sha256_file, write_arrays
from .protocol import PROTOCOL, PROTOCOL_SHA256

SCHEMA = "task2q-proxy-v2"
PANELS = ("D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W",
          "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S")
ROLES = {role: tuple(bounds) for role, bounds in PROTOCOL["roles"].items()}
FEATURE_NAMES = ("speed_ratio", "flow_ratio", "occupancy", "quality",
                 "speed_missing", "flow_missing", "occupancy_missing", "eligible",
                 "age", "speed_margin", "speed_delta", "sin_time", "cos_time")
STATIC_NAMES = ("length_km", "lanes_div8", "free_speed_div130",
                "capacity_div12000", "threshold_ratio", "isolated")
FIELDS = {"speed": "speed_kmh", "flow": "flow_vph", "occupancy": "occupancy",
          "quality": "pct_observed", "eligible": "is_score_eligible"}


def role_for_date(value: str) -> str:
    value = Date.fromisoformat(str(value)[:10]).isoformat()
    for role, (start, end) in ROLES.items():
        if start <= value <= end:
            return role
    raise ValueError(f"Date outside frozen public-train split: {value}")


def _panel(root: str | Path, panel: str) -> Path:
    if panel not in PANELS:
        raise ValueError(f"Unsupported Task 2 panel: {panel}")
    return Path(root) / "corridors" / panel


def network(root: str | Path, panel: str) -> dict[str, np.ndarray]:
    """Only released internal directed neighbors; no fabricated CONN nodes."""
    directory = _panel(root, panel) / "network"
    links = pd.read_csv(directory / "links.csv", dtype={"link_id": str})
    if links.link_id.duplicated().any():
        raise ValueError("Duplicate links in network")
    links = links.sort_values("link_id").set_index("link_id")
    ids = links.index.to_numpy(dtype=str)
    index = {value: i for i, value in enumerate(ids)}
    free = links.free_speed_kmh.to_numpy(dtype=np.float32)
    capacity = links.capacity_vph.to_numpy(dtype=np.float32)
    length = links.length_km.to_numpy(dtype=np.float32)
    lanes = links.lanes.to_numpy(dtype=np.float32)
    if not all(np.isfinite(v).all() and (v > 0).all()
               for v in (free, capacity, length, lanes)):
        raise ValueError("Nonpositive/missing released physical link parameters")
    threshold = free * .60
    fd_path = directory / "fd_parameters.csv"
    if fd_path.exists():
        fd = pd.read_csv(fd_path, dtype={"link_id": str})
        if fd.link_id.duplicated().any():
            raise ValueError("Duplicate FD links")
        if "v_cut" in fd:
            cuts = fd.set_index("link_id").reindex(ids).v_cut.to_numpy(dtype=float)
            use = np.isfinite(cuts) & (cuts > 0)
            threshold[use] = cuts[use]
    topology = pd.read_csv(directory / "lwr_mainline_topology.csv").fillna("")
    edges = set()
    for row in topology.to_dict("records"):
        node = str(row.get("mainline_link_id", row.get("link_id", "")))
        if node not in index:
            continue
        for target in str(row.get("outgoing_link_ids", "")).split(";"):
            if target in index and target != node:
                edges.add((index[node], index[target]))
        for source in str(row.get("incoming_link_ids", "")).split(";"):
            if source in index and source != node:
                edges.add((index[source], index[node]))
    edge_index = np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2).T
    isolated = np.ones(len(ids), dtype=bool)
    isolated[edge_index.ravel()] = False
    static = np.stack((length, lanes / 8, free / 130, capacity / 12000,
                       threshold / free, isolated.astype(np.float32)), axis=-1)
    return {"link_ids": ids, "static": static.astype(np.float32),
            "edges": edge_index, "edge_index": edge_index, "threshold": threshold,
            "free_speed": free, "capacity": capacity, "length": length,
            "length_km": length, "isolated": isolated}


def _network_sources(root: str | Path, panel: str) -> dict[str, str]:
    directory = _panel(root, panel) / "network"
    names = ("links.csv", "fd_parameters.csv", "lwr_mainline_topology.csv")
    return {f"network/{name}": sha256_file(directory / name)
            for name in names if (directory / name).exists()}


def _day_path(root: str | Path, panel: str, date: str, masked: bool = False) -> Path:
    Date.fromisoformat(date)
    directory = "mainline_states_masked" if masked else "mainline_states"
    if masked:
        return _masked_path(root, panel, "train", date)
    return (_panel(root, panel) / "train" / directory / f"year_month={date[:7]}" /
            f"synthetic_mainline_{date.replace('-', '_')}.parquet")


def _masked_path(root: str | Path, panel: str, split: str, date: str) -> Path:
    directory = _panel(root, panel) / split / "mainline_states_masked"
    matches = list(directory.rglob(f"synthetic_mainline_{date.replace('-', '_')}.parquet"))
    if len(matches) > 1:
        raise ValueError(f"Ambiguous masked release date {panel}/{split}/{date}")
    return matches[0] if matches else directory / f"missing_{date}.parquet"


def _dense(path: Path, ids: np.ndarray, date: str) -> dict[str, np.ndarray]:
    frame = pd.read_parquet(path, columns=["timestamp", "link_id", *FIELDS.values()])
    expected = pd.date_range(date, periods=288, freq="5min", tz="UTC")
    return _frame_arrays(frame, ids, expected, path.name)


def _frame_arrays(frame: pd.DataFrame, ids: np.ndarray,
                  expected: pd.DatetimeIndex, source_name: str) -> dict[str, np.ndarray]:
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    if frame.duplicated(["timestamp", "link_id"]).any():
        raise ValueError(f"Duplicate time/link cells: {source_name}")
    if not frame.timestamp.isin(expected).all():
        raise ValueError(f"Unexpected timestamp: {source_name}")
    if not frame.link_id.isin(ids).all():
        raise ValueError(f"Unknown link in {source_name}")
    result = {name: frame.pivot(index="timestamp", columns="link_id", values=column)
              .reindex(index=expected, columns=ids).to_numpy(dtype=np.float32)
              for name, column in FIELDS.items()}
    result["eligible"] = np.isfinite(result["eligible"]) & (result["eligible"] > 0)
    return result


def day(root: str | Path, panel: str, date: str,
        cache_dir: str | Path | None = None) -> dict:
    """Materialize one <=352-link day, never all days simultaneously."""
    role_for_date(date)
    path = _day_path(root, panel, date)
    masked = _day_path(root, panel, date, masked=True)
    sources = _network_sources(root, panel)
    sources["train/mainline_states"] = sha256_file(path)
    if masked.exists():
        sources["train/mainline_states_masked"] = sha256_file(masked)
    key = content_key("day", sources, {"schema": SCHEMA, "panel": panel, "date": date})
    found = read_arrays(cache_dir, key) if cache_dir is not None else None
    if found:
        return {**found[0], "key": key, "cache_hit": True,
                "source_hashes": sources, "record": found[1]}
    net = network(root, panel)
    result = _dense(path, net["link_ids"], date)
    # If T's public masked release is unavailable, withhold the optional frame;
    # never silently grant the model the unmasked current observation.
    origin = (_dense(masked, net["link_ids"], date) if masked.exists() else
              {name: np.full_like(value, False if name == "eligible" else np.nan)
               for name, value in result.items()})
    result.update({f"origin_{name}": value for name, value in origin.items()})
    result["times_ns"] = pd.date_range(date, periods=288, freq="5min", tz="UTC").as_unit("ns").asi8
    record = (write_arrays(cache_dir, key, result,
                           {"sources": sources, "schema": SCHEMA})
              if cache_dir is not None else None)
    return {**result, "key": key, "cache_hit": False,
            "source_hashes": sources, "record": record}


def features(history: dict[str, np.ndarray], net: dict, times_ns: np.ndarray) -> np.ndarray:
    """Pure <=T transform; missingness and finite default values are explicit."""
    speed, flow, occupancy = (np.asarray(history[k], dtype=np.float32)
                              for k in ("speed", "flow", "occupancy"))
    valid = np.isfinite(speed)
    ratio = np.where(valid, speed / net["free_speed"], 1.)
    age = np.zeros_like(speed)
    for step in range(speed.shape[0]):
        age[step] = np.where(valid[step], 0., (age[step - 1] if step else 0.) + 1.)
    delta = np.zeros_like(speed)
    delta[1:] = np.where(valid[1:] & valid[:-1], ratio[1:] - ratio[:-1], 0.)
    minute = ((np.asarray(times_ns) // (60 * 1_000_000_000)) % 1440).astype(float)
    angle = minute * (2 * np.pi / 1440)
    repeated = lambda v: np.broadcast_to(v[:, None], speed.shape)
    arrays = (np.clip(ratio, 0, 2),
              np.clip(np.nan_to_num(flow / net["capacity"], nan=0.), 0, 4),
              np.clip(np.nan_to_num(occupancy, nan=0.), 0, 1),
              np.clip(np.nan_to_num(history["quality"], nan=0.) / 100., 0, 1),
              ~valid, ~np.isfinite(flow), ~np.isfinite(occupancy),
              np.asarray(history["eligible"], dtype=bool), np.minimum(age, 13) / 13,
              np.where(valid, net["threshold"] / net["free_speed"] - ratio, 0.),
              delta, repeated(np.sin(angle)), repeated(np.cos(angle)))
    return np.nan_to_num(np.stack(arrays, axis=-1), nan=0., posinf=4., neginf=-4.).astype(np.float32)


def infer_condition(history: dict[str, np.ndarray], net: dict) -> int:
    """Contract-like proxy condition uses only the twelve pre-T frames."""
    queue = np.isfinite(history["speed"][:12]) & (history["speed"][:12] <= net["threshold"])
    queue &= np.asarray(history["eligible"][:12], dtype=bool)
    return int(np.any(queue.sum(axis=0) >= 2))


def make_sample(values: dict, net: dict, origin_slot: int, panel: str,
                role: str | None = None) -> dict:
    if not 12 <= origin_slot <= 281:
        raise ValueError("A same-day window needs T-60 and T+30 within the day")
    origin = pd.Timestamp(int(values["times_ns"][origin_slot]), tz="UTC")
    actual_role = role_for_date(origin.strftime("%Y-%m-%d"))
    if role is not None and role != actual_role:
        raise ValueError("Role disagrees with frozen date split")
    history = {name: np.concatenate((values[name][origin_slot-12:origin_slot],
                                     values[f"origin_{name}"][origin_slot:origin_slot+1]))
               for name in FIELDS}
    future_speed = values["speed"][origin_slot+1:origin_slot+7]
    future_quality = values["quality"][origin_slot+1:origin_slot+7]
    m = (values["eligible"][origin_slot+1:origin_slot+7].astype(bool)
         & np.isfinite(future_speed) & np.isfinite(future_quality) & (future_quality >= 75))
    y = np.where(m, future_speed <= net["threshold"], False).astype(np.float32)
    soft = 1 / (1 + np.exp(-np.clip((net["threshold"] - future_speed) / 5., -30, 30)))
    soft = np.where(m, soft, 0).astype(np.float32)
    result = {"x": features(history, net, values["times_ns"][origin_slot-12:origin_slot+1]),
              "y": y, "m": m, "label_mask": m, "soft_y": soft,
              "speed_y": np.where(m, future_speed / net["free_speed"], 0).astype(np.float32),
              "condition": infer_condition(history, net), "panel": panel,
              "origin": origin.isoformat().replace("+00:00", "Z"), "role": actual_role,
              "has_future_queue": bool(y[m].sum() > 0), "label_coverage": float(m.mean()),
              "network": net, "feature_names": FEATURE_NAMES,
              "label_kind": "noisy_observation_proxy_not_queue_true"}
    return result


def _balanced_pick(candidates: list[dict], maximum: int | None) -> list[dict]:
    """Deterministic date-local strata, preserving natural negatives explicitly."""
    if maximum is None or len(candidates) <= maximum:
        return candidates
    groups = {}
    for entry in candidates:
        groups.setdefault((entry["has_future_queue"], entry["condition"], entry["date"]), []).append(entry)
    selected = []
    # Future-positive windows first, then auxiliary negatives. No future data is
    # a feature: these flags only define label-side sampling/metric populations.
    ordered = [groups[k] for k in sorted(groups, reverse=True)]
    turns = max(1, int(np.ceil(maximum / len(ordered))))
    ordered = [[group[i] for i in np.linspace(0, len(group)-1, min(turns, len(group)), dtype=int)]
               for group in ordered]
    cursor = 0
    while len(selected) < maximum:
        advanced = False
        for group in ordered:
            if cursor < len(group):
                selected.append(group[cursor])
                advanced = True
                if len(selected) == maximum:
                    break
        if not advanced:
            break
        cursor += 1
    return sorted(selected, key=lambda item: (item["date"], item["origin_slot"]))


def prepare(root: str | Path, cache_dir: str | Path, out_dir: str | Path,
            panels: Iterable[str] = PANELS, roles: Iterable[str] = ("fit", "tune"),
            max_days: int | None = 2, max_windows: int | None = 8, stride: int = 6,
            engineering: bool = True, allow_reserved: bool = False) -> dict:
    """Build a bounded engineering index or an explicit full fit/tune index.

    max_days and max_windows are both caps per panel/role. This
    function does not fit any model. Reserved-role future labels need a separate
    explicit opt-in, so ordinary smoke cannot open the final proxy audit.
    """
    root, cache_dir, out_dir = Path(root), Path(cache_dir), Path(out_dir)
    panels, roles = tuple(panels), tuple(roles)
    if not roles or any(role not in ROLES for role in roles):
        raise ValueError("Unknown/empty frozen roles")
    if set(roles) & {"calibrate", "audit"} and not allow_reserved:
        raise ValueError("Reserved future labels blocked; use an explicit frozen calibration/audit stage")
    if stride < 1 or (max_days is not None and max_days < 1) or (max_windows is not None and max_windows < 1):
        raise ValueError("stride and sample/day caps must be positive")
    if not engineering and (max_days is not None or max_windows is not None):
        raise ValueError("Full preparation must explicitly set max_days/max_windows=None")
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": SCHEMA, "engineering": engineering, "features": list(FEATURE_NAMES),
                "static_names": list(STATIC_NAMES), "roles": {r: ROLES[r] for r in roles},
                "scope": {"panels": panels, "max_days_per_panel_role": max_days,
                          "max_windows_per_panel_role": max_windows, "stride_slots": stride,
                          "same_day_only": True, "reserved_labels_opened": bool(set(roles) & {"calibrate", "audit"})},
                "cache_relative": __import__("os").path.relpath(cache_dir.resolve(), out_dir.resolve()).replace("\\", "/"),
                "assets": {"days": {}, "networks": {}}, "samples": [], "warnings": [],
                "label_kind": "noisy_observation_proxy_not_queue_true"}
    for panel in panels:
        net = network(root, panel)
        net_sources = _network_sources(root, panel)
        net_key = content_key("network", net_sources, {"schema": SCHEMA, "panel": panel})
        found = read_arrays(cache_dir, net_key)
        net_record = found[1] if found else write_arrays(cache_dir, net_key, net, {"sources": net_sources})
        manifest["assets"]["networks"][net_key] = net_record
        all_paths = sorted((_panel(root, panel) / "train" / "mainline_states").glob("year_month=*/*.parquet"))
        for role in roles:
            selected_paths = [p for p in all_paths if ROLES[role][0] <= p.stem.removeprefix("synthetic_mainline_").replace("_", "-") <= ROLES[role][1]]
            if max_days is not None:
                # Spaced days sample beyond midnight/one event and avoid a
                # first-day-only smoke being mistaken for seasonal validation.
                idx = np.linspace(0, len(selected_paths)-1, min(max_days, len(selected_paths)), dtype=int)
                selected_paths = [selected_paths[i] for i in idx]
            if not selected_paths:
                manifest["warnings"].append(f"{panel}/{role}: no released days")
            role_candidates = []
            for path in selected_paths:
                date = path.stem.removeprefix("synthetic_mainline_").replace("_", "-")
                values = day(root, panel, date, cache_dir)
                manifest["assets"]["days"][values["key"]] = values["record"]
                candidates = []
                for slot in range(12, 282, stride):
                    sample = make_sample(values, net, slot, panel, role)
                    if not sample["m"].any():
                        continue
                    candidates.append({"id": f"proxy_{panel}_{date}_{slot:03d}", "panel": panel,
                                       "date": date, "origin_slot": slot, "origin": sample["origin"],
                                       "role": role, "condition": sample["condition"],
                                       "has_future_queue": sample["has_future_queue"],
                                       "label_coverage": sample["label_coverage"],
                                       "day_key": values["key"], "network_key": net_key})
                role_candidates.extend(candidates)
                if not any(s["has_future_queue"] for s in candidates):
                    manifest["warnings"].append(f"{panel}/{date}: no future-positive proxy window; engineering-only negatives")
            manifest["samples"].extend(_balanced_pick(role_candidates, max_windows))
    manifest["sample_count"] = len(manifest["samples"])
    manifest["primary_sample_count"] = sum(s["has_future_queue"] for s in manifest["samples"])
    manifest["frozen_split_sha256"] = hashlib.sha256(json.dumps(ROLES, sort_keys=True).encode()).hexdigest()
    manifest["protocol_sha256"] = PROTOCOL_SHA256
    atomic_json(out_dir / "index.json", manifest)
    return manifest


def load_sample(index_path: str | Path, entry: dict | int,
                cache_dir: str | Path | None = None) -> dict:
    index_path = Path(index_path)
    manifest = json.loads(index_path.read_text(encoding="utf-8"))
    if manifest["schema"] != SCHEMA or tuple(manifest["features"]) != FEATURE_NAMES:
        raise ValueError("Unknown dataset schema/features")
    if isinstance(entry, int):
        entry = manifest["samples"][entry]
    directory = Path(cache_dir) if cache_dir is not None else index_path.parent / manifest["cache_relative"]
    net_loaded = read_arrays(directory, entry["network_key"])
    day_loaded = read_arrays(directory, entry["day_key"])
    if net_loaded is None or day_loaded is None:
        raise ValueError("Missing/corrupt cache: rerun prepare from the original released inputs")
    for kind, key, loaded in (("networks", entry["network_key"], net_loaded),
                              ("days", entry["day_key"], day_loaded)):
        if loaded[1]["sha256"] != manifest["assets"][kind][key]["sha256"]:
            raise ValueError("Cache differs from the frozen sample index; rerun prepare")
    return make_sample(day_loaded[0], net_loaded[0], entry["origin_slot"], entry["panel"], entry["role"])


def released_sample(root: str | Path, panel: str, split: str, window_id: str,
                    cache_dir: str | Path | None = None) -> dict:
    """Inference sample: only published history and the masked row at T.

    Future labels are withheld for every split, including official train windows.
    cache_dir is accepted for interface parity; source loading is deliberately
    small and does not use the full-day proxy cache containing future labels.
    """
    if split not in ("train", "validation", "private"):
        raise ValueError("Unknown official split")
    net = network(root, panel)
    directory = Path(root) / "task2" / panel / split
    index = pd.read_csv(directory / "window_index.csv")
    rows = index[index.window_id == window_id]
    if len(rows) != 1:
        raise ValueError("Missing or duplicated official window ID")
    row = rows.iloc[0]
    origin = pd.Timestamp(row.forecast_origin)
    if origin.tzinfo is None:
        raise ValueError("Origin must be timezone aware")
    origin = origin.tz_convert("UTC")
    if row.condition not in ("queue_onset", "queue_ongoing"):
        raise ValueError("Unknown published condition")
    expected = pd.date_range(end=origin - pd.Timedelta(minutes=5), periods=12, freq="5min")
    frame = pd.read_parquet(directory / "window_history.parquet",
                            filters=[("window_id", "==", window_id)],
                            columns=["window_id", "timestamp", "link_id", *FIELDS.values()])
    if len(frame) != 12 * len(net["link_ids"]):
        raise ValueError("Official history must have exactly 12 frames for every link")
    history = _frame_arrays(frame, net["link_ids"], expected, "window_history")
    masked = _masked_path(root, panel, split, origin.strftime("%Y-%m-%d"))
    if masked.exists():
        current_frame = pd.read_parquet(masked,
                          filters=[("timestamp", "==", origin.isoformat().replace("+00:00", "Z"))],
                          columns=["timestamp", "link_id", *FIELDS.values()])
        current = _frame_arrays(current_frame, net["link_ids"], pd.DatetimeIndex([origin]), "masked_T")
    else:
        current = {name: np.full((1, len(net["link_ids"])),
                                False if name == "eligible" else np.nan)
                   for name in FIELDS}
    history = {name: np.concatenate([history[name], current[name]]) for name in FIELDS}
    shape = (6, len(net["link_ids"]))
    empty_labels, empty_mask = np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=bool)
    return {"x": features(history, net, expected.append(pd.DatetimeIndex([origin])).as_unit("ns").asi8),
            "y": empty_labels, "m": empty_mask, "label_mask": empty_mask,
            "soft_y": empty_labels.copy(), "speed_y": empty_labels.copy(),
            "condition": int(row.condition == "queue_ongoing"), "panel": panel,
            "origin": origin.isoformat().replace("+00:00", "Z"), "role": "inference",
            "split": split, "window_id": window_id, "network": net,
            "has_future_queue": None, "label_coverage": 0.,
            "feature_names": FEATURE_NAMES, "label_kind": "withheld"}
