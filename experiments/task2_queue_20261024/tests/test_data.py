"""Data invariants that would otherwise quietly inflate proxy scores."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from task2q.cache import content_key, read_arrays
from task2q.data import (FEATURE_NAMES, day, load_sample, make_sample, network,
                         prepare, released_sample, role_for_date)


@pytest.fixture
def release(tmp_path):
    root = tmp_path / "public"
    panel = root / "corridors" / "D7_I210_E"
    net = panel / "network"
    net.mkdir(parents=True)
    pd.DataFrame({"link_id": ["B", "A", "C"], "length_km": [.2, .4, .1],
                  "lanes": [3, 2, 1], "free_speed_kmh": [100., 100., 90.],
                  "capacity_vph": [6000., 4000., 2000.]}).to_csv(net / "links.csv", index=False)
    pd.DataFrame({"link_id": ["A", "B", "C"], "v_cut": [55., np.nan, np.nan]}).to_csv(net / "fd_parameters.csv", index=False)
    pd.DataFrame({"mainline_link_id": ["A", "B", "C"],
                  "outgoing_link_ids": ["B;CONN-external", "", ""],
                  "incoming_link_ids": ["CONN-in", "A", ""]}).to_csv(net / "lwr_mainline_topology.csv", index=False)
    for date in ("2030-06-01", "2030-12-01"):
        frame = pd.DataFrame({"timestamp": np.repeat(pd.date_range(date, periods=288, freq="5min", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), 3),
                              "link_id": np.tile(["A", "B", "C"], 288),
                              "speed_kmh": 90., "flow_vph": 1000., "occupancy": .1,
                              "pct_observed": 100., "is_score_eligible": 1})
        # Observed queue in one past slot and horizon, plus an unknown future:
        frame.loc[(np.arange(len(frame)) // 3 >= 11) & (frame.link_id == "A"), "speed_kmh"] = 35.
        frame.loc[frame.index == 14 * 3 + 1, ["speed_kmh", "flow_vph", "occupancy"]] = np.nan
        for kind in ("mainline_states", "mainline_states_masked"):
            directory = panel / "train" / kind / ("mask_regime=R1" if kind.endswith("masked") else f"year_month={date[:7]}")
            directory.mkdir(parents=True, exist_ok=True)
            released = frame.copy()
            if kind.endswith("masked"):
                released.loc[12 * 3, ["speed_kmh", "flow_vph", "occupancy"]] = np.nan
                released.loc[12 * 3, ["pct_observed", "is_score_eligible"]] = 0
            released.to_parquet(directory / f"synthetic_mainline_{date.replace('-', '_')}.parquet", index=False)
        official = root / "task2" / "D7_I210_E" / "train"
        official.mkdir(parents=True, exist_ok=True)
        if date == "2030-06-01":
            pd.DataFrame({"window_id": ["window-1"], "forecast_origin": [date + "T01:00:00Z"],
                          "condition": ["queue_onset"], "future_coverage": [1.0]}).to_csv(official / "window_index.csv", index=False)
            history = frame.iloc[:12 * 3].copy()
            history["window_id"] = "window-1"
            history.to_parquet(official / "window_history.parquet", index=False)
    return root


def test_threshold_uses_fd_then_links_and_internal_topology(release):
    net = network(release, "D7_I210_E")
    assert net["link_ids"].tolist() == ["A", "B", "C"]
    np.testing.assert_allclose(net["threshold"], [55., 60., 54.])
    assert net["edges"].tolist() == [[0], [1]]
    assert net["isolated"].tolist() == [False, False, True]
    assert net["static"].shape == (3, 6)


def test_proxy_features_are_causal_and_current_frame_is_masked(release):
    net = network(release, "D7_I210_E")
    values = day(release, "D7_I210_E", "2030-06-01")
    sample = make_sample(values, net, 12, "D7_I210_E", "fit")
    altered = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in values.items()}
    for key in ("speed", "flow", "occupancy", "quality", "eligible"):
        altered[key][13:] = 0
        altered["origin_" + key][13:] = 0
    other = make_sample(altered, net, 12, "D7_I210_E", "fit")
    np.testing.assert_array_equal(sample["x"], other["x"])
    assert sample["condition"] == other["condition"] == 0
    assert sample["x"][-1, 0, FEATURE_NAMES.index("speed_missing")] == 1
    # A second past queued observation establishes ongoing, while T does not.
    values["speed"] = values["speed"].copy()
    values["speed"][10, 0] = 30.
    assert make_sample(values, net, 12, "D7_I210_E")["condition"] == 1
    assert np.isfinite(sample["x"]).all()
    assert sample["x"].shape == (13, 3, len(FEATURE_NAMES))
    assert sample["label_kind"] == "noisy_observation_proxy_not_queue_true"


def test_unknown_future_is_masked_not_a_negative(release):
    sample = make_sample(day(release, "D7_I210_E", "2030-06-01"),
                         network(release, "D7_I210_E"), 12, "D7_I210_E")
    assert not sample["m"][1, 1]
    assert sample["m"].sum() == 17
    assert sample["y"][0, 0] == 1


def test_frozen_roles_and_same_day_boundary_guards(release):
    assert role_for_date("2030-11-30") == "fit"
    assert role_for_date("2030-12-01") == "tune"
    assert role_for_date("2031-01-01") == "calibrate"
    assert role_for_date("2031-02-01") == "audit"
    with pytest.raises(ValueError):
        role_for_date("2031-03-01")
    values = day(release, "D7_I210_E", "2030-06-01")
    for slot in (11, 282):
        with pytest.raises(ValueError):
            make_sample(values, network(release, "D7_I210_E"), slot, "D7_I210_E")
    with pytest.raises(ValueError, match="Role"):
        make_sample(values, network(release, "D7_I210_E"), 12, "D7_I210_E", "tune")


def test_cached_days_reuse_and_invalidate_content_corruption(release, tmp_path):
    cache = tmp_path / "cache"
    first = day(release, "D7_I210_E", "2030-06-01", cache)
    assert not first["cache_hit"]
    second = day(release, "D7_I210_E", "2030-06-01", cache)
    assert second["cache_hit"] and first["key"] == second["key"]
    (cache / f"{first['key']}.npz").write_bytes(b"partial-cache")
    assert read_arrays(cache, first["key"]) is None
    rebuilt = day(release, "D7_I210_E", "2030-06-01", cache)
    assert not rebuilt["cache_hit"]
    np.testing.assert_array_equal(first["speed"], rebuilt["speed"])
    copied = tmp_path / "copied-public"
    shutil.copytree(release, copied)
    assert day(copied, "D7_I210_E", "2030-06-01", cache)["key"] == first["key"]


def test_index_is_portable_and_reserved_labels_default_blocked(release, tmp_path):
    base = tmp_path / "bundle"
    manifest = prepare(release, base / "cache", base / "index", panels=["D7_I210_E"],
                       max_days=1, max_windows=3, stride=1)
    assert manifest["sample_count"] == 6
    assert manifest["primary_sample_count"] > 0
    assert not manifest["scope"]["reserved_labels_opened"]
    sample = load_sample(base / "index/index.json", 0)
    relocated = tmp_path / "relocated"
    shutil.copytree(base, relocated)
    np.testing.assert_array_equal(sample["x"], load_sample(relocated / "index/index.json", 0)["x"])
    with pytest.raises(ValueError, match="Reserved"):
        prepare(release, base / "cache", base / "bad", roles=["audit"])
    with pytest.raises(ValueError, match="Full preparation"):
        prepare(release, base / "cache", base / "bad", engineering=False)
    # A changed cache sidecar alone cannot change a frozen dataset manifest.
    entry = manifest["samples"][0]
    record_path = base / "cache" / f"{entry['day_key']}.json"
    record = json.loads(record_path.read_text())
    record["sha256"] = "0" * 64
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="corrupt"):
        load_sample(base / "index/index.json", 0)


def test_content_key_has_no_machine_location():
    assert content_key("day", {"states": "abc"}, {"date": "2030-06-01"}) == content_key(
        "day", {"states": "abc"}, {"date": "2030-06-01"})


def test_released_inference_uses_history_and_masked_T_without_proxy_future(release):
    first = released_sample(release, "D7_I210_E", "train", "window-1")
    assert first["x"].shape == (13, 3, len(FEATURE_NAMES))
    assert first["origin"] == "2030-06-01T01:00:00Z"
    assert first["condition"] == 0 and first["role"] == "inference"
    assert first["label_kind"] == "withheld" and not first["m"].any()
    assert first["x"][-1, 0, FEATURE_NAMES.index("speed_missing")] == 1
    # Delete all unmasked days: inference must never depend on training future.
    shutil.rmtree(release / "corridors/D7_I210_E/train/mainline_states")
    second = released_sample(release, "D7_I210_E", "train", "window-1")
    np.testing.assert_array_equal(first["x"], second["x"])
    masked = release / "corridors/D7_I210_E/train/mainline_states_masked/mask_regime=R1/synthetic_mainline_2030_06_01.parquet"
    frame = pd.read_parquet(masked)
    frame[frame.timestamp <= "2030-06-01T01:00:00Z"].to_parquet(masked, index=False)
    third = released_sample(release, "D7_I210_E", "train", "window-1")
    np.testing.assert_array_equal(first["x"], third["x"])


def test_ambiguous_masked_day_is_rejected(release):
    original = release / "corridors/D7_I210_E/train/mainline_states_masked/mask_regime=R1/synthetic_mainline_2030_06_01.parquet"
    duplicate = original.parent.parent / "mask_regime=R2" / original.name
    duplicate.parent.mkdir()
    shutil.copyfile(original, duplicate)
    with pytest.raises(ValueError, match="Ambiguous"):
        day(release, "D7_I210_E", "2030-06-01")


def test_proxy_threshold_boundary_and_75_quality_are_included(release):
    values = day(release, "D7_I210_E", "2030-06-01")
    for key in ("speed", "quality", "eligible"):
        values[key] = values[key].copy()
    values["speed"][13, 0] = 55.
    values["quality"][13, 0] = 75.
    values["eligible"][13, 0] = True
    sample = make_sample(values, network(release, "D7_I210_E"), 12, "D7_I210_E")
    assert sample["m"][0, 0] and sample["y"][0, 0] == 1


def test_window_cap_is_across_days_and_keeps_date_groups(release, tmp_path):
    panel = release / "corridors/D7_I210_E/train"
    for relative in ("mainline_states/year_month=2030-06", "mainline_states_masked/mask_regime=R1"):
        path = panel / relative / "synthetic_mainline_2030_06_01.parquet"
        frame = pd.read_parquet(path)
        frame["timestamp"] = frame.timestamp.str.replace("2030-06-01", "2030-06-02", regex=False)
        frame.to_parquet(path.with_name("synthetic_mainline_2030_06_02.parquet"), index=False)
    manifest = prepare(release, tmp_path / "cache", tmp_path / "index", panels=["D7_I210_E"],
                       roles=["fit"], max_days=2, max_windows=4, stride=6)
    assert manifest["sample_count"] == 4
    assert {sample["date"] for sample in manifest["samples"]} == {"2030-06-01", "2030-06-02"}
