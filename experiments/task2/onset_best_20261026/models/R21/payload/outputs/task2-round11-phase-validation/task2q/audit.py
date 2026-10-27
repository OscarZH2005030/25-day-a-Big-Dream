"""Bounded release audit: Task 2 windows and static network only.

This module never opens the large mainline observation tables or the sealed
February proxy labels. Published all-zero templates are not training labels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from .metrics import PANEL_FAMILIES, CONDITIONS
except ImportError:
    from metrics import PANEL_FAMILIES, CONDITIONS

HISTORY_COLUMNS = (
    "window_id", "timestamp", "link_id", "speed_kmh", "flow_vph",
    "occupancy", "pct_observed", "is_score_eligible",
)
EXCLUDED_PREDICTION_FIELDS = (
    "future_coverage", "forecast_start", "forecast_end", "date", "window_id",
    "queue_true", "proxy_queue", "future_speed", "future_is_score_eligible",
    "future_ramp_flow", "future_boundary_flow", "task1_bidirectional_predictions",
)
FROZEN_SPLIT_CONTRACT = {
    "fit": ["2030-06-01", "2030-11-30"],
    "tune": ["2030-12-01", "2030-12-31"],
    "calibrate": ["2031-01-01", "2031-01-31"],
    "audit": ["2031-02-01", "2031-02-28"],
    "boundary_policy": "whole 60-minute history and 30-minute horizon in one role",
    "audit_access": "sealed; no February future labels opened by engineering smoke",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_release(data_root, output_path=None, panels=None, splits=None,
                  raise_on_error=False) -> dict:
    """Audit all 8 directions x 3 splits by default; fail closed on bad schemas.

    An explicit subset is an incomplete audit, clearly reported as such. A tiny
    fixture can declare a different windows_per_panel_per_split in its contract.
    No future-coverage field, template zero, or future eligibility is a feature.
    """
    root = Path(data_root).resolve()
    selected_panels = list(PANEL_FAMILIES if panels is None else panels)
    selected_splits = list(("train", "validation", "private") if splits is None else splits)
    errors, warnings, hashes, panel_reports = [], [], {}, {}

    def check(ok, message):
        if not bool(ok):
            errors.append(message)

    def source(path):
        path = Path(path)
        hashes[path.relative_to(root).as_posix()] = sha256_file(path)

    config_path = root / "config" / "competition_contract.json"
    contract = {}
    try:
        source(config_path)
        contract = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        errors.append(f"contract: {exc}")
    queue_contract = contract.get("tasks", {}).get("queue", {})
    declared_panels = queue_contract.get("scored_panels", list(PANEL_FAMILIES))
    check(set(declared_panels) == set(PANEL_FAMILIES), "contract scored_panels differ from frozen 8-direction scope")
    check(queue_contract.get("history_minutes", 60) == 60, "history_minutes must be 60")
    check(queue_contract.get("forecast_horizon_minutes", 30) == 30, "horizon must be 30 minutes")
    check(queue_contract.get("step_minutes", 5) == 5, "step must be 5 minutes")
    expected_counts = queue_contract.get("windows_per_panel_per_split", {c: 5 for c in CONDITIONS})
    for name in ("synthetic_release_v1.json",):
        optional = root / "config" / name
        if optional.is_file():
            source(optional)

    for panel in selected_panels:
        if panel not in PANEL_FAMILIES:
            errors.append(f"Unscored or unknown Task 2 panel: {panel}")
            continue
        panel_report = {"family": PANEL_FAMILIES[panel], "splits": {}}
        panel_reports[panel] = panel_report
        network = root / "corridors" / panel / "network"
        try:
            link_path, fd_path = network / "links.csv", network / "fd_parameters.csv"
            topology_path = network / "lwr_mainline_topology.csv"
            for path in (link_path, fd_path, topology_path):
                source(path)
            links, fd, topology = pd.read_csv(link_path), pd.read_csv(fd_path), pd.read_csv(topology_path)
            required = {"link_id", "length_km", "lanes", "free_speed_kmh", "capacity_vph"}
            check(required.issubset(links.columns), f"{panel}: missing required links columns")
            if not required.issubset(links.columns):
                continue
            check(not links.link_id.duplicated().any(), f"{panel}: duplicate mainline links")
            check(not fd.link_id.duplicated().any(), f"{panel}: duplicate FD link IDs")
            link_ids = set(links.link_id.astype(str))
            check(len(link_ids) > 0, f"{panel}: empty network")
            for column in sorted(required - {"link_id"}):
                values = pd.to_numeric(links[column], errors="coerce")
                check(values.notna().all() and (values > 0).all(), f"{panel}: invalid links.{column}")
            fallback = pd.to_numeric(links.free_speed_kmh, errors="coerce").to_numpy() * .60
            if "v_cut" in fd:
                lookup = fd.set_index("link_id")["v_cut"]
                released_cut = pd.to_numeric(links.link_id.map(lookup), errors="coerce").to_numpy()
                present = np.isfinite(released_cut)
                threshold = np.where(present, released_cut, fallback)
                source_description = "fd_parameters.v_cut with links.free_speed_kmh*0.60 fallback"
                v_cut_count = int(present.sum())
            else:
                threshold, v_cut_count = fallback, 0
                source_description = "links.free_speed_kmh*0.60 (no released fd v_cut)"
            check(np.isfinite(threshold).all() and (threshold > 0).all(), f"{panel}: invalid queue thresholds")
            internal_edges = set()
            topology_key = "mainline_link_id" if "mainline_link_id" in topology else "link_id"
            check(topology_key in topology, f"{panel}: topology has no mainline ID")
            if topology_key in topology:
                check(set(topology[topology_key].astype(str)) == link_ids, f"{panel}: topology/link scope mismatch")
                for _, row in topology.iterrows():
                    this = str(row[topology_key])
                    if this not in link_ids:
                        continue
                    for col, incoming in (("incoming_link_ids", True), ("outgoing_link_ids", False)):
                        value = row.get(col)
                        if pd.isna(value):
                            continue
                        for neighbor in str(value).split(";"):
                            if neighbor in link_ids and neighbor != this:
                                internal_edges.add((neighbor, this) if incoming else (this, neighbor))
            connected = {node for edge in internal_edges for node in edge}
            panel_report["network"] = {
                "links": len(link_ids), "internal_edges": len(internal_edges),
                "isolated_links": len(link_ids - connected),
                "threshold_source": source_description, "released_v_cut_links": v_cut_count,
                "threshold_min_kmh": float(np.min(threshold)),
                "threshold_max_kmh": float(np.max(threshold)),
                "flow_units": "released flow_vph and capacity_vph are total-link; no silent per-lane conversion",
                "external_CONN_edges_used": False,
            }
        except (OSError, KeyError, ValueError) as exc:
            errors.append(f"{panel}/network: {type(exc).__name__}: {exc}")
            continue

        for split in selected_splits:
            tag = f"{panel}/{split}"
            try:
                folder = root / "task2" / panel / split
                index_path = folder / "window_index.csv"
                history_path = folder / "window_history.parquet"
                template_path = folder / "sample_submission_queue.csv"
                for path in (index_path, history_path, template_path):
                    source(path)
                index = pd.read_csv(index_path)
                history = pd.read_parquet(history_path)
                template = pd.read_csv(template_path)
                required_index = {"window_id", "panel", "split", "forecast_origin", "condition"}
                check(required_index.issubset(index.columns), f"{tag}: index schema incomplete")
                check(set(history.columns) == set(HISTORY_COLUMNS), f"{tag}: history schema violates input allowlist")
                check(set(template.columns) == {"window_id", "timestamp", "link_id", "queue_pred"}, f"{tag}: template schema differs")
                if not (required_index.issubset(index.columns) and set(HISTORY_COLUMNS).issubset(history.columns)
                        and {"window_id", "timestamp", "link_id", "queue_pred"}.issubset(template.columns)):
                    continue
                for table_name, table in (("index", index), ("history", history), ("template", template)):
                    check("queue_true" not in table, f"{tag}: unexpected official truth in {table_name}")
                check(not index.window_id.duplicated().any(), f"{tag}: duplicate index window")
                check(set(index.panel.astype(str)) == {panel}, f"{tag}: wrong index panel")
                check(set(index["split"].astype(str)) == {split}, f"{tag}: wrong index split")
                check(set(index.condition) == set(CONDITIONS), f"{tag}: condition set differs")
                counts = index.condition.value_counts().to_dict()
                for condition in CONDITIONS:
                    check(counts.get(condition, 0) == expected_counts.get(condition, 5), f"{tag}: wrong {condition} count")
                windows = set(index.window_id.astype(str))
                check(set(history.window_id.astype(str)) == windows, f"{tag}: history window set mismatch")
                check(set(template.window_id.astype(str)) == windows, f"{tag}: template window set mismatch")
                check(not history.duplicated(["window_id", "timestamp", "link_id"]).any(), f"{tag}: duplicate history keys")
                check(not template.duplicated(["window_id", "timestamp", "link_id"]).any(), f"{tag}: duplicate template keys")
                check(set(history.link_id.astype(str)) == link_ids, f"{tag}: history link set mismatch")
                check(set(template.link_id.astype(str)) == link_ids, f"{tag}: template link set mismatch")
                check(pd.to_numeric(template.queue_pred, errors="coerce").eq(0).all(), f"{tag}: template queue_pred is not all-zero placeholder")
                htimes = pd.to_datetime(history.timestamp, utc=True, errors="coerce")
                ftimes = pd.to_datetime(template.timestamp, utc=True, errors="coerce")
                check(htimes.notna().all() and ftimes.notna().all(), f"{tag}: invalid timestamps")
                origins = pd.to_datetime(index.forecast_origin, utc=True, errors="coerce")
                check(origins.notna().all(), f"{tag}: invalid forecast origins")
                bounds = contract.get("split", {}).get(split, {})
                for pos, row in enumerate(index.itertuples(index=False)):
                    origin = origins.iloc[pos]
                    if pd.isna(origin):
                        continue
                    hsel, fsel = history.window_id.eq(row.window_id), template.window_id.eq(row.window_id)
                    expected_history = set(origin - pd.to_timedelta(np.arange(1, 13)*5, unit="min"))
                    expected_future = set(origin + pd.to_timedelta(np.arange(1, 7)*5, unit="min"))
                    check(set(htimes[hsel]) == expected_history, f"{tag}/{row.window_id}: history is not T-60..T-5")
                    check(set(ftimes[fsel]) == expected_future, f"{tag}/{row.window_id}: template is not T+5..T+30")
                    check(int(hsel.sum()) == 12*len(link_ids), f"{tag}/{row.window_id}: history grid incomplete")
                    check(int(fsel.sum()) == 6*len(link_ids), f"{tag}/{row.window_id}: template grid incomplete")
                    if bounds.get("start") and bounds.get("end"):
                        lo = pd.Timestamp(bounds["start"], tz="UTC")
                        hi = pd.Timestamp(bounds["end"], tz="UTC") + pd.Timedelta(days=1)
                        check(origin-pd.Timedelta(minutes=60) >= lo and origin+pd.Timedelta(minutes=30) < hi,
                              f"{tag}/{row.window_id}: full window crosses official split boundary")
                    for field, expected_time in (("history_start", origin-pd.Timedelta(minutes=60)),
                                                 ("history_end", origin),
                                                 ("forecast_start", origin+pd.Timedelta(minutes=5)),
                                                 ("forecast_end", origin+pd.Timedelta(minutes=30))):
                        if field in index:
                            check(pd.to_datetime(index.iloc[pos][field], utc=True) == expected_time,
                                  f"{tag}/{row.window_id}: invalid {field}")
                pct = pd.to_numeric(history.pct_observed, errors="coerce")
                eligible = pd.to_numeric(history.is_score_eligible, errors="coerce")
                check(pct.between(0, 100).all(), f"{tag}: pct_observed outside [0,100]")
                check(eligible.isin([0, 1]).all(), f"{tag}: eligibility is not binary")
                expected_eligible = pct.ge(75) & history.speed_kmh.notna() & history.flow_vph.notna()
                check(eligible.eq(expected_eligible.astype(int)).all(), f"{tag}: history eligibility contradicts released quality contract")
                panel_report["splits"][split] = {
                    "windows": len(index), "condition_counts": {str(k): int(v) for k, v in counts.items()},
                    "history_rows": len(history), "template_rows": len(template),
                    "history_steps": 12, "forecast_steps": 6,
                    "null_cells": {c: int(history[c].isna().sum()) for c in ("speed_kmh", "flow_vph", "occupancy")},
                    "eligible_history_rows": int(eligible.eq(1).sum()),
                    "history_quality_mean": float(pct.mean()),
                    "template_is_label": False, "official_truth_available": False,
                    "index_columns_excluded_from_prediction": sorted(set(index.columns) & set(EXCLUDED_PREDICTION_FIELDS)),
                }
            except (OSError, KeyError, ValueError, TypeError) as exc:
                errors.append(f"{tag}: {type(exc).__name__}: {exc}")

    complete = set(selected_panels) == set(PANEL_FAMILIES) and set(selected_splits) == {"train", "validation", "private"}
    if not complete:
        warnings.append("Audit covers an explicit subset; it is not the full release audit")
    report = {
        "schema_version": 1, "passed": not errors, "errors": errors, "warnings": warnings,
        "scope": "all_eight_panels_three_splits" if complete else "explicit_subset",
        "official_truth_available": False,
        "label_note": "No organizer Task 2 queue labels are released, including train; zeros are placeholders",
        "panels": panel_reports, "source_hashes": hashes,
        "history_feature_source_allowlist": list(HISTORY_COLUMNS),
        "excluded_prediction_fields": list(EXCLUDED_PREDICTION_FIELDS),
        "frozen_split_contract": FROZEN_SPLIT_CONTRACT,
        "large_observation_tables_opened": False, "february_future_labels_opened": False,
        "summary": {
            "panels": len(panel_reports),
            "links": sum(p.get("network", {}).get("links", 0) for p in panel_reports.values()),
            "isolated_links": sum(p.get("network", {}).get("isolated_links", 0) for p in panel_reports.values()),
            "windows": sum(s.get("windows", 0) for p in panel_reports.values() for s in p["splits"].values()),
            "template_rows": sum(s.get("template_rows", 0) for p in panel_reports.values() for s in p["splits"].values()),
            "source_files_hashed": len(hashes),
        },
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    if errors and raise_on_error:
        raise ValueError("Task 2 audit failed: " + "; ".join(errors[:10]))
    return report


audit = audit_release
