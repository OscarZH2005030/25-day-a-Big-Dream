"""Proxy Task 2 metrics with the released window-to-family macro reduction.

Released observations are proxy targets, never organizer queue truth. Incomplete
panel/condition subsets have diagnostics only and cannot become a Task 2 score.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping
import numpy as np

PANEL_FAMILIES = {
    "D7_I10_E": "D7_I10", "D7_I10_W": "D7_I10",
    "D7_I210_E": "D7_I210", "D7_I210_W": "D7_I210",
    "D7_I405_N": "D7_I405", "D7_I405_S": "D7_I405",
    "D12_I5_N": "D12_I5", "D12_I5_S": "D12_I5",
}
CONDITIONS = ("queue_onset", "queue_ongoing")


def condition_name(value) -> str:
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return CONDITIONS[int(value)]
    aliases = {"onset": "queue_onset", "ongoing": "queue_ongoing",
               "0": "queue_onset", "1": "queue_ongoing"}
    value = aliases.get(str(value), str(value))
    if value not in CONDITIONS:
        raise ValueError(f"Unknown queue condition: {value!r}")
    return value


def window_iou(y, pred, mask) -> float:
    """Space-time binary IoU; no eligible cells = 0, empty union = 1."""
    y, pred, mask = np.asarray(y), np.asarray(pred), np.asarray(mask)
    if y.shape != pred.shape or y.shape != mask.shape or y.ndim != 2:
        raise ValueError("y, pred and mask must have identical [horizon, link] shape")
    eligible = mask.astype(bool)
    if not np.any(eligible):
        return 0.0
    for name, array in (("y", y), ("pred", pred)):
        selected = array[eligible]
        if not np.all(np.isfinite(selected)) or not np.all(np.isin(selected, [0, 1])):
            raise ValueError(f"{name} must be finite binary values on eligible cells")
    truth, forecast = y[eligible].astype(bool), pred[eligible].astype(bool)
    union = np.count_nonzero(truth | forecast)
    return float(np.count_nonzero(truth & forecast) / union) if union else 1.0


def aggregate_metrics(records: Iterable[Mapping], strict_complete: bool = False) -> dict:
    """Reduce windows -> panel-condition -> panel -> family -> four families.

    Each record supplies panel, condition, window_id and iou (or y/pred/mask).
    Window counts and panel sizes never change macro weights. Unknown panels,
    duplicate windows, non-finite scores and missing groups are not silently zeroed.
    """
    groups = defaultdict(list)
    seen = set()
    for record in records:
        panel = str(record["panel"])
        if panel not in PANEL_FAMILIES:
            raise ValueError(f"Panel is outside Task 2: {panel}")
        condition = condition_name(record["condition"])
        key = (panel, str(record["window_id"]))
        if key in seen:
            raise ValueError(f"Duplicate window: {key}")
        seen.add(key)
        score = record.get("iou")
        if score is None:
            score = window_iou(record["y"], record["pred"], record.get("mask", record.get("m")))
        score = float(score)
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Window IoU must be finite and in [0, 1]")
        groups[(panel, condition)].append(score)
    missing = [f"{p}/{c}" for p in PANEL_FAMILIES for c in CONDITIONS if (p, c) not in groups]
    complete = not missing
    if strict_complete and not complete:
        raise ValueError("Incomplete Task 2 proxy scope: " + ", ".join(missing))
    pc = {f"{p}/{c}": float(np.mean(v)) for (p, c), v in sorted(groups.items())}
    by_panel = {}
    for panel in PANEL_FAMILIES:
        present = [pc[f"{panel}/{c}"] for c in CONDITIONS if f"{panel}/{c}" in pc]
        if present:
            by_panel[panel] = float(np.mean(present))
    by_family = {}
    for family in dict.fromkeys(PANEL_FAMILIES.values()):
        present = [v for p, v in by_panel.items() if PANEL_FAMILIES[p] == family]
        if present:
            by_family[family] = float(np.mean(present))
    diagnostic = float(np.mean(list(by_family.values()))) if by_family else None
    return {
        "metric": "proxy_IoU", "official_truth_available": False,
        "scope": "complete_eight_panel_proxy" if complete else "incomplete_diagnostic",
        "complete": complete, "proxy_IoU": diagnostic if complete else None,
        "diagnostic_macro_IoU": diagnostic, "missing_groups": missing,
        "n_windows": len(seen), "by_panel_condition": pc,
        "by_panel": by_panel, "by_family": by_family,
        "counts": {f"{p}/{c}": len(v) for (p, c), v in sorted(groups.items())},
        "aggregation": "window -> panel-condition -> panel -> family -> four-family macro",
    }


aggregate_proxy_metrics = aggregate_metrics
