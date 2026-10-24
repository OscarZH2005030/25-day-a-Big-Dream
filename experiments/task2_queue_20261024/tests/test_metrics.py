import numpy as np
import pytest
from task2q.metrics import PANEL_FAMILIES, CONDITIONS, aggregate_metrics, window_iou


def test_window_iou_masks_cells_and_handles_empty_cases():
    truth = np.array([[1, 1], [0, 0]])
    pred = np.array([[1, 0], [1, 0]])
    mask = np.array([[1, 0], [1, 1]], dtype=bool)
    assert window_iou(truth, pred, mask) == .5
    assert window_iou(truth, pred, np.zeros_like(mask)) == 0
    assert window_iou(np.zeros((6, 2)), np.zeros((6, 2)), np.ones((6, 2), bool)) == 1
    with pytest.raises(ValueError, match="binary"):
        window_iou(truth, pred*.9, mask)


def test_four_level_macro_is_not_window_or_cell_micro():
    records = []
    for panel in PANEL_FAMILIES:
        for condition in CONDITIONS:
            # More windows in this group cannot increase its macro weight.
            n = 9 if (panel, condition) == ("D7_I10_E", "queue_onset") else 1
            score = 1 if n == 9 else 0
            records += [dict(panel=panel, condition=condition,
                             window_id=f"{condition}-{i}", iou=score) for i in range(n)]
    result = aggregate_metrics(records, strict_complete=True)
    assert result["proxy_IoU"] == .0625  # 1 / 2 conditions / 2 directions / 4 families
    assert result["by_panel"]["D7_I10_E"] == .5
    assert result["by_family"]["D7_I10"] == .25
    assert result["n_windows"] == 24
    assert result["scope"] == "complete_eight_panel_proxy"


def test_incomplete_diagnostics_never_claim_overall_score():
    records = [dict(panel="D7_I10_E", condition=0, window_id="one", iou=.75)]
    result = aggregate_metrics(records)
    assert result["proxy_IoU"] is None
    assert result["diagnostic_macro_IoU"] == .75
    assert result["scope"] == "incomplete_diagnostic"
    assert len(result["missing_groups"]) == 15
    with pytest.raises(ValueError, match="Incomplete"):
        aggregate_metrics(records, strict_complete=True)
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_metrics(records*2)
    with pytest.raises(ValueError, match="outside Task 2"):
        aggregate_metrics([dict(panel="D12_I405_N", condition=0, window_id="x", iou=.7)])


def test_numeric_string_conditions_keep_both_classes():
    records = [dict(panel=panel, condition=str(condition), window_id=f"w{condition}", iou=condition)
               for panel in PANEL_FAMILIES for condition in (0, 1)]
    result = aggregate_metrics(records, strict_complete=True)
    assert result["proxy_IoU"] == .5
    assert result["missing_groups"] == []
    assert result["by_panel_condition"]["D7_I10_E/queue_onset"] == 0.
    assert result["by_panel_condition"]["D7_I10_E/queue_ongoing"] == 1.
