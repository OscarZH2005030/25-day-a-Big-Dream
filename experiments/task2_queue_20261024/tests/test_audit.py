import json
import numpy as np
import pandas as pd
from task2q.audit import audit_release
from task2q.metrics import PANEL_FAMILIES


def fixture_release(root):
    config = root / "config"
    config.mkdir(parents=True)
    contract = {"tasks": {"queue": {"scored_panels": list(PANEL_FAMILIES),
                 "windows_per_panel_per_split": {"queue_onset": 1, "queue_ongoing": 1}}}}
    (config / "competition_contract.json").write_text(json.dumps(contract))
    network = root / "corridors" / "D7_I10_E" / "network"
    network.mkdir(parents=True)
    pd.DataFrame(dict(link_id=["A", "B"], length_km=[.5, .6], lanes=[3, 2],
                      free_speed_kmh=[100, 100], capacity_vph=[6000, 4000])).to_csv(network/"links.csv", index=False)
    pd.DataFrame(dict(link_id=["A", "B"], v_cut=[20, np.nan])).to_csv(network/"fd_parameters.csv", index=False)
    pd.DataFrame(dict(mainline_link_id=["A", "B"], incoming_link_ids=["CONN-X", ""],
                      outgoing_link_ids=["", "CONN-Y"])).to_csv(network/"lwr_mainline_topology.csv", index=False)
    folder = root / "task2" / "D7_I10_E" / "train"
    folder.mkdir(parents=True)
    rows, history, template = [], [], []
    for i, condition in enumerate(("queue_onset", "queue_ongoing")):
        origin = pd.Timestamp("2030-06-02T12:00Z") + pd.Timedelta(days=i)
        wid = f"window-{i}"
        rows.append(dict(window_id=wid, panel="D7_I10_E", split="train", condition=condition,
                         forecast_origin=origin.isoformat(), future_coverage=.9))
        for step in range(-12, 0):
            for link in ("A", "B"):
                history.append(dict(window_id=wid, timestamp=(origin+pd.Timedelta(minutes=5*step)).isoformat(),
                    link_id=link, speed_kmh=70., flow_vph=2000., occupancy=.1, pct_observed=100, is_score_eligible=1))
        for step in range(1, 7):
            for link in ("A", "B"):
                template.append(dict(window_id=wid, timestamp=(origin+pd.Timedelta(minutes=5*step)).isoformat(),
                                     link_id=link, queue_pred=0))
    pd.DataFrame(rows).to_csv(folder/"window_index.csv", index=False)
    pd.DataFrame(history).to_parquet(folder/"window_history.parquet", index=False)
    pd.DataFrame(template).to_csv(folder/"sample_submission_queue.csv", index=False)
    return folder


def test_audit_threshold_fallback_external_edges_and_feature_exclusion(tmp_path):
    fixture_release(tmp_path)
    report = audit_release(tmp_path, panels=["D7_I10_E"], splits=["train"])
    assert report["passed"], report["errors"]
    assert report["scope"] == "explicit_subset"
    net = report["panels"]["D7_I10_E"]["network"]
    assert net["threshold_min_kmh"] == 20 and net["threshold_max_kmh"] == 60
    assert net["released_v_cut_links"] == 1 and net["isolated_links"] == 2
    assert report["summary"]["source_files_hashed"] == 7
    assert report["official_truth_available"] is False
    assert report["february_future_labels_opened"] is False
    details = report["panels"]["D7_I10_E"]["splits"]["train"]
    assert "future_coverage" in details["index_columns_excluded_from_prediction"]


def test_audit_catches_origin_leakage_bad_template_and_future_columns(tmp_path):
    folder = fixture_release(tmp_path)
    history = pd.read_parquet(folder/"window_history.parquet")
    history.loc[0, "timestamp"] = "2030-06-02T12:00:00Z"  # T belongs only in optional separate frame.
    history["future_speed"] = 40.
    history.to_parquet(folder/"window_history.parquet", index=False)
    template = pd.read_csv(folder/"sample_submission_queue.csv")
    template.loc[0, "queue_pred"] = 1
    template.to_csv(folder/"sample_submission_queue.csv", index=False)
    report = audit_release(tmp_path, panels=["D7_I10_E"], splits=["train"])
    assert not report["passed"]
    assert any("allowlist" in error for error in report["errors"])
    assert any("T-60..T-5" in error for error in report["errors"])
    assert any("not all-zero" in error for error in report["errors"])
