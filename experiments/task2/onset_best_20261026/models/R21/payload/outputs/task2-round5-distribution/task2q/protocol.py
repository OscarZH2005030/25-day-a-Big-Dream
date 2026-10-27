"""Frozen proxy development boundaries, independent of Task 1's used holdout."""
from __future__ import annotations
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

PROTOCOL = {
    'schema': 'task2_proxy_protocol_v1',
    'target': 'noisy_observation_proxy_not_official_queue_truth',
    'roles': {
        'fit': ['2030-06-01', '2030-11-30'],
        'tune': ['2030-12-01', '2030-12-31'],
        'calibrate': ['2031-01-01', '2031-01-31'],
        'audit': ['2031-02-01', '2031-02-28'],
    },
    'reserved_label_roles': ['calibrate', 'audit'],
    'history_offsets_minutes': list(range(-60, 1, 5)),
    'origin_observation': 'released_masked_at_T_only; missing_kept_as_missing',
    'target_offsets_minutes': [5, 10, 15, 20, 25, 30],
    'label_quality_minimum': 75,
    'threshold': 'released_v_cut_else_0.60_times_links_free_speed',
    'condition': 'history_only_ongoing_if_same_link_has_two_eligible_queue_cells',
    'features_forbidden': ['future_coverage', 'future_eligibility', 'queue_true',
                           'future_ramps', 'task1_offline_predictions'],
    'split_rule': 'whole_history_and_horizon_within_role; same_day_windows',
    'aggregation': ['window', 'panel_condition', 'panel', 'family'],
    'default_model': {'width': 128, 'layers': 4, 'heads': 4, 'dropout': 0.1},
    'engineering_limits': {'optimizer_updates': 4, 'tree_rounds': 12,
                           'max_days_per_role': 2, 'max_windows_per_panel_role': 8},
    'formal_training_available': False,
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')


PROTOCOL_SHA256 = hashlib.sha256(canonical(PROTOCOL)).hexdigest()


def role_for_day(day):
    day = str(day)[:10]
    for role, (first, last) in PROTOCOL['roles'].items():
        if first <= day <= last:
            return role
    raise ValueError('Date outside the frozen public-training calendar: ' + day)


def split_manifest():
    days = {}
    for role, (start, end) in PROTOCOL['roles'].items():
        current, last = date.fromisoformat(start), date.fromisoformat(end)
        while current <= last:
            if current.isoformat() in days:
                raise ValueError('Overlapping frozen date roles')
            days[current.isoformat()] = role
            current += timedelta(days=1)
    assert len(days) == 273
    return {'protocol': PROTOCOL, 'protocol_sha256': PROTOCOL_SHA256, 'date_roles': days}


def write_frozen_split(path):
    path = Path(path)
    expected = split_manifest()
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != expected:
            raise ValueError('Existing frozen split differs; do not overwrite it')
        return expected
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return expected
