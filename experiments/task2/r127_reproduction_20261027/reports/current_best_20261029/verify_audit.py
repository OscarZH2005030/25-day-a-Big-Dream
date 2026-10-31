"""Verify the accepted R127 identity and this publication's saved evidence."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPRO = ROOT.parent.parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    manifest = read(ROOT / 'SHA256.json')
    for name, expected in manifest.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    decision = read(ROOT / 'accepted-best.json')
    original = read(REPRO / 'accepted-model.json')
    assert decision['model'] == 'R127'
    assert decision['thresholds'] == [.5, .25]
    assert decision['official_S_queue'] is None
    assert decision['benchmark_comparison'] == original['comparison']
    asset = read(REPRO / 'release-assets.json')[0]
    assert decision['model_archive_sha256'] == asset['sha256']
    audit = read(ROOT / 'calendar_verification.json')
    assert audit['passed'] and not audit['parameters_changed']
    assert audit['fixed_clock_scored_windows'] == 6552
    assert audit['balanced_event_scored_windows'] == 1152
    assert audit['inference_windows'] == 160
    with (ROOT / 'monthly_comparison.csv').open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 12
    scored = [r for r in rows if r['status'] == 'balanced_positive_event_proxy']
    assert len(scored) == 9 and all(int(r['windows']) == 128 for r in scored)
    assert [r['status'] for r in rows[-3:]] == ['predicted_labels_unavailable', 'predicted_labels_unavailable', 'data_unavailable']
    inference = read(ROOT / 'publication_verification.json')
    assert inference['published_reference_prediction_rows_matched'] == 174000
    assert inference['frozen_model_files_verified'] == 1214
    assert read(ROOT / 'failure_analysis_verification.json')['model_files_unchanged'] == 6
    print(f'PASS: {len(manifest)} file hashes; unchanged R127; both benchmarks; monthly scopes; 174000 prediction rows')


if __name__ == '__main__':
    main()
