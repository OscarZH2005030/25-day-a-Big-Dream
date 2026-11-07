"""Verify the registered road runs and aggregate only station-level results."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    args = parser.parse_args()
    root = args.workspace
    queue = root / 'work/daytime/time_llm_replication_queue'
    registry = read_json(queue / 'registry.json')
    status = read_json(queue / 'status.json')
    assert status['status'] == 'complete'
    assert [task['name'] for task in registry['tasks']] == [run['name'] for run in status['completed']]
    runs = [(run['name'], root / run['output'], run['exit_code'], True) for run in status['completed']]
    runs.extend((name, root / 'outputs/daytime' / name, 0, False)
                for name in ('time_llm_full12', 'time_llm_west_fit'))
    rows, failures, hashes = [], [], {}
    for name, folder, exit_code, registered in runs:
        protocol = read_json(folder / 'protocol.json')
        hashes[name] = {'protocol_sha256': sha(folder / 'protocol.json')}
        if registered:
            assert protocol['runner_sha256'] == registry['runner_sha256']
        if exit_code:
            log = queue / f'{name}.stderr.log'
            failures.append({'run': name, 'exit_code': exit_code, 'stderr_sha256': sha(log),
                             'reason': log.read_text(encoding='utf-8').strip().splitlines()[-1]})
            continue
        acceptance = read_json(folder / 'acceptance.json')
        assert acceptance['completed'] and acceptance['official_score'] is None
        assert acceptance['adapter_sha256'] == sha(folder / 'adapter.pt')
        hashes[name].update(adapter_sha256=acceptance['adapter_sha256'],
                            metrics_sha256=sha(folder / 'metrics.csv'),
                            per_station_sha256=sha(folder / 'per_station.csv'))
        metrics = read_csv(folder / 'metrics.csv')
        station_metrics = read_csv(folder / 'per_station.csv')
        for variable in sorted({row['variable'] for row in metrics}):
            methods = {row['method']: row for row in metrics if row['variable'] == variable}
            assert set(methods) == {'time_llm', 'persistence', 'previous_day'}
            counts = {int(row['n']) for row in methods.values()}
            assert len(counts) == 1
            n = counts.pop()
            result = {'run': name, 'road': protocol['road'], 'variable': variable,
                      'joint': len({row['variable'] for row in metrics}) == 2, 'n': n}
            for method, row in methods.items():
                parts = [part for part in station_metrics if part['method'] == method and part['variable'] == variable]
                assert sum(int(part['n']) for part in parts) == n
                for metric in ('mae', 'mse'):
                    reconstructed = sum(int(part['n']) * float(part[metric]) for part in parts) / n
                    assert math.isclose(reconstructed, float(row[metric]), rel_tol=1e-12)
                assert math.isclose(math.sqrt(float(row['mse'])), float(row['rmse']), rel_tol=1e-12)
                result[method + '_rmse'] = float(row['rmse'])
            rows.append(result)
    out = root / 'outputs/daytime/time_llm_replication_audit'
    out.mkdir(exist_ok=True)
    with (out / 'by_road.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = {}
    for variable in ('flow_vph', 'speed_kmh'):
        road_rows = [row for row in rows if row['joint'] and row['variable'] == variable]
        summaries[variable] = {
            'roads': len(road_rows), 'points': sum(row['n'] for row in road_rows),
            'beats_both': sum(row['time_llm_rmse'] < min(row['persistence_rmse'], row['previous_day_rmse']) for row in road_rows),
            'beats_persistence': sum(row['time_llm_rmse'] < row['persistence_rmse'] for row in road_rows),
            'beats_previous_day': sum(row['time_llm_rmse'] < row['previous_day_rmse'] for row in road_rows),
        }
    audit = {'summaries': summaries, 'failed_runs': failures, 'hashes': hashes,
             'registry_sha256': sha(queue / 'registry.json'), 'audit_sha256': sha(Path(__file__)),
             'scope': 'Exposed June forecasting diagnostics. Independent aggregate reconstruction and adapter hash checks.'}
    (out / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({'summaries': summaries, 'failed_runs': failures}, indent=2))


if __name__ == '__main__':
    main()
