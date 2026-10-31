import json,hashlib
from pathlib import Path
from task2q.cache import atomic_json
ROOT=Path(__file__).resolve().parent
TASK1=Path(r'C:\Users\18049\Documents\Codex\2026-10-24\trafficflowbench-186-global-road-adaptation\outputs\TrafficFlowBench-Best-20261026')
def guard(stage):
    m=json.loads((TASK1/'package-manifest.json').read_text(encoding='utf8'));h={n:hashlib.sha256((TASK1/n).read_bytes()).hexdigest() for n in m['files']};assert all(h[n]==v['sha256'] for n,v in m['files'].items());atomic_json(ROOT/('task1-guard-'+stage+'.json'),dict(passed=True,files=len(h),hashes=h));return h
