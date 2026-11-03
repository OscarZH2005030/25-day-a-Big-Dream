"""Preserve first-attempt numerical warnings and verify recorded input/source bytes."""
from pathlib import Path
import hashlib
import json
import re

root=Path(__file__).resolve().parents[2]
out=root/'outputs/daytime'
old=out/'task4_crossmonth_shared_noise_stress'
new=out/'task4_crossmonth_shared_noise_stress_qr'
data=Path('C:/Users/Lenovo/Downloads/2026-ieee-big-data-traffic-flow-bench/kaggle_public')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol=json.loads((old/'protocol.json').read_text())
source=root/'work/daytime/task4_crossmonth_shared_noise_stress.py'
assert sha(source)==protocol['source_sha256']
inputs=json.loads((old/'input_sha256.json').read_text())
for name,digest in inputs.items():assert sha(data/name)==digest,name
log=root/'work/daytime/task4_crossmonth_shared_noise_stress.log'
text=log.read_text(encoding='utf-8-sig')
rconds=[float(v) for v in re.findall(r'rcond = (\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)',text)]
result=dict(first_attempt_preserved=True,source_hash_matches_registration=True,all_recorded_input_hashes_verified=len(inputs),warning_count=text.count('LinAlgWarning:'),minimum_reported_reciprocal_condition=min(rconds) if rconds else None,source_sha256=sha(source),protocol_sha256=sha(old/'protocol.json'),original_method_rows_sha256=sha(old/'all_cases.csv'),original_log_sha256=sha(log),scope='Warnings alone do not prove inaccurate predictions. First-attempt numerical results remain provisional until independent QR/SVD verification and same-case comparison; no scientific outcome inferred here.')
(new/'original_attempt_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(result)
