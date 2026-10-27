import json,hashlib
from pathlib import Path
r=Path(__file__).resolve().parent;m=json.loads((r/'archive-manifest.json').read_text(encoding='utf8'))
for n,v in m['files'].items():
 h=hashlib.sha256((r/n).read_bytes()).hexdigest();assert h==v['sha256'],n
print('PASS',len(m['files']),'saved files')
