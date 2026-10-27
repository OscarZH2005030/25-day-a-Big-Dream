# 中文说明：从锁定版本公开直链下载，不使用账号或Cookie，核验SHA256。
from pathlib import Path
from urllib.request import urlopen,Request
import json,hashlib
O=Path(__file__).resolve().parent;W=O.parents[1]/'work/fivemin_v4';W.mkdir(parents=True,exist_ok=True)
for src in json.loads((O/'sources.json').read_text()):
 p=W/src['file']
 if not p.exists():p.write_bytes(urlopen(Request(src['url'],headers={'User-Agent':'TrafficFlowBench research'}),timeout=60).read())
 assert hashlib.sha256(p.read_bytes()).hexdigest()==src['sha256'],p
 print('verified',p.name,flush=True)
