# 中文说明：下载Caltrans公开实测资料并保留来源哈希；不登录或修改外部服务。
from pathlib import Path
from urllib.request import urlopen,Request
from urllib.parse import urlencode
import json,hashlib
W=Path('work/external_v3');O=Path('outputs/external_v3');W.mkdir(parents=True,exist_ok=True);O.mkdir(parents=True,exist_ok=True)
expected={v['file']:v['sha256'] for v in json.loads((O/'sources.json').read_text(encoding='utf-8-sig'))} if (O/'sources.json').exists() else {}
base='https://dot.ca.gov/-/media/dot-media/programs/traffic-operations/documents/census/2024/'
sources={'d7_ramps_2024.xlsx':base+'2024-d7-ramp-adt-a11y.xlsx','d12_ramps_2024.xlsx':base+'2024-d12-ramp-adt-a11y.xlsx','peak_hours_2024.xlsx':base+'2024-ca-peak-hours-a11y.xlsx'}
api='https://caltrans-gis.dot.ca.gov/arcgis/rest/services/CHhighway/Traffic_AADT/FeatureServer/0'
sources['aadt_metadata.json']=api+'?f=pjson'
sources['aadt_selected_2024.json']=api+'/query?'+urlencode({'where':"DISTRICT IN ('07','12','7') AND RTE IN ('5','005','10','010','210','405')",'outFields':'*','returnGeometry':'true','outSR':'4326','f':'json','resultRecordCount':2000})
manifest=[]
for name,url in sources.items():
 p=W/name
 if not p.exists():p.write_bytes(urlopen(Request(url,headers={'User-Agent':'TrafficFlowBench academic research'}),timeout=60).read())
 if name in expected and hashlib.sha256(p.read_bytes()).hexdigest()!=expected[name]:raise ValueError('External source changed; refusing silent refresh: '+name)
 manifest.append({'file':name,'url':url,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()});print(name,p.stat().st_size,flush=True)
(O/'sources.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
import pandas as pd
for name in sources:
 if name.endswith('.xlsx'):
  book=pd.ExcelFile(W/name)
  print(name,book.sheet_names)
  for s in book.sheet_names[:1]: print(book.parse(s,header=None).head(14).to_string(index=False,header=False))
j=json.loads((W/'aadt_selected_2024.json').read_text());print('aadt rows',len(j.get('features',[])),j.get('error'))
