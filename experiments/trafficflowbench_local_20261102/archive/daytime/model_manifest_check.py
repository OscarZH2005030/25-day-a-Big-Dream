"""Read-only provenance check for frozen packages used in daytime tests."""
import json
import common as d
rows=[]
for name in ['context','late55']:
 folder=d.ROOT/'outputs/overnight'/(name+'_candidate');manifest=folder/'manifest.json';entries=json.loads(manifest.read_text())['files'];bad=[relative for relative,expected in entries.items() if d.sha(folder/relative)!=expected]
 rows.append(dict(package=name,manifest_sha256=d.sha(manifest),files=len(entries),passed=not bad,mismatches=bad));assert not bad,(name,bad)
d.save(d.OUT/'starting_model_provenance.json',dict(packages=rows,read_only=True,kaggle_upload=False));print('frozen model provenance verified',flush=True)
