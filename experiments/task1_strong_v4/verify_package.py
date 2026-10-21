"""Verify the published model bundle against its relative SHA256 manifest."""
import argparse,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--weights-only',action='store_true');args=p.parse_args()
    manifest=json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'));checked=0;failures=[]
    for r in manifest['files']:
        if args.weights_only and not r['path'].endswith(('.pt','.npz')):continue
        f=ROOT/r['path']
        if not f.exists() or f.stat().st_size!=r['bytes']:failures.append(r['path']);continue
        h=hashlib.sha256()
        with f.open('rb') as source:
            for b in iter(lambda:source.read(8*1024**2),b''):h.update(b)
        if h.hexdigest()!=r['sha256']:failures.append(r['path'])
        checked+=1
    print(json.dumps(dict(passed=not failures,checked=checked,failures=failures),indent=2))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
