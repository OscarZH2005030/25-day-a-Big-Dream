"""Download checksum-pinned release assets or extract already downloaded local assets."""
import argparse,hashlib,json,urllib.request,zipfile,shutil,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path.home()/'tfb-r127-assets');ap.add_argument('--local-assets',type=Path);ap.add_argument('--training',action='store_true');args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True);manifest=json.loads((HERE/'release-assets.json').read_text(encoding='utf8'))
    for asset in manifest:
        if asset['kind']=='training' and not args.training:continue
        path=(args.local_assets/asset['name']) if args.local_assets else args.output/asset['name']
        if args.local_assets:assert path.is_file(),'Missing local asset: '+str(path)
        if not path.exists():
            temporary=path.with_suffix(path.suffix+'.partial')
            if args.local_assets:shutil.copyfile(args.local_assets/asset['name'],temporary)
            else:
                with urllib.request.urlopen(asset['url'],timeout=120) as response,open(temporary,'wb') as f:shutil.copyfileobj(response,f,1024*1024)
            assert sha(temporary)==asset['sha256'],'Downloaded checksum differs';temporary.replace(path)
        assert path.stat().st_size==asset['bytes'] and sha(path)==asset['sha256']
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            target=args.output.resolve()
            for name in z.namelist():assert (target/name).resolve().is_relative_to(target),'Unsafe archive member'
            if os.name=='nt' and max(len(str(target/name)) for name in z.namelist())>=250:raise ValueError('Choose a shorter --output directory, e.g. your user-home/tfb-r127-assets; no Windows security/registry change is needed.')
            marker=args.output/('.'+asset['root']+'.extraction.json')
            if (args.output/asset['root']).exists():assert marker.exists() and json.loads(marker.read_text())['sha256']==asset['sha256'],'Incomplete/existing extraction: select a new empty --output directory'
            else:z.extractall(args.output);marker.write_text(json.dumps(dict(sha256=asset['sha256'],complete=True)),encoding='utf8')
        print(asset['name'],'SHA256+CRC PASS',str(args.output/asset['root']),flush=True)
if __name__=='__main__':main()
