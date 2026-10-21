"""Download and verify the complete GitHub Release asset before extracting."""
import argparse,hashlib,json,urllib.request,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--destination',type=Path,default=ROOT/'downloaded')
    args=parser.parse_args();info=json.loads((ROOT/'release-assets.json').read_text())
    args.destination.mkdir(parents=True,exist_ok=True);archive=args.destination/info['filename']
    if not archive.exists():
        temporary=archive.with_suffix('.partial')
        with urllib.request.urlopen(info['url']) as response,temporary.open('wb') as out:
            while block:=response.read(4*1024**2):out.write(block)
        temporary.replace(archive)
    h=hashlib.sha256()
    with archive.open('rb') as source:
        for b in iter(lambda:source.read(8*1024**2),b''):h.update(b)
    assert h.hexdigest()==info['sha256'],'Archive SHA256 mismatch'
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target=(args.destination/member.filename).resolve()
            assert target.is_relative_to(args.destination.resolve()),'Unsafe archive path'
        z.extractall(args.destination)
    print('Verified and extracted to',args.destination.resolve())

if __name__=='__main__':main()
