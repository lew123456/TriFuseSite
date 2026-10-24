"""List and download PDB or AlphaFold coordinate inputs."""
from __future__ import annotations
import argparse
import hashlib
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def is_structure(payload):
    return len(payload)>100 and (b'\nATOM  ' in payload or b'_atom_site.' in payload or payload.startswith(b'ATOM  '))

def fetch(request,root=ROOT):
    root=Path(root).resolve()
    path=(root/request['path']).resolve()
    if root not in path.parents:raise ValueError('Structure destination escapes repository')
    row={'path':request['path'],'url':request['url'],'source':request['source']}
    if path.exists():
        payload=path.read_bytes()
        if not is_structure(payload):return dict(row,status='invalid_existing_file')
        return dict(row,status='existing',sha256=hashlib.sha256(payload).hexdigest())
    urls=[request['url']]
    if request['source']=='PDB':urls.append(request['url'].replace('.pdb','.cif'))
    errors=[]
    for url in urls:
        destination=path.with_suffix('.cif') if url.endswith('.cif') else path
        if destination.exists():
            payload=destination.read_bytes()
            if is_structure(payload):return dict(row,path=destination.relative_to(root).as_posix(),status='existing',sha256=hashlib.sha256(payload).hexdigest())
            errors.append({'url':url,'error':'invalid existing file; not overwritten'});continue
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'TriFuseSite-structure-preparation/1.0'})
            with urllib.request.urlopen(req,timeout=45) as response:payload=response.read()
            if not is_structure(payload):raise ValueError('Response is not a recognizable coordinate file')
            destination.parent.mkdir(parents=True,exist_ok=True)
            # Exclusive creation: existing files are never silently overwritten.
            with destination.open('xb') as handle:handle.write(payload)
            return dict(row,path=destination.relative_to(root).as_posix(),url=url,status='downloaded',sha256=hashlib.sha256(payload).hexdigest())
        except (OSError,ValueError,urllib.error.URLError) as error:
            errors.append({'url':url,'error':str(error)})
    return dict(row,status='unresolved',errors=errors)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',nargs='+',choices=['ppi','smpi','dpi','rpi'],default=['ppi','smpi','dpi','rpi'])
    p.add_argument('--download',action='store_true',help='Perform network downloads; otherwise only print the request plan')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--report',default='data/structure_download_report.json')
    a=p.parse_args()
    rows=json.loads((ROOT/'dataset/metadata/structure_requests.json').read_text(encoding='utf-8'))
    rows=[r for r in rows if any(x.split('/')[0] in a.tasks for x in r['uses'])]
    if not a.download:
        print(json.dumps({'unique_structures':len(rows),'tasks':a.tasks,'download':False,'examples':rows[:3]},indent=2));return
    if a.workers<1:raise ValueError('--workers must be positive')
    with ThreadPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(fetch,rows))
    report=Path(a.report);report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps({'records':results},indent=2),encoding='utf-8')
    print(json.dumps({s:sum(r['status']==s for r in results) for s in sorted({r['status'] for r in results})}))
    if any(r['status'] in ('unresolved','invalid_existing_file') for r in results):
        raise SystemExit('Some structures could not be downloaded. See the download report.')

if __name__=='__main__':main()
