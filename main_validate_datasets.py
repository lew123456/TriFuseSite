"""Validate bundled real sequences, labels, split membership and checksums."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT=Path(__file__).resolve().parent
def validate(root=ROOT,checksums=True):
    root=Path(root)
    specs=json.loads((root/'dataset/metadata/partition_catalog.json').read_text(encoding='utf-8'))['partitions']
    results=[]
    for spec in specs:
        rows=[json.loads(s) for s in (root/spec['raw_manifest']).read_text(encoding='utf-8').splitlines() if s.strip()]
        ids=[r['id'] for r in rows]
        assert len(ids)==len(set(ids))==spec['proteins'],spec['name']
        assert ids==(root/spec['ids']).read_text(encoding='utf-8').splitlines(),spec['name']
        for r in rows:
            assert len(r['sequence'])==len(r['labels']) and set(r['labels'])<={0,1},r['id']
        residues=sum(len(r['sequence']) for r in rows);positive=sum(sum(r['labels']) for r in rows)
        assert residues==spec['residues'] and positive==spec['positive'],spec['name']
        expected=''.join('>'+r['id']+'\n'+r['sequence']+'\n'+''.join(map(str,r['labels']))+'\n' for r in rows)
        assert (root/spec['triplet_fasta']).read_text(encoding='utf-8')==expected,spec['name']
        results.append({'name':spec['name'],'proteins':len(rows),'residues':residues,'positive':positive,'status':'pass'})
    count=0
    if checksums:
        hashes=json.loads((root/'dataset/SHA256SUMS.json').read_text(encoding='utf-8'))
        for name,expected in hashes.items():
            path=root/'dataset'/name
            assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
            count+=1
    return {'partitions':results,'checked_files':count,'status':'pass'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--skip-checksums',action='store_true');a=p.parse_args()
    print(json.dumps(validate(checksums=not a.skip_checksums),indent=2))
