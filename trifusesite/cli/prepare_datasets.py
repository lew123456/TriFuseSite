from __future__ import annotations
import argparse
import json
from pathlib import Path
from .prepare import prepare
from trifusesite.data.dataset import read_jsonl, save_jsonl

ROOT=Path(__file__).resolve().parents[2]

def load_catalog(root=ROOT):
    return json.loads((Path(root)/'dataset/metadata/partition_catalog.json').read_text(encoding='utf-8'))['partitions']

def load_structure_index(path):
    if path is None:
        return {}
    path=Path(path).resolve()
    result={}
    for row in read_jsonl(path):
        key=(row['task'],row['id'])
        if key in result:
            raise ValueError(f'Duplicate structure-index entry: {key}')
        if 'path' not in row:
            raise ValueError('Structure index requires path; null means verified unavailable')
        if row['path'] is None and not row.get('reason'):
            raise ValueError('An unavailable structure requires a recorded reason')
        item=dict(row)
        if row['path'] is not None:
            p=Path(row['path']);item['path']=str(p if p.is_absolute() else (path.parent/p).resolve())
        result[key]=item
    return result

def resolve_records(spec, root=ROOT, structure_index=None, geometry_root=None):
    root=Path(root).resolve()
    manifest=root/spec['raw_manifest']
    index=structure_index or {}
    records=[]; missing=[]
    for source in read_jsonl(manifest):
        row=dict(source)
        structure=(manifest.parent/row['structure_path']).resolve()
        override=index.get((spec['task'],row['id']))
        if override is not None:
            structure=Path(override['path']) if override['path'] is not None else None
            for field in ('chain','confidence_mode','structure_source'):
                if field in override:row[field]=override[field]
            if structure is None: row['missing_structure_reason']=override['reason']
        elif not structure.exists() and structure.with_suffix('.cif').is_file():
            structure=structure.with_suffix('.cif')
        row['structure_path']=str(structure) if structure else None
        if structure is not None and not structure.is_file():
            missing.append({'id':row['id'],'kind':'structure_input','path':str(structure)})
        if spec['task']=='ppi':
            features={}
            for kind,value in row['ppi_features'].items():
                path=(manifest.parent/value).resolve()
                if geometry_root is not None and kind in ('pseudopositions','distance_map'):
                    path=Path(geometry_root).resolve()/kind/(row['id']+'.npy')
                features[kind]=str(path)
                if not path.is_file():missing.append({'id':row['id'],'kind':kind,'path':str(path)})
            row['ppi_features']=features
        records.append(row)
    return records,missing

def main(argv=None):
    p=argparse.ArgumentParser(prog='trifusesite prepare-datasets')
    p.add_argument('--tasks',nargs='+',choices=['ppi','smpi','dpi','rpi'],default=['ppi','smpi','dpi','rpi'])
    p.add_argument('--splits',nargs='+',help='Optional split names, e.g. train valid test or test60')
    p.add_argument('--structure-index',help='Optional JSONL mapping task/id to exact local structure path or verified null')
    p.add_argument('--ppi-geometry-root',help='Directory containing pseudopositions/ID.npy and distance_map/ID.npy')
    p.add_argument('--output-root',default='data',help='Prepared cache root; matches the task configurations by default')
    p.add_argument('--check-only',action='store_true',help='Report input availability without preparing or masking anything')
    a=p.parse_args(argv)
    specs=[s for s in load_catalog() if s['task'] in a.tasks and (not a.splits or s['split'] in a.splits)]
    if not specs:raise ValueError('No selected benchmark partitions')
    index=load_structure_index(a.structure_index)
    jobs=[];report=[]
    for spec in specs:
        rows,missing=resolve_records(spec,structure_index=index,geometry_root=a.ppi_geometry_root)
        summary={'task':spec['task'],'split':spec['split'],'proteins':len(rows),
                 'verified_missing_structures':sum(r['structure_path'] is None for r in rows),
                 'missing_input_files':len(missing),'ready':not missing,'missing_examples':missing[:3]}
        report.append(summary);jobs.append((spec,rows,missing))
    print(json.dumps(report,indent=2))
    if a.check_only:return
    if any(missing for _,_,missing in jobs):
        raise FileNotFoundError('Required input files are missing. Run --check-only and supply the listed paths or a structure index.')
    for spec,_,_ in jobs:
        target=Path(a.output_root)/spec['task']/spec['split']
        if target.exists() and any(target.iterdir()):
            raise FileExistsError(f'Use an empty prepared output directory: {target}')
    for spec,rows,_ in jobs:
        target=Path(a.output_root)/spec['task']/spec['split']
        resolved=target/'resolved_raw_manifest.jsonl'
        save_jsonl(resolved,rows)
        print(prepare(resolved,target,spec['task']))

if __name__=='__main__':main()
