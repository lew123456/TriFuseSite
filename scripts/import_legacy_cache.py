from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trifusesite.data.features import handcrafted62
from trifusesite.data.graphs import radius_graph
from trifusesite.data.dataset import save_jsonl

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--task', required=True, choices=['smpi', 'dpi', 'rpi'])
    p.add_argument('--confidence-policy', required=True, choices=['preserve', 'observed'],
                   help='Preserve verified normalized confidence or use unit confidence at observed CA positions')
    a = p.parse_args()
    inp, out = Path(a.input_dir).resolve(), Path(a.output_dir).resolve()
    if inp == out or (out.exists() and any(out.iterdir())):
        raise ValueError('Use a new empty output folder')
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(inp.glob('*.npz')):
        with np.load(path, allow_pickle=False) as z:
            r = {k: z[k] for k in z.files}
        pid, sequence = str(r['id']), str(r['sequence'])
        n = len(sequence)
        mask = r['structure_mask'].astype(bool)
        ca = r['ca'].astype(np.float32)
        ca[~mask] = 0
        conf = r['confidence'].astype(np.float32) if a.confidence_policy == 'preserve' else mask.astype(np.float32)
        conf[~mask] = 0
        adjacency, edges, edge, _ = radius_graph(ca, mask)
        np.savez_compressed(out / f'{pid}.npz', id=pid, sequence=sequence, task=a.task,
                            labels=r['labels'], features=handcrafted62(sequence), ca=ca,
                            structure_mask=mask, confidence=conf, patch_adjacency=adjacency,
                            global_edge_index=edges, global_edge_features=edge)
        embedding = path.with_suffix('.esm.npy')
        if embedding.exists():
            shutil.copy2(embedding, out / f'{pid}.esm.npy')
        records.append({'id': pid, 'npz': f'{pid}.npz', 'esm': f'{pid}.esm.npy'})
    if not records:
        raise ValueError('No legacy NPZ records found')
    save_jsonl(out / 'manifest.jsonl', records)
