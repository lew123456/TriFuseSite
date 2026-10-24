"""Prepare caches from sequence-indexed records without changing any labels."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from trifusesite.config import ModelConfig
from trifusesite.data.dataset import read_jsonl, save_jsonl, resolve_path
from trifusesite.data.features import handcrafted62
from trifusesite.data.graphs import radius_graph
from trifusesite.data.structures import map_structure

def prepare(manifest, output, task):
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records, report = [], []
    seen = set()
    for row in read_jsonl(manifest):
        pid, seq = row['id'], row['sequence'].upper()
        if not pid or any(c in pid for c in '/\\:') or pid in seen or pid in ('.', '..'):
            raise ValueError('Protein IDs must be unique safe filenames')
        seen.add(pid)
        n = len(seq)
        if not n:
            raise ValueError('Empty sequence')
        labels = np.asarray(row.get('labels', [-1]*n), dtype=np.int64)
        if labels.shape != (n,) or not np.isin(labels, [-1, 0, 1]).all():
            raise ValueError(f'{pid}: invalid labels')
        source = row.get('structure_source', 'PDB')
        structure = row.get('structure_path')
        structure = resolve_path(manifest.parent, structure) if structure else None
        mode = row.get('confidence_mode', 'plddt' if source == 'AlphaFoldDB_v6' else 'observed')
        xyz, atom_mask, confidence, chain = map_structure(seq, structure, row.get('chain'), mode)
        if 'confidence_path' in row:
            confidence = np.load(resolve_path(manifest.parent, row['confidence_path']), allow_pickle=False).astype(np.float32)
        if confidence.shape != (n,) or not np.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any():
            raise ValueError('Confidence must be an aligned [L] array in [0,1]')
        mask = atom_mask[:, 1]
        confidence[~mask] = 0
        if task == 'ppi':
            fields = row['ppi_features']
            def load(key):
                return np.load(resolve_path(manifest.parent, fields[key]), allow_pickle=False).astype(np.float32)
            parts = [load(k) for k in ('pssm', 'hmm', 'dssp', 'resAF')]
            for part, dim in zip(parts, (20, 20, 14, 7)):
                if part.shape != (n, dim):
                    raise ValueError(f'{pid}: invalid precomputed PPI feature dimensions')
            pseudo = load('pseudopositions')
            if pseudo.shape != (n, 3) or not np.isfinite(pseudo).all():
                raise ValueError('PPI pseudopositions must be finite [L,3]')
            radial = np.linalg.norm(pseudo - pseudo[0], axis=1, keepdims=True) / 15
            features = np.concatenate(parts + [radial], axis=1)
            adjacency, edges, edge_features, fallback = radius_graph(
                pseudo, np.ones(n, bool), ModelConfig(), distance_map=load('distance_map'))
        else:
            features = handcrafted62(seq)
            adjacency, edges, edge_features, fallback = radius_graph(xyz[:, 1], mask)
        np.savez_compressed(output / f'{pid}.npz', id=pid, sequence=seq, task=task,
                            labels=labels, features=features, ca=xyz[:, 1],
                            atom_coordinates=xyz, atom_mask=atom_mask,
                            structure_mask=mask, confidence=confidence,
                            patch_adjacency=adjacency, global_edge_index=edges,
                            global_edge_features=edge_features)
        records.append({'id': pid, 'npz': f'{pid}.npz', 'esm': f'{pid}.esm.npy'})
        report.append({'id': pid, 'length': n, 'valid_labels': int((labels >= 0).sum()),
                       'structure_source': source if structure else 'missing',
                       'chain': chain, 'coverage': float(mask.mean()), 'sequence_graph_fallback': fallback,
                       'confidence_mode': mode})
    save_jsonl(output / 'manifest.jsonl', records)
    save_jsonl(output / 'preprocessing.jsonl', report)
    return output / 'manifest.jsonl'

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, help='Raw JSONL, see dataset/README.md')
    p.add_argument('--output', required=True)
    p.add_argument('--task', required=True, choices=['ppi', 'smpi', 'dpi', 'rpi'])
    a = p.parse_args()
    print(prepare(a.manifest, a.output, a.task))
