from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

INPUT_KEYS = ('residue_features', 'esm_embedding', 'patch_adjacency',
              'global_edge_index', 'global_edge_features', 'ca',
              'structure_mask', 'confidence')

def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def save_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')

def resolve_path(base, value):
    p = Path(value)
    return p if p.is_absolute() else Path(base) / p

class ProteinDataset:
    """One complete, sequence-indexed protein per sample; never pad/drop residues."""
    def __init__(self, manifest, task):
        self.manifest = Path(manifest).resolve()
        self.records = read_jsonl(self.manifest)
        self.task = task
        if not self.records:
            raise ValueError('Manifest is empty')
        ids = [r['id'] for r in self.records]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate protein IDs in manifest')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        base = self.manifest.parent
        with np.load(resolve_path(base, r['npz']), allow_pickle=False) as archive:
            a = {key: archive[key] for key in archive.files}
        if str(a['task']) != self.task or str(a['id']) != r['id']:
            raise ValueError('Task or ID mismatch in cached protein')
        sequence = str(a['sequence'])
        n = len(sequence)
        esm = np.load(resolve_path(base, r['esm']), allow_pickle=False)
        if esm.shape != (n, 1280):
            raise ValueError(f'{r["id"]}: expected ESM layer-33 embedding [L,1280]')
        if a['features'].shape != (n, 62) or a['labels'].shape != (n,):
            raise ValueError('Feature/label lengths must equal the benchmark sequence length')
        if not np.isin(a['labels'], [-1, 0, 1]).all():
            raise ValueError('Only binary labels and -1 for unavailable labels are allowed')
        mask = np.asarray(a['structure_mask'], bool)
        ca = np.asarray(a['ca'], np.float32).copy()
        ca[~mask] = 0
        sample = {'id': r['id'], 'sequence': sequence,
                  'labels': torch.as_tensor(a['labels'], dtype=torch.long),
                  'residue_features': torch.as_tensor(a['features'], dtype=torch.float32),
                  'esm_embedding': torch.as_tensor(esm.astype(np.float32)),
                  'patch_adjacency': torch.as_tensor(a['patch_adjacency'], dtype=torch.float32),
                  'global_edge_index': torch.as_tensor(a['global_edge_index'], dtype=torch.long),
                  'global_edge_features': torch.as_tensor(a['global_edge_features'], dtype=torch.float32),
                  'ca': torch.as_tensor(ca), 'structure_mask': torch.as_tensor(mask),
                  'confidence': torch.as_tensor(a['confidence'], dtype=torch.float32)}
        for key in INPUT_KEYS:
            if sample[key].is_floating_point() and not torch.isfinite(sample[key]).all():
                raise ValueError(f'Nonfinite input {key} for {r["id"]}')
        return sample

def model_inputs(sample, device):
    return {key: sample[key].to(device) for key in INPUT_KEYS}
