"""Frozen layer-33 ESM-2 embeddings: 1022 window, 128 overlap, 894 stride."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from trifusesite.data.dataset import read_jsonl, resolve_path

@torch.inference_mode()
def encode(sequence, tokenizer, model, device):
    n = len(sequence)
    total = np.zeros((n, 1280), dtype=np.float32)
    counts = np.zeros((n, 1), dtype=np.float32)
    for start in range(0, n, 894):
        end = min(n, start + 1022)
        tokens = tokenizer(sequence[start:end], return_tensors='pt', add_special_tokens=True).to(device)
        hidden = model(**tokens).last_hidden_state[0, 1:end-start+1].float().cpu().numpy()
        if hidden.shape != (end-start, 1280):
            raise ValueError('Tokenizer/model did not return one embedding per residue')
        total[start:end] += hidden
        counts[start:end] += 1
        if end == n:
            break
    return (total / counts).astype(np.float16)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--model', default='facebook/esm2_t33_650M_UR50D', help='Model ID or local model directory')
    p.add_argument('--allow-download', action='store_true', help='Explicitly allow pretrained ESM-2 download')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=not a.allow_download)
    model = AutoModel.from_pretrained(a.model, local_files_only=not a.allow_download).to(a.device).eval()
    if model.config.hidden_size != 1280 or model.config.num_hidden_layers != 33:
        raise ValueError('Table S1 requires the 33-layer, 1280-dimensional ESM-2 650M model')
    model.requires_grad_(False)
    manifest = Path(a.manifest).resolve()
    for row in read_jsonl(manifest):
        with np.load(resolve_path(manifest.parent, row['npz']), allow_pickle=False) as data:
            sequence = str(data['sequence'])
        out = resolve_path(manifest.parent, row['esm'])
        if out.exists():
            raise FileExistsError(f'Embedding already exists: {out}; use a new cache directory')
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, encode(sequence, tokenizer, model, a.device))
        print(row['id'], len(sequence), flush=True)
