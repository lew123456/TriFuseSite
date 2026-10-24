from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trifusesite.data.dataset import read_jsonl, save_jsonl, resolve_path

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='Prepared training manifest, not test data')
    p.add_argument('--validation-count', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    source = Path(a.input).resolve()
    rows = read_jsonl(source)
    if not 0 < a.validation_count < len(rows):
        raise ValueError('Invalid validation count')
    order = list(range(len(rows)))
    random.Random(a.seed).shuffle(order)
    validation = set(order[:a.validation_count])
    for row in rows:
        for key in ('npz', 'esm'):
            row[key] = str(resolve_path(source.parent, row[key]).resolve())
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    save_jsonl(out / 'train.jsonl', [r for i, r in enumerate(rows) if i not in validation])
    save_jsonl(out / 'valid.jsonl', [r for i, r in enumerate(rows) if i in validation])
    (out / 'split.json').write_text(json.dumps({'seed': a.seed, 'validation_count': a.validation_count,
                                               'train_count': len(rows)-a.validation_count,
                                               'algorithm': 'Python random.Random shuffle'}), encoding='utf-8')
