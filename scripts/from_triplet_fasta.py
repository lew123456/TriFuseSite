"""Import >ID / SEQUENCE / binary-label records without remapping labels."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trifusesite.data.dataset import save_jsonl

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    lines = [x.strip() for x in Path(a.input).read_text(encoding='utf-8').splitlines() if x.strip()]
    if len(lines) % 3:
        raise ValueError('Expected three lines per benchmark record')
    rows = []
    for i in range(0, len(lines), 3):
        name, sequence, labels = lines[i:i+3]
        if not name.startswith('>') or len(sequence) != len(labels) or set(labels) - {'0', '1'}:
            raise ValueError(f'Invalid benchmark record at line {i+1}')
        rows.append({'id': name[1:].split()[0], 'sequence': sequence.upper(),
                     'labels': [int(c) for c in labels], 'structure_path': None})
    save_jsonl(a.output, rows)
    print('Add the actual structure_path, chain, structure_source and PPI feature paths before preparation.')
