import argparse
import torch
from trifusesite.engine import load_checkpoint, predict
from trifusesite.data.dataset import ProteinDataset, save_jsonl

def main(argv=None):
    p = argparse.ArgumentParser(prog='trifusesite predict')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--intermediates-dir', help='Optional X0/Hgraph/H arrays for downstream visualization')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args(argv)
    model, checkpoint = load_checkpoint(a.checkpoint, a.device)
    rows = predict(model, ProteinDataset(a.manifest, checkpoint['task']), a.device, a.intermediates_dir)
    for row in rows:
        row['threshold'] = checkpoint['threshold']
        row['predictions'] = [int(p >= checkpoint['threshold']) for p in row['probabilities']]
    save_jsonl(a.output, rows)


if __name__ == '__main__':
    main()
