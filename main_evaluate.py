import argparse
import torch
from trifusesite.engine import load_checkpoint, predict, write_json
from trifusesite.data.dataset import ProteinDataset, save_jsonl
from trifusesite.metrics import evaluate_rows

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--predictions', help='Optional JSONL residue probability export')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    model, checkpoint = load_checkpoint(a.checkpoint, a.device)
    rows = predict(model, ProteinDataset(a.manifest, checkpoint['task']), a.device)
    result = evaluate_rows(rows, checkpoint['threshold'], checkpoint['auprc_method'])
    result.update(task=checkpoint['task'], seed=checkpoint['seed'],
                  checkpoint=str(a.checkpoint), threshold_source=checkpoint['threshold_source'])
    write_json(a.output, result)
    if a.predictions:
        save_jsonl(a.predictions, rows)
    print(result['pooled'])
