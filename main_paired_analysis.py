import argparse
import json
from pathlib import Path
from trifusesite.data.dataset import read_jsonl
from trifusesite.engine import write_json
from trifusesite.statistics import paired_summary

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bound', nargs=10, required=True, help='10 main_evaluate.py JSON outputs')
    p.add_argument('--unbound', nargs=10, required=True)
    p.add_argument('--pairs', required=True, help='JSONL: bound_id, unbound_id, optional pair_id')
    p.add_argument('--bootstrap-seed', required=True, type=int)
    p.add_argument('--bootstrap-samples', default=10000, type=int)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if a.bootstrap_samples < 1:
        raise ValueError('bootstrap_samples must be positive')
    load = lambda paths: [json.loads(Path(x).read_text(encoding='utf-8')) for x in paths]
    write_json(a.output, paired_summary(load(a.bound), load(a.unbound), read_jsonl(a.pairs),
                                        bootstrap_seed=a.bootstrap_seed, bootstrap_samples=a.bootstrap_samples))
