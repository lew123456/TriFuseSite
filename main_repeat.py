"""Run the explicit seed schedule with one separately trained model per run."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
import torch
from trifusesite.config import load_config
from trifusesite.engine import write_json

def build_plan(config_path, output, seeds=None, epochs=None, device='cpu'):
    path = Path(config_path).resolve()
    config = load_config(path)
    seeds = config['runtime']['paired_seeds'] if seeds is None else seeds
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Repeat seeds must be nonempty and distinct')
    root = Path(__file__).resolve().parent
    output = Path(output).resolve()
    commands = []
    for seed in seeds:
        run_dir = output / f'seed_{seed}'
        base = [sys.executable, '-X', 'utf8']
        train = base + [str(root/'main_train.py'), '--config', str(path),
                        '--output', str(run_dir/'training'), '--seed', str(seed), '--device', device]
        if config['task'] == 'ppi':
            if not all(key in config['data'] for key in ('bound25', 'unbound25')):
                raise ValueError('Paired PPI runs require bound25 and unbound25 manifests')
            cv = base + [str(root/'main_cross_validate.py'), '--configs', str(path),
                          '--train-manifest', config['data']['train'],
                          '--output', str(run_dir/'cv'), '--seed', str(seed), '--device', device]
            if epochs is not None:
                cv += ['--epochs', str(epochs)]
            commands.append({'seed': seed, 'stage': 'five_fold_selection', 'argv': cv})
            train += ['--refit', '--cv-selection', str(run_dir/'cv/selection.json')]
            # Refit duration comes from training-only fold selection.
            evaluations = [('bound25', config['data']['bound25']), ('unbound25', config['data']['unbound25'])]
        else:
            if epochs is not None:
                train += ['--epochs', str(epochs)]
            evaluations = [('test', config['data']['test'])]
        commands.append({'seed': seed, 'stage': 'training', 'argv': train})
        for name, manifest in evaluations:
            argv = base + [str(root/'main_evaluate.py'), '--checkpoint', str(run_dir/'training/model.pt'),
                           '--manifest', manifest, '--output', str(run_dir/(name+'_metrics.json')),
                           '--device', device]
            commands.append({'seed': seed, 'stage': name+'_evaluation', 'argv': argv})
    return {'task': config['task'], 'seeds': seeds, 'commands': commands}

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--epochs', type=int, help='Maximum epochs per validation training; default 20')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    plan = build_plan(a.config, a.output, a.seeds, a.epochs, a.device)
    output = Path(a.output)
    if output.exists() and any(p.name != 'dry_run_plan.json' for p in output.iterdir()) and not a.dry_run:
        raise FileExistsError('Use a new empty repeat output directory')
    write_json(output/('dry_run_plan.json' if a.dry_run else 'run_plan.json'), plan)
    if a.dry_run:
        print(json.dumps(plan, indent=2))
    else:
        for command in plan['commands']:
            print(command['seed'], command['stage'], flush=True)
            subprocess.run(command['argv'], check=True)
