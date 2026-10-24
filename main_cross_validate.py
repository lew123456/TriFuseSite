"""Five-fold PPI hyperparameter comparison on the training set only."""
from __future__ import annotations
import argparse
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from trifusesite.config import load_config, ModelConfig
from trifusesite.data.dataset import ProteinDataset
from trifusesite.engine import train, predict, write_json
from trifusesite.metrics import evaluate_rows, validation_mcc_threshold

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configs', nargs='+', required=True, help='Candidate PPI YAML configurations')
    p.add_argument('--train-manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int)
    p.add_argument('--seed', type=int)
    w = p.add_mutually_exclusive_group()
    w.add_argument('--class-weights', nargs=2, type=float)
    w.add_argument('--balanced-class-weights', action='store_true')
    p.add_argument('--auprc-method', choices=['average_precision', 'trapezoid'])
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    base_runtime = load_config(a.configs[0])['runtime']
    seed = a.seed if a.seed is not None else base_runtime['seed']
    method = a.auprc_method or base_runtime['auprc_method']
    dataset = ProteinDataset(a.train_manifest, 'ppi')
    if len(dataset) < 5:
        raise ValueError('Five-fold evaluation requires at least five proteins')
    folds = list(KFold(5, shuffle=True, random_state=seed).split(range(len(dataset))))
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    summaries = []
    for candidate, path in enumerate(a.configs):
        config = load_config(path)
        if config['task'] != 'ppi':
            raise ValueError('This entry point implements the PPI five-fold protocol')
        runtime = config['runtime']
        epochs = a.epochs if a.epochs is not None else runtime['epochs']
        weights = (a.class_weights if a.class_weights is not None else
                   'balanced' if a.balanced_class_weights else runtime['class_weights'])
        scores, epoch_choices, oof = [], [], []
        for fold, (train_indices, valid_indices) in enumerate(folds):
            training, validation = Subset(dataset, train_indices), Subset(dataset, valid_indices)
            directory = out / f'candidate_{candidate}' / f'fold_{fold+1}'
            model, metadata = train(config, training, validation, directory,
                                    epochs=epochs, seed=seed, class_weights=weights,
                                    auprc_method=method, fit_threshold=True, device=a.device)
            rows = predict(model, validation, a.device)
            result = evaluate_rows(rows, metadata['threshold'], method)
            scores.append(result['pooled']['AUPRC'])
            epoch_choices.append(metadata['selected_epoch'])
            oof.extend(rows)
            write_json(directory / 'split_ids.json', {
                'train': [dataset.records[i]['id'] for i in train_indices],
                'valid': [dataset.records[i]['id'] for i in valid_indices]})
        summaries.append({'config': path, 'fold_AUPRC': scores,
                          'model_config': ModelConfig(**config['model']).to_dict(),
                          'mean_validation_AUPRC': float(np.mean(scores)),
                          'selected_epochs': epoch_choices,
                          'suggested_refit_epochs': max(1, int(round(np.median(epoch_choices)))),
                          'oof_MCC_threshold': validation_mcc_threshold(oof,
                              runtime['threshold_grid_min'], runtime['threshold_grid_max'],
                              runtime['threshold_grid_points'])})
    selected = max(range(len(summaries)), key=lambda i: summaries[i]['mean_validation_AUPRC'])
    write_json(out / 'selection.json', {'candidates': summaries, 'selected_candidate': selected,
                                      'seed': seed, 'auprc_method': method,
                                      'training_protein_ids': [r['id'] for r in dataset.records],
                                      'selection_data': 'training_set_out_of_fold_predictions'})
