from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from trifusesite.config import load_config, ModelConfig
from trifusesite.data.dataset import ProteinDataset
from trifusesite.engine import train

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--train-manifest')
    p.add_argument('--valid-manifest')
    p.add_argument('--refit', action='store_true', help='Train all training proteins for the chosen epoch count; no validation selection')
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, help='Default: config runtime.epochs (20)')
    p.add_argument('--seed', type=int, help='Default: config runtime.seed (2025)')
    p.add_argument('--cv-selection', help='PPI refit: selection.json from main_cross_validate.py')
    w = p.add_mutually_exclusive_group()
    w.add_argument('--class-weights', type=float, nargs=2, metavar=('W0', 'W1'))
    w.add_argument('--balanced-class-weights', action='store_true', help='N/(2*Nc) from training labels only; operational choice')
    t = p.add_mutually_exclusive_group()
    t.add_argument('--threshold', type=float)
    t.add_argument('--fit-validation-threshold', action='store_true', help='Fit validation MCC on the SMPI threshold grid (default with validation)')
    p.add_argument('--auprc-method', choices=['average_precision', 'trapezoid'])
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p

def main():
    a = parser().parse_args()
    c = load_config(a.config)
    runtime = c['runtime']
    epochs = a.epochs if a.epochs is not None else runtime['epochs']
    seed = a.seed if a.seed is not None else runtime['seed']
    method = a.auprc_method or runtime['auprc_method']
    weights = (a.class_weights if a.class_weights is not None else
               'balanced' if a.balanced_class_weights else runtime['class_weights'])
    training = ProteinDataset(a.train_manifest or c['data']['train'], c['task'])
    valid_path = a.valid_manifest or c['data'].get('valid')
    if a.refit and a.valid_manifest:
        raise ValueError('--refit and --valid-manifest cannot be combined')
    if c['task'] == 'ppi' and not a.refit and not valid_path:
        raise ValueError('Use main_cross_validate.py for five-fold PPI selection, then --refit for Train334')
    validation = None if a.refit or not valid_path else ProteinDataset(valid_path, c['task'])
    threshold, provenance = a.threshold, None
    if a.cv_selection:
        if not a.refit or c['task'] != 'ppi':
            raise ValueError('--cv-selection is only for full-training PPI --refit')
        selection = json.loads(Path(a.cv_selection).read_text(encoding='utf-8'))
        selected = selection['candidates'][selection['selected_candidate']]
        if selected['model_config'] != ModelConfig(**c['model']).to_dict():
            raise ValueError('Selected CV architecture differs from the requested refit configuration')
        if selection['auprc_method'] != method:
            raise ValueError('CV/refit AUPRC methods differ')
        if selection['seed'] != seed:
            raise ValueError('CV/refit seeds differ; use the corresponding seed-specific selection file')
        if sorted(selection['training_protein_ids']) != sorted(r['id'] for r in training.records):
            raise ValueError('CV/refit training protein membership differs')
        if threshold is None:
            threshold = selected['oof_MCC_threshold']
            provenance = 'training_OOF_MCC'
        if a.epochs is None:
            epochs = selected['suggested_refit_epochs']
    if a.refit and threshold is None:
        raise ValueError('Refit needs --cv-selection or a previously locked --threshold; it never fits on test labels')
    if a.refit and a.fit_validation_threshold:
        raise ValueError('No validation set is held out during full-training refit')
    fit_threshold = threshold is None
    if fit_threshold and runtime['threshold_rule'] != 'validation_mcc':
        raise ValueError('Unknown configured threshold rule')
    train(c, training, validation, a.output, epochs=epochs, seed=seed,
          class_weights=weights, threshold=threshold, fit_threshold=fit_threshold,
          auprc_method=method, device=a.device, threshold_provenance=provenance)

if __name__ == '__main__':
    main()
