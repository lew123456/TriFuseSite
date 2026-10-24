from __future__ import annotations
import copy
import json
import random
import platform
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from .config import ModelConfig
from .models.trifusesite import TriFuseSite
from .data.dataset import model_inputs
from .losses import weighted_binary_loss
from .metrics import evaluate_rows, validation_mcc_threshold

RELEASE_FORMAT = 'TriFuseSite-manuscript-v1'

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

@torch.no_grad()
def predict(model, dataset, device, intermediates_dir=None):
    model.eval()
    result = []
    for sample in dataset:
        if intermediates_dir:
            logits, states = model(**model_inputs(sample, device), return_intermediates=True)
            directory = Path(intermediates_dir)
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory / (sample['id'] + '.npz'),
                                **{k: states[k].cpu().numpy() for k in ('X0', 'Hgraph', 'H')},
                                labels=sample['labels'].numpy())
        else:
            logits = model(**model_inputs(sample, device))
        result.append({'id': sample['id'], 'sequence': sample['sequence'],
                       'labels': sample['labels'].tolist(),
                       'probabilities': logits.softmax(-1)[:, 1].cpu().tolist()})
    return result

def balanced_weights(dataset):
    counts = np.zeros(2, dtype=np.int64)
    for sample in dataset:
        y = sample['labels'].numpy()
        counts += np.bincount(y[y >= 0], minlength=2)
    if (counts == 0).any():
        raise ValueError('Both classes must occur to derive balanced weights')
    return (counts.sum() / (2.0 * counts)).tolist()

def train(config, training, validation, output, *, epochs, seed, class_weights,
          auprc_method, threshold=None, fit_threshold=False, device='cpu',
          threshold_provenance=None):
    if epochs < 1:
        raise ValueError('epochs must be positive')
    if fit_threshold and validation is None:
        raise ValueError('Threshold fitting requires a held-out validation set')
    if not fit_threshold and threshold is None:
        raise ValueError('Supply a fixed threshold or enable validation-only threshold fitting')
    if fit_threshold and threshold is not None:
        raise ValueError('Fixed and validation-fitted thresholds are mutually exclusive')
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError('Fixed threshold must be in [0,1]')
    ordered_ids = [training[i]['id'] for i in range(len(training))]
    train_ids = set(ordered_ids)
    if validation is not None and train_ids & {validation[i]['id'] for i in range(len(validation))}:
        raise ValueError('Training/validation protein IDs overlap')
    if validation is not None:
        valid_classes = set()
        for sample in validation:
            valid_classes.update(sample['labels'][sample['labels'] >= 0].tolist())
        if valid_classes != {0, 1}:
            raise ValueError('Validation AP selection requires both labeled classes')
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    model = TriFuseSite(ModelConfig(**config.get('model', {})), config['task']).to(device)
    t = config['training']
    patience = int(t.get('early_stopping_patience', 5))
    min_delta = float(t.get('checkpoint_min_delta', 1e-5))
    if patience < 1 or min_delta < 0:
        raise ValueError('Invalid early-stopping settings')
    precision = t.get('precision', 'cuda_bfloat16')
    if precision not in ('cuda_bfloat16', 'float32'):
        raise ValueError('precision must be cuda_bfloat16 or float32')
    use_amp = torch.device(device).type == 'cuda' and precision == 'cuda_bfloat16'
    if use_amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError('The SMPI precision policy requires CUDA BF16 support; select float32 explicitly on older GPUs')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t['learning_rate'], weight_decay=t['weight_decay'])
    weights = balanced_weights(training) if class_weights == 'balanced' else list(class_weights)
    best_score, best_state, best_epoch, stale = -float('inf'), None, 0, 0
    history = []
    canonical_order = sorted(range(len(training)), key=lambda i: ordered_ids[i])
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        order = list(canonical_order)
        np.random.default_rng(seed + epoch).shuffle(order)
        for i in order:
            sample = training[int(i)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16) if use_amp else nullcontext():
                logits = model(**model_inputs(sample, device))
                loss = weighted_binary_loss(logits, sample['labels'].to(device), weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss for {sample["id"]}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t['gradient_clip'], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        record = {'epoch': epoch, 'loss': float(np.mean(losses))}
        if validation is not None:
            rows = predict(model, validation, device)
            validation_metrics = evaluate_rows(rows, threshold if threshold is not None else 0.5, auprc_method)
            record['validation'] = validation_metrics['pooled']
            score = validation_metrics['pooled']['AUPRC']
        else:
            score = float(epoch)
        if validation is None or score > best_score + min_delta:
            best_score, best_epoch = score, epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        record['stale_epochs'] = stale
        history.append(record)
        print(json.dumps(record), flush=True)
        write_json(out / 'history.json', history)
        if validation is not None and stale >= patience:
            break
    model.load_state_dict(best_state, strict=True)
    if fit_threshold:
        r = config['runtime']
        threshold = validation_mcc_threshold(predict(model, validation, device),
                                              r['threshold_grid_min'], r['threshold_grid_max'],
                                              r['threshold_grid_points'])
    saved = {'format': RELEASE_FORMAT, 'task': config['task'],
             'model_config': model.config.to_dict(), 'state_dict': best_state,
             'training_config': copy.deepcopy(t), 'epochs_requested': epochs,
             'epochs_completed': len(history),
             'selected_epoch': best_epoch, 'seed': seed, 'class_weights': weights,
             'threshold': float(threshold),
             'threshold_source': 'validation_MCC' if fit_threshold else (threshold_provenance or 'user_fixed'),
             'threshold_search': ({k: config['runtime'][k] for k in
                                   ('threshold_grid_min', 'threshold_grid_max', 'threshold_grid_points')}
                                  if fit_threshold else None),
             'auprc_method': auprc_method,
             'training_defaults_source': 'SMPI_train_eval',
             'early_stopping_applied': validation is not None,
             'precision_used': 'cuda_bfloat16' if use_amp else 'float32',
             'shuffle_rule': 'sorted_protein_ids_then_default_rng(seed+epoch)',
             'selection': 'validation_AUPRC' if validation is not None else 'final_epoch',
             'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                             'numpy': np.__version__, 'device': str(device),
                             'cuda_build': torch.version.cuda}}
    torch.save(saved, out / 'model.pt')
    metadata = {key: value for key, value in saved.items() if key != 'state_dict'}
    write_json(out / 'run.json', metadata)
    return model, metadata

def load_checkpoint(path, device='cpu', expected_task=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != RELEASE_FORMAT:
        raise ValueError('Checkpoint architecture format does not match this model')
    if expected_task and checkpoint['task'] != expected_task:
        raise ValueError('Checkpoint task mismatch')
    model = TriFuseSite(ModelConfig(**checkpoint['model_config']), checkpoint['task'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.to(device).eval(), checkpoint
