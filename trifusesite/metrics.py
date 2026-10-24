from __future__ import annotations
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             matthews_corrcoef, roc_auc_score, precision_recall_curve,
                             average_precision_score, auc, brier_score_loss)

def residue_metrics(labels, probabilities, threshold, auprc_method):
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError('Aligned 1D labels and probabilities are required')
    keep = (y == 0) | (y == 1)
    y, p = y[keep], p[keep]
    if not len(y) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Empty valid labels or invalid probabilities')
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must be in [0,1]')
    prediction = (p >= threshold).astype(int)
    if auprc_method == 'average_precision':
        area = float(average_precision_score(y, p)) if y.sum() else 0.0
    elif auprc_method == 'trapezoid':
        precision, recall, _ = precision_recall_curve(y, p)
        area = float(auc(recall, precision))
    else:
        raise ValueError('Specify average_precision or trapezoid explicitly')
    return {'ACC': float(accuracy_score(y, prediction)),
            'Precision': float(precision_score(y, prediction, zero_division=0)),
            'Recall': float(recall_score(y, prediction, zero_division=0)),
            'F1': float(f1_score(y, prediction, zero_division=0)),
            'MCC': float(matthews_corrcoef(y, prediction)),
            'AUROC': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            'AUPRC': area, 'Brier': float(brier_score_loss(y, p)),
            'residues': int(len(y)), 'positives': int(y.sum())}

def evaluate_rows(rows, threshold, auprc_method):
    per_protein = {r['id']: residue_metrics(r['labels'], r['probabilities'], threshold, auprc_method)
                   for r in rows if np.isin(r['labels'], [0, 1]).any()}
    if not per_protein:
        raise ValueError('Evaluation requires ground-truth labels')
    labels = np.concatenate([np.asarray(r['labels']) for r in rows])
    probs = np.concatenate([np.asarray(r['probabilities']) for r in rows])
    return {'pooled': residue_metrics(labels, probs, threshold, auprc_method),
            'per_protein': per_protein, 'threshold': float(threshold),
            'auprc_method': auprc_method, 'aggregation': 'pooled_valid_residues'}

def validation_mcc_threshold(rows, grid_min=0.05, grid_max=0.95, grid_points=181):
    """SMPI rule: validation MCC, 181 thresholds in [0.05,0.95].

    Use p >= threshold; the first (smallest) threshold wins a tied maximum.
    Invoke only on held-out validation or training-set out-of-fold predictions.
    """
    y = np.concatenate([np.asarray(r['labels']) for r in rows])
    p = np.concatenate([np.asarray(r['probabilities']) for r in rows])
    valid = np.isin(y, [0, 1])
    if not valid.any():
        raise ValueError('Validation threshold fitting requires labeled residues')
    if not 0 <= grid_min < grid_max <= 1 or grid_points < 2:
        raise ValueError('Invalid threshold search grid')
    if not np.isfinite(p[valid]).all() or (p[valid] < 0).any() or (p[valid] > 1).any():
        raise ValueError('Invalid validation probabilities')
    thresholds = np.linspace(grid_min, grid_max, grid_points)
    scores = [matthews_corrcoef(y[valid], p[valid] >= t) for t in thresholds]
    return float(thresholds[np.argmax(scores)])
