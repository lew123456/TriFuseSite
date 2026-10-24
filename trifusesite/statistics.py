from __future__ import annotations
import numpy as np
from scipy.stats import wilcoxon

METRICS = ('ACC', 'Precision', 'Recall', 'F1', 'MCC', 'AUROC', 'AUPRC', 'Brier')

def paired_summary(bound_runs, unbound_runs, pairs, *, bootstrap_seed, bootstrap_samples=10000):
    """S3: average each protein across runs, then test 25 protein-level deltas.

    Confidence intervals use percentile bootstrapping over paired proteins.
    """
    if len(bound_runs) != 10 or len(unbound_runs) != 10 or len(pairs) != 25:
        raise ValueError('The manuscript paired protocol requires 10 runs and 25 protein pairs')
    seeds = [r['seed'] for r in bound_runs]
    if len(set(seeds)) != 10 or set(seeds) != {r['seed'] for r in unbound_runs}:
        raise ValueError('Bound and unbound inputs need the same 10 unique run seeds')
    if len({p['bound_id'] for p in pairs}) != 25 or len({p['unbound_id'] for p in pairs}) != 25:
        raise ValueError('Pair mapping must be one-to-one')
    methods = {r['auprc_method'] for r in bound_runs + unbound_runs}
    if len(methods) != 1:
        raise ValueError('Cannot mix AUPRC integration conventions')
    for b in bound_runs:
        u = next(r for r in unbound_runs if r['seed'] == b['seed'])
        if b['threshold'] != u['threshold']:
            raise ValueError('Bound and unbound must use the same locked threshold within a run')
    pooled = {}
    for name, runs in [('bound', bound_runs), ('unbound', unbound_runs)]:
        pooled[name] = {}
        for metric in METRICS:
            values = [r['pooled'][metric] for r in runs]
            if any(v is None for v in values):
                pooled[name][metric] = {'mean': None, 'sample_sd': None}
            else:
                pooled[name][metric] = {'mean': float(np.mean(values)),
                                        'sample_sd': float(np.std(values, ddof=1))}
    rows = []
    for pair in pairs:
        record = dict(pair)
        for metric in ('MCC', 'AUPRC'):
            b = float(np.mean([r['per_protein'][pair['bound_id']][metric] for r in bound_runs]))
            u = float(np.mean([r['per_protein'][pair['unbound_id']][metric] for r in unbound_runs]))
            record.update({f'{metric}_bound': b, f'{metric}_unbound': u, f'delta_{metric}': u-b})
        rows.append(record)
    rng = np.random.default_rng(bootstrap_seed)
    stats = {}
    for metric in ('MCC', 'AUPRC'):
        delta = np.array([r[f'delta_{metric}'] for r in rows])
        p = (1.0 if np.all(delta == 0) else float(wilcoxon(
            delta, alternative='two-sided', zero_method='wilcox', correction=False, method='auto').pvalue))
        sampled = delta[rng.integers(0, 25, size=(bootstrap_samples, 25))].mean(1)
        stats[metric] = {'mean_delta': float(delta.mean()), 'wilcoxon_p': p,
                         'ci95': np.quantile(sampled, [0.025, 0.975]).tolist(),
                         'lower_unbound': int((delta < 0).sum()),
                         'higher_unbound': int((delta > 0).sum())}
    return {'pooled_run_summary': pooled, 'per_protein': rows, 'paired_statistics': stats,
            'analysis_unit': 'protein_pair_after_run_averaging', 'runs': 10, 'pairs': 25,
            'auprc_method': methods.pop(), 'ci_method': 'percentile_bootstrap_of_protein_pairs',
            'bootstrap_samples': bootstrap_samples, 'bootstrap_seed': bootstrap_seed,
            'wilcoxon': {'alternative': 'two-sided', 'zero_method': 'wilcox', 'method': 'auto', 'correction': False}}
