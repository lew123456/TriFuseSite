# Command Guide

Run commands from the repository root. Use `python main.py <command> --help`
for all options. After installation, `trifusesite <command>` and
`python -m trifusesite <command>` provide the same interface.

## Commands

| Command | Purpose |
|---|---|
| `validate-datasets` | Check benchmark records, split membership and file checksums. |
| `fetch-structures` | List or download PDB and AlphaFold coordinate inputs. |
| `prepare-datasets` | Convert benchmark records into model input caches. |
| `prepare` | Prepare inputs from a custom raw JSONL manifest. |
| `extract-esm` | Generate frozen ESM-2 residue embeddings. |
| `train` | Train a task model using a YAML configuration. |
| `predict` | Export residue binding probabilities and binary predictions. |
| `evaluate` | Calculate pooled-residue and per-protein metrics. |
| `cross-validate` | Perform five-fold PPI model selection. |
| `repeat` | Run a seed schedule for repeated training and evaluation. |
| `paired-analysis` | Summarize repeated bound/unbound evaluations. |

See the [data guide](../../dataset/README.md#preprocessing-workflow) for
structure acquisition, input masks and ESM-2 feature preparation.

## Training

With the task inputs prepared, select the corresponding configuration:

```bash
python main.py train --config config/smpi.yaml --output runs/smpi --device cuda
python main.py train --config config/dpi.yaml --output runs/dpi --device cuda
python main.py train --config config/rpi.yaml --output runs/rpi --device cuda
```

Each command trains an independent task model. Shared architecture and runtime
settings are in [config/base.yaml](../../config/base.yaml); task configurations
specify the input manifests. Use a new output directory for each run.

Training writes the following files under `--output`:

| File | Contents |
|---|---|
| `model.pt` | Selected model state, task, architecture, classification threshold and training metadata. |
| `run.json` | Run settings, selected epoch, threshold and environment information. |
| `history.json` | Epoch-level training and validation records. |

## Checkpoints, Prediction and Evaluation

TriFuseSite task checkpoints are not distributed with this repository.
Generate a checkpoint using the training commands above, then pass its path
through `--checkpoint`. A compatible checkpoint stores the model state together
with the task, architecture and threshold metadata expected by
[engine.py](../engine.py).

ESM-2 feature-extraction weights are separate from the trained TriFuseSite
checkpoint. Their online and offline loading options are described in the
[data guide](../../dataset/README.md#esm-2-embeddings).

For a model trained into `runs/smpi/`:

```bash
python main.py predict \
    --checkpoint runs/smpi/model.pt \
    --manifest data/smpi/test/manifest.jsonl \
    --output runs/smpi/predictions.jsonl

python main.py evaluate \
    --checkpoint runs/smpi/model.pt \
    --manifest data/smpi/test/manifest.jsonl \
    --output runs/smpi/test_metrics.json
```

Prediction exports a JSONL record for each protein, including residue
probabilities, the checkpoint threshold and binary predictions. Evaluation
reports ACC, Precision, Recall, F1, MCC, AUROC, AUPRC and Brier score, with pooled
and per-protein results. Add `--predictions` to `evaluate` for a probability
export, or `--intermediates-dir` to `predict` for `X0`, `Hgraph` and `H` arrays.

## PPI Cross-Validation

Select the PPI configuration through five-fold training-set cross-validation,
then refit on Train334 using the selected epoch count and out-of-fold threshold:

```bash
python main.py cross-validate \
    --configs config/ppi.yaml \
    --train-manifest data/ppi/train/manifest.jsonl \
    --output runs/ppi_cv

python main.py train \
    --config config/ppi.yaml \
    --refit \
    --cv-selection runs/ppi_cv/selection.json \
    --output runs/ppi
```

## Repeated and Paired-Structure Experiments

Inspect the configured seed schedule and commands before execution:

```bash
python main.py repeat \
    --config config/ppi.yaml \
    --output runs/ppi_repeated \
    --dry-run
```

Remove `--dry-run` to execute the schedule. Every PPI seed has its own
cross-validation, refit and bound/unbound evaluations. The two conformations
are evaluated using the same checkpoint and classification threshold.

`paired-analysis` takes ten bound and ten unbound evaluation JSON files in
matching run order, the pair list `dataset/PPI/pairs.jsonl`, an explicit
`--bootstrap-seed` and an output path. Use `--bootstrap-samples` to select the
number of resamples; its default is 10,000.

```bash
python main.py paired-analysis --help
```
