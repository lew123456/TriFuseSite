# TriFuseSite

**TriFuseSite: A Unified Multisource Framework for Conformation-Robust Protein Binding-Site Prediction**

TriFuseSite predicts residue-level binding sites for protein–protein (PPI),
protein–small molecule (SMPI), protein–DNA (DPI), and protein–RNA (RPI)
interactions. It integrates frozen ESM-2 sequence representations, residue
descriptors, and multiscale structural information. Each task is trained
independently using the same network architecture.

## Model

- **Semantic fusion:** gated residual fusion of ESM-2 embeddings and 62-dimensional residue descriptors.
- **Patch and Global encoders:** local patch aggregation and edge-aware global graph attention.
- **Cross-branch attention:** Patch-derived queries and keys guide aggregation of Global values.
- **Distance geometry:** continuous RBF distance encoding with masked geometric message passing.
- **Prediction:** gated feature fusion followed by a residue-level binary classifier.

Network and training parameters are specified in `config/base.yaml`;
task-specific data paths are in `config/ppi.yaml`, `smpi.yaml`, `dpi.yaml`, and `rpi.yaml`.

## Project structure

```text
TriFuseSite/
├── config/                 # Model and training configurations
├── dataset/                # Benchmark sequences, labels, splits and PPI features
├── trifusesite/
│   ├── models/             # TriFuseSite model
│   ├── modules/            # Semantic, graph and geometric encoders
│   ├── data/               # Feature extraction, graphs and data loaders
│   ├── losses/             # Class-weighted loss
│   ├── engine.py           # Training and inference
│   ├── metrics.py          # Residue-level evaluation
│   └── statistics.py       # Paired-structure statistics
├── scripts/                # Input conversion utilities
├── tests/                  # Model and data tests
├── main_prepare_datasets.py
├── main_extract_esm.py
├── main_train.py
├── main_cross_validate.py
├── main_predict.py
└── main_evaluate.py
```

## Installation

The study environment uses Python 3.12.3 and PyTorch 2.8.0 with CUDA 12.8.

```bash
git clone https://github.com/lew123456/TriFuseSite.git
cd TriFuseSite
conda create -n trifusesite python=3.12.3
conda activate trifusesite
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e . --no-deps
```

For CPU execution, install the CPU build of PyTorch. Run the following commands
from the repository root.

## Datasets

| Task | Training | Validation | Test sets |
|---|---:|---:|---|
| PPI | 334 | Five-fold cross-validation | Test60, Test315-28 (287), Btest25, UBtest25 |
| SMPI | 1628 | 348 | Test348 |
| DPI | 458 | 115 | Test129 |
| RPI | 396 | 99 | Test117 |

Sequences, binary residue labels, split lists, the 25-pair mapping, and PPI
PSSM/HMM/DSSP/resAF descriptors are included in `dataset/`. See
[dataset/README.md](dataset/README.md) for input files and data sources.

```bash
python main_validate_datasets.py
python main_prepare_datasets.py --tasks smpi --check-only
```

## Data preparation

Place the coordinate files at the paths specified by the dataset manifests,
or supply a JSONL structure index through `--structure-index`. Public PDB and
AlphaFold DB v6 structures can be downloaded with:

```bash
python main_fetch_structures.py --tasks smpi dpi rpi --download
python main_prepare_datasets.py --tasks smpi
```

For PPI, supply the benchmark pseudopositions and distance maps using
`--ppi-geometry-root`. The [input schema](dataset/README.md#model-input-preparation) describes these arrays
and the structure index. Missing coordinates are handled through structural
masks while retaining sequence positions and labels.

Extract frozen ESM-2 layer-33 embeddings for each prepared split:

```bash
python main_extract_esm.py --manifest data/smpi/train/manifest.jsonl --allow-download
python main_extract_esm.py --manifest data/smpi/valid/manifest.jsonl --allow-download
python main_extract_esm.py --manifest data/smpi/test/manifest.jsonl --allow-download
```

ESM-2 uses `esm2_t33_650M_UR50D`, a 1022-residue window, 128-residue overlap,
and float16 caches. For offline extraction, pass
`--model /path/to/local/esm2_t33_650M_UR50D` instead of `--allow-download`.

## Training

Train SMPI, DPI, or RPI using the corresponding configuration:

```bash
python main_train.py --config config/smpi.yaml --output runs/smpi --device cuda
python main_train.py --config config/dpi.yaml --output runs/dpi --device cuda
python main_train.py --config config/rpi.yaml --output runs/rpi --device cuda
```

Default training uses AdamW, learning rate `2e-4`, weight decay `1e-4`, one
protein per batch, and gradient clipping at 1. The maximum is 20 epochs with
five-epoch patience. Checkpoint selection uses validation average precision;
the classification threshold maximizes validation MCC over 0.05–0.95 in steps
of 0.005. Command-line arguments can override the configured settings.

PPI uses five-fold selection followed by training on the full Train334 set:

```bash
python main_cross_validate.py --configs config/ppi.yaml --train-manifest data/ppi/train/manifest.jsonl --output runs/ppi_cv
python main_train.py --config config/ppi.yaml --refit --cv-selection runs/ppi_cv/selection.json --output runs/ppi
```

## Prediction and evaluation

Training writes `model.pt`, including the task configuration and classification
threshold. Pretrained TriFuseSite task checkpoints are not included in this
repository. Use a checkpoint produced by the training commands above.

```bash
python main_predict.py --checkpoint runs/smpi/model.pt --manifest data/smpi/test/manifest.jsonl --output runs/smpi/predictions.jsonl
python main_evaluate.py --checkpoint runs/smpi/model.pt --manifest data/smpi/test/manifest.jsonl --output runs/smpi/test_metrics.json
```

Evaluation reports ACC, Precision, Recall, F1, MCC, AUROC, AUPRC and Brier score,
with pooled-residue and per-protein results. Add `--intermediates-dir` to
`main_predict.py` to export `X0`, `Hgraph` and `H` representations.

## Repeated and paired experiments

```bash
python main_repeat.py --config config/ppi.yaml --output runs/ppi_repeated --dry-run
```

Remove `--dry-run` to execute the seed schedule in `config/base.yaml`. Each
bound/unbound pair is evaluated using the same checkpoint and threshold.
`main_paired_analysis.py` summarizes ten runs using the pair list
`dataset/PPI/pairs.jsonl`; run it with `--help` for the required input paths.

## Tests

```bash
python -m unittest discover -s tests -v
```
