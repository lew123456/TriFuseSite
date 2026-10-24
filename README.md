# TriFuseSite

**A Unified Multisource Framework for Conformation-Robust Protein Binding-Site Prediction**

[Data preparation](dataset/README.md) · [Command guide](trifusesite/cli/README.md) · [Configurations](config/)

## 🧬 Abstract

TriFuseSite is a multisource framework for residue-level protein binding-site
prediction across protein–protein (PPI), protein–small molecule (SMPI),
protein–DNA (DPI), and protein–RNA (RPI) interactions. It combines frozen ESM-2
sequence representations, residue descriptors, and multiscale structural
information to integrate sequence context with local and global geometry.
All four tasks use the same network architecture, with task-specific models
trained independently.

## 🧠 Model Architecture

| Component | Role |
|---|---|
| Semantic fusion | Integrates 1,280-dimensional ESM-2 embeddings with 62-dimensional residue descriptors through a learnable scalar gate. |
| Local Patch branch | Aggregates local residue neighborhoods using three patch-encoding layers. |
| Global branch | Models residue-graph context with eight edge-aware graph-attention layers. |
| Cross-branch attention | Uses Patch-derived queries and keys to aggregate Global value representations. |
| Distance geometry | Encodes continuous distances with RBF features and two geometric attention layers. |
| Residue classifier | Combines graph and geometric representations to predict a binding probability for each residue. |

The implementation is organized into [model definitions](trifusesite/models/)
and [network modules](trifusesite/modules/). Shared architecture and training
settings are defined in [config/base.yaml](config/base.yaml).

## 📁 Project Structure

```text
TriFuseSite/
├── config/                 # Shared settings and task-specific YAML files
├── dataset/                # Benchmark sequences, labels, splits and PPI descriptors
├── scripts/                # Data conversion and Bash/PowerShell launchers
├── tests/                  # Model, data and command-line tests
├── trifusesite/            # Core Python package
│   ├── cli/                # Command implementations and detailed usage guide
│   ├── data/               # Sequence, structure and graph preprocessing
│   ├── losses/             # Weighted residue-classification loss
│   ├── models/             # TriFuseSite network definition
│   ├── modules/            # Semantic, Patch/Global and geometry encoders
│   ├── engine.py           # Training, checkpoints and inference
│   ├── metrics.py          # Residue-level evaluation metrics
│   └── statistics.py       # Paired-structure statistical analysis
├── main.py                 # Unified command-line entry point
├── pyproject.toml          # Package metadata and console command
└── requirements.txt        # Python dependencies
```

## 🛠️ Installation

```bash
git clone https://github.com/lew123456/TriFuseSite.git
cd TriFuseSite
conda create -n trifusesite python=3.12.3
conda activate trifusesite
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e . --no-deps
```

The study environment uses PyTorch 2.8.0 with CUDA 12.8. For CPU execution,
install the CPU build of PyTorch instead.

## 📊 Benchmark Datasets

| Task | Training | Validation | Test sets | Configuration |
|---|---:|---|---|---|
| PPI | 334 | Five-fold cross-validation | Test60, Test315-28 (287), Btest25, UBtest25 | [ppi.yaml](config/ppi.yaml) |
| SMPI | 1,628 | 348 | Test348 | [smpi.yaml](config/smpi.yaml) |
| DPI | 458 | 115 | Test129 | [dpi.yaml](config/dpi.yaml) |
| RPI | 396 | 99 | Test117 | [rpi.yaml](config/rpi.yaml) |

Benchmark records, split lists, paired-structure mappings and PPI descriptors
are organized in [dataset/](dataset/). The [data guide](dataset/README.md)
describes file formats, structure mapping and feature preparation.

## 🚀 Getting Started

Run commands from the repository root. First follow the
[data preparation workflow](dataset/README.md#preprocessing-workflow), then
select the configuration for the interaction task.

```bash
python main.py train \
    --config config/smpi.yaml \
    --output runs/smpi \
    --device cuda
```

The [command guide](trifusesite/cli/README.md) covers training outputs,
prediction, evaluation, PPI cross-validation and paired-structure experiments.

```bash
python main.py --help
python main.py train --help
```

After installation, `trifusesite` can also be used in place of `python main.py`.

## 🧪 Tests

```bash
python -m unittest discover -s tests -v
```
