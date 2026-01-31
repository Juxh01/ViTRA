# ViTRA
**Vision Transformer Replication Analysis in Segmentation and Classification Tasks**

## Overview

This project provides an empirical evaluation framework for analyzing Vision Transformer replication methods across segmentation and classification tasks.

**Key Features:**
- Vision Transformer implementations for segmentation and classification
- Comparative analysis of different replication strategies
- Comprehensive evaluation using reliable metrics
- Hydra-based configuration management
- Integration with Weights & Biases for experiment tracking

## Quick Start

### Prerequisites
- Python ~3.11
- [uv](https://docs.astral.sh/uv/) (recommended package manager)
- SWIG (required for some dependencies)

### Installation

```bash
# Clone the repository
git clone https://github.com/Juxh01/ViTRA.git
cd ViTRA

# Setup uv virtual environment
uv venv --python 3.11

# Activate environment (Linux/macOS)
source .venv/bin/activate

# Install dependencies and setup development environment
make install
```

This will:
- Install the project in development mode (including dev dependencies)
- Install and configure pre-commit hooks for code quality

## Usage & Experiments

This project utilizes a Makefile to streamline development tasks and experiment execution. Training commands parse the provided Hydra config to configure `torchrun` for standalone or distributed execution.

### Development Commands

Routine development tasks can be executed via the following commands:

```bash
# Format code using black and isort
make format

# Run static code analysis and quality checks (ruff)
make check

# Run all pre-commit hooks
make pre-commit

# Run the test suite
make test
```

### Running Experiments

Training jobs for classification and segmentation can be launched directly via make. The system dynamically reads the provided configuration file to determine distributed training parameters (such as nproc_per_node, nnodes, or master_addr) and launches the script using torchrun.

#### Classification

Default config: `configs/classification.yaml`
```bash
make classification
```

Custom config via `CONF_CLS`:
```bash
make classification CONF_CLS=configs/classification.yaml
```

#### Segmentation

Default config: `configs/segmentation.yaml`
```bash
make segmentation
```

Custom config via `CONF_SEG`:
```bash
make segmentation CONF_SEG=configs/segmentation.yaml
```

#### Ablation sweeps (assumes a working SLURM cluster)

These ablation runs are launched via Hydra multirun (`-m`) and assume a working SLURM cluster setup. The node and GPU configuration should be specified in the respective YAML config files in the `distributed` section.

- FGVC Aircraft ablation (config: `configs/Classification_FGVGAircraft.yaml`)
```bash
make sweep-ablation-fgvc
```

- SBDataset ablation (classification) (config: `configs/Classification_SBDataset.yaml`)
```bash
make sweep-ablation-sb-cls
```

- SBDataset ablation (segmentation) (config: `configs/Segmentation_SBDataset.yaml`)
```bash
make sweep-ablation-sb-seg
```

### Distributed Training Configuration

The execution environment relies on a `distributed` section within the YAML config. The Makefile reads these values to construct the `torchrun` launch command.

Example:
```yaml
distributed:
  nproc_per_node: 2    # GPUs per node
  nnodes: 1            # Number of nodes
  node_rank: 0         # Rank of the current node
  master_addr: "localhost"
  master_port: 29500
  standalone: true     # Set to false for multi-node setups
```