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
- Install SWIG dependency
- Install the project in development mode with all dependencies
- Set up pre-commit hooks for code quality

## Usage & Experiments

This project utilizes a Makefile to streamline development tasks and experiment execution. The training commands automatically parse Hydra configuration files to configure torchrun for both standalone and distributed environments.

### Development Commands

Routine development tasks can be executed via the following commands:

```bash
# Format code using black and isort
make format

# Run static code analysis and quality checks (ruff)
make check

# Run the test suite
make test
```

### Running Experiments

Training jobs for classification and segmentation can be launched directly via make. The system dynamically reads the provided configuration file to determine distributed training parameters (such as nproc_per_node, nnodes, or master_addr) and launches the script using torchrun.

#### Classification

To run the default classification experiment defined in `configs/classification.yaml`:

```bash
make classification
```

To run a classification experiment with a custom configuration file (e.g., for a specific cluster node or experimental setup):

```bash
make classification CONF_CLS=configs/experiments/my_cluster_config.yaml
```

#### Segmentation

To run the default segmentation experiment defined in `configs/segmentation.yaml`:

```bash
make segmentation
```

To run a segmentation experiment with a custom configuration file:

```bash
make segmentation CONF_SEG=configs/experiments/my_segmentation_config.yaml
```

### Distributed Training Configuration

The execution environment relies on a `distributed` section within your YAML configuration files to set up the process group. The Makefile reads these values to construct the launch command.

Ensure your configuration files include the necessary keys if you are running in a multi-node environment. Example configuration structure:

```yaml
distributed:
  nproc_per_node: 2    # GPUs per node
  nnodes: 1            # Number of nodes
  node_rank: 0         # Rank of the current node
  master_addr: "localhost"
  master_port: 29500
  standalone: true     # Set to false for multi-node setups
```