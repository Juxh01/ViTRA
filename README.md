# ViTRA
 **Vision Transformer Replication Analysis in Segmentation and Classification Tasks**

##  Overview

This project provides an empirical evaluation framework for analyzing Vision Transformer replication methods across segmentation and classification tasks.

**Key Features:**
- Vision Transformer implementations for segmentation and classification
- Comparative analysis of different replication strategies
- Comprehensive evaluation using rliable metrics
- Hydra-based configuration management
- Integration with Weights & Biases for experiment tracking

##  Quick Start

### Prerequisites
- Python ~3.11
- [uv](https://docs.astral.sh/uv/) (recommended package manager)
- SWIG (for some dependencies)

### Installation

```bash
# Clone the repository
git clone https://github.com/Juxh01/ViTRA.git
cd ViTRA

# Setup uv
uv venv --python 3.11

# Activate environment (linux)
source .venv/bin/activate

# Install dependencies and setup development environment
make install
```

This will:
- Install SWIG dependency
- Install the project in development mode with all dependencies
- Set up pre-commit hooks for code quality

### Usage

```bash
# Format code
make format

# Run code quality checks
make check

# Run tests
make test
```