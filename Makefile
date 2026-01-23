NAME := ViTRA
PACKAGE_NAME := vitra

DIR := "${CURDIR}"
SOURCE_DIR := ${PACKAGE_NAME}
DIST := dist
TESTS_DIR := tests
# Path to standard configuration files
CONF_CLS ?= configs/classification.yaml
CONF_SEG ?= configs/segmentation.yaml

# Defaults 
PYTHON ?= python
TORCHRUN ?= torchrun
PYTEST ?= uv run pytest
PIP ?= uv pip
MAKE ?= make
PRECOMMIT ?= uv run pre-commit
RUFF ?= uv run ruff
PIP2 ?= pip

# Command builder
define build_torchrun_cmd
	$(PYTHON) -c "from omegaconf import OmegaConf; \
	conf = OmegaConf.load('$(1)'); \
	dist = conf.distributed; \
	cmd = f'--nproc_per_node={dist.nproc_per_node} --nnodes={dist.nnodes}'; \
	cmd += ' --standalone' if dist.get('standalone', False) else f' --node_rank={dist.node_rank} --master_addr={dist.master_addr} --master_port={dist.master_port}'; \
	print(cmd)"
endef

.PHONY: help install check format pre-commit clean clean-build build publish test
.PHONY: classification segmentation setup-master setup-worker sweep-segmentation

help:
	@echo "Makefile ${NAME}"
	@echo "* install          to install all requirements and install pre-commit"
	@echo "* clean            to clean any doc or build files"
	@echo "* check            to check the source code for issues"
	@echo "* format           to format the code with black and isort"
	@echo "* classification   to run classification training"
	@echo "* segmentation     to run segmentation training"
	@echo "* setup-master     to generate SSH keys for the cluster (Run on Node 0)"
	@echo "* setup-worker     to authorize the master key (Run on Node 1)"

install:
	$(PIP) install swig
	$(PIP) install -e ".[dev]"
	@mkdir -p .dependencies
	@if [ ! -d ".dependencies/hydra-smac-sweeper" ]; then \
		echo ">>> Cloning hydra-smac-sweeper source..."; \
		git clone https://github.com/automl/hydra-smac-sweeper.git .dependencies/hydra-smac-sweeper; \
	fi
	@echo ">>> Re-installing SMAC Sweeper with 'editable_mode=compat'..."
	$(PIP) install -e .dependencies/hydra-smac-sweeper --config-settings editable_mode=compat
	$(PRECOMMIT) install

check:
	$(RUFF) format --check source tests
	$(RUFF) check source tests

pre-commit:
	$(PRECOMMIT) run --all-files

format:
	uv run isort source tests
	$(RUFF) format --silent source tests
	$(RUFF) check --fix --silent source tests --exit-zero
	$(RUFF) check --fix source tests --exit-zero

test:
	$(PYTEST) ${TESTS_DIR}

classification:
	@echo "Starte Classification Training mit Config: $(CONF_CLS)"
	$(eval TORCH_FLAGS := $(shell $(call build_torchrun_cmd,$(CONF_CLS))))
	$(TORCHRUN) $(TORCH_FLAGS) source/experiments/classification.py 

segmentation:
	@echo "Starte Segmentation Training mit Config: $(CONF_SEG)"
	$(eval TORCH_FLAGS := $(shell $(call build_torchrun_cmd,$(CONF_SEG))))
	$(TORCHRUN) $(TORCH_FLAGS) source/experiments/segmentation.py 

sweep-classification:
	@echo "Starting SMAC Sweep with SLURM..."
	$(PYTHON) source/experiments/DeMo_GP_classification.py \
		-m

sweep-segmentation:
	@echo "Starting SMAC Sweep with SLURM..."
	$(PYTHON) source/experiments/DeMo_GP_segmentation.py \
		-m

sweep-ablation-fgvc:
	@echo "Starte Ablation Sweep für FGVC Aircraft..."
	$(PYTHON) source/experiments/Ablation_Sweep.py --config-name=Classification_FGVGAircraft -m

sweep-ablation-sb-cls:
	@echo "Starte Ablation Sweep für SBDataset Classification..."
	$(PYTHON) source/experiments/Ablation_Sweep.py --config-name=Classification_SBDataset -m

sweep-ablation-sb-seg:
	@echo "Starte Ablation Sweep für SBDataset Segmentation..."
	$(PYTHON) source/experiments/Ablation_Sweep.py --config-name=Segmentation_SBDataset -m