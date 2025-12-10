NAME := ViTRA
PACKAGE_NAME := vitra

DIR := "${CURDIR}"
SOURCE_DIR := ${PACKAGE_NAME}
DIST := dist
TESTS_DIR := tests
# Path to standard configuration files
CONF_CLS ?= configs/classification.yaml
CONF_SEG ?= configs/segmentation.yaml

# Defaults (Assumes you have activated your environment)
PYTHON ?= python
TORCHRUN ?= torchrun
PYTEST ?= uv run pytest
PIP ?= uv pip
MAKE ?= make
PRECOMMIT ?= uv run pre-commit
RUFF ?= uv run ruff

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

sweep-segmentation:
ifndef MASTER
	$(error MASTER is undefined. Run: make sweep MASTER=192.168.1.X NODE1=user@192.168.1.Y)
endif
ifndef NODE1
	$(error NODE1 is undefined. Run: make sweep MASTER=192.168.1.X NODE1=user@192.168.1.Y)
endif
	@echo "Starting SMAC Sweep Driver on Master Node..."
	@echo "Master IP: $(MASTER)"
	@echo "Node 1:    $(NODE1)"
	$(PYTHON) source/experiments/DeMo_GP_segmentation.py --config-name DeMo_GP_segmentation \
		distributed.master_addr=$(MASTER) \
		distributed.node1_addr=$(NODE1)
# --- SSH Cluster Automation ---

# Run this on Node 0 (Master)
setup-master:
	@echo ">>> Generating SSH Key..."
	@mkdir -p $(HOME)/.ssh
	@chmod 700 $(HOME)/.ssh
	# Generate key if it doesn't exist, preventing overwrite prompts
	@if [ ! -f $(HOME)/.ssh/id_cluster ]; then \
		ssh-keygen -t ed25519 -f $(HOME)/.ssh/id_cluster -N ""; \
	fi
	@echo ">>> Configuring SSH..."
	@touch $(HOME)/.ssh/config
	@chmod 600 $(HOME)/.ssh/config
	@if ! grep -q "IdentityFile $(HOME)/.ssh/id_cluster" $(HOME)/.ssh/config; then \
		echo "IdentityFile $(HOME)/.ssh/id_cluster" >> $(HOME)/.ssh/config; \
	fi
	@echo ""
	@echo ">>> SUCCESS. COPY THE KEY BELOW TO RUN ON WORKER:"
	@echo ""
	@cat $(HOME)/.ssh/id_cluster.pub
	@echo ""

# Run this on Node 1 (Worker)
# Usage: make setup-worker KEY="ssh-ed25519 AAAA..."
setup-worker:
ifndef KEY
	$(error KEY is undefined. Run: make setup-worker KEY="paste_key_here")
endif
	@echo ">>> Authorizing Master Key..."
	@mkdir -p $(HOME)/.ssh
	@chmod 700 $(HOME)/.ssh
	@echo "$(KEY)" >> $(HOME)/.ssh/authorized_keys
	@chmod 600 $(HOME)/.ssh/authorized_keys
	@echo ">>> Worker Configured."