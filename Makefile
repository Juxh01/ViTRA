NAME := ViTRA
PACKAGE_NAME := vitra

DIR := "${CURDIR}"
SOURCE_DIR := ${PACKAGE_NAME}
DIST := dist
TESTS_DIR := tests
# Path to standard configuration files
CONF_CLS ?= configs/classification.yaml
CONF_SEG ?= configs/segmentation.yaml

.PHONY: help install check format pre-commit clean clean-build build publish test
.PHONY: classification segmentation

help:
	@echo "Makefile ${NAME}"
	@echo "* install      	  to install all equirements and install pre-commit"
	@echo "* clean            to clean any doc or build files"
	@echo "* check            to check the source code for issues"
	@echo "* format           to format the code with black and isort"
	@echo "* pre-commit       to run the pre-commit check"
	@echo "* build            to build a dist"
	@echo "* publish          to help publish the current branch to pypi"
	@echo "* test             to run the tests"

PYTHON ?= python
PYTEST ?= uv run pytest
PIP ?= uv pip
MAKE ?= make
PRECOMMIT ?= uv run pre-commit
RUFF ?= uv run ruff
# Default torchrun command
TORCHRUN ?= torchrun

define build_torchrun_cmd
	$(PYTHON) -c "from omegaconf import OmegaConf; \
	conf = OmegaConf.load('$(1)'); \
	dist = conf.distributed; \
	cmd = f'--nproc_per_node={dist.nproc_per_node} --nnodes={dist.nnodes}'; \
	cmd += ' --standalone' if dist.get('standalone', False) else f' --node_rank={dist.node_rank} --master_addr={dist.master_addr} --master_port={dist.master_port}'; \
	print(cmd)"
endef

install:
	$(PIP) install swig
	$(PIP) install -e ".[dev]"
	pre-commit install

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
	


