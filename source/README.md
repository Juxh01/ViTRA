# `source/` — Training & Evaluation Code

This folder contains the core training/evaluation pipeline, plus supporting utilities and runnable experiment entrypoints.

## Core modules

### `setup.py` ([source/setup.py](setup.py))
Environment and training setup utilities:
- **Reproducibility & schedulers:** [`source.setup.set_seed`](setup.py), [`source.setup.steup_lr_scheduler`](setup.py)
- **Data:** dataset construction via [`source.setup.get_dataset`](setup.py) and transforms via [`source.setup.get_transform`](setup.py)
- **Models:** Vision Transformer setup via [`source.setup.get_ViT`](setup.py) (classification + segmentation variants)
- **Distributed / FSDP + replication:** process group init via [`source.setup.setup_process_group`](setup.py) and sharded/replicated training via [`source.setup.setup_distributed_training`](setup.py)
- **Task entry setup:** [`source.setup.setup_classification`](setup.py) and [`source.setup.setup_segmentation`](setup.py) return `(model, loaders, sampler, optimizer, scheduler)`.

### `train.py` ([source/train.py](train.py))
One unified training loop used by both tasks:
- Metrics factory: [`source.train.get_metrics`](train.py) (classification metrics, segmentation metrics + extras)
- Training loop: [`source.train.train`](train.py)
  - Runs epoch train/val loops, logs to W&B, tracks the “best” metric (accuracy for classification, mIoU for segmentation).
  - Saves best model via [`source.utils.BestModelLogger.BestModelLogger`](utils/BestModelLogger.py) and saves final model state with FSDP full-state dict.

### `evaluate.py` ([source/evaluate.py](evaluate.py))
Adversarial/robustness evaluation utilities:
- Implements segmentation and classification evaluation entrypoints: [`source.evaluate.evaluate_segmentation`](evaluate.py), [`source.evaluate.evaluate_classification`](evaluate.py)
- Uses a normalization wrapper: [`source.evaluate.NormalizationWrapper`](evaluate.py)
- **Note:** `evaluate.py` was **not used in the final report** and is kept for completeness; the file header notes that adversarial evaluation was not meaningful due to model collapse, and that attacks should be integrated into training.

## `utils/` ([source/utils/](utils))
Supporting functionality used by the training/evaluation pipeline:
- **Adversarial attacks / losses:** [`source.utils.AdvAttack.apgd_largereps`](utils/AdvAttack.py), [`source.utils.AdvAttack.apgd_restarts`](utils/AdvAttack.py), [`source.utils.AdvAttack.apgd_train`](utils/AdvAttack.py), plus helpers like [`source.utils.AdvAttack.compute_iou_acc`](utils/AdvAttack.py).
- **Best-checkpoint logging + visualization:** [`source.utils.BestModelLogger.BestModelLogger`](utils/BestModelLogger.py) (saves best model and optionally logs segmentation overlays to W&B).
- **Segmentation metrics not in TorchMetrics:**
  - Boundary IoU: [`source.utils.BoundaryIoU.BoundaryIoU`](utils/BoundaryIoU.py)
  - HD95: [`source.utils.HausdorffDistance95.HausdorffDistance95`](utils/HausdorffDistance95.py)
- **Sweep support:** [`source.utils.SweepUtils.set_sweep_config`](utils/SweepUtils.py) adjusts config fields for ablation sweeps.
- **Dataset extension:** [`source.utils.SBDatasetMutlilabel.SBDatasetMultiLabel`](utils/SBDatasetMutlilabel.py) adapts SBDataset for multi-label classification.

## `experiments/` ([source/experiments/](experiments))
Hydra entrypoints that glue config → setup → training → optional evaluation:
- Classification run: [source/experiments/classification.py](experiments/classification.py)
- Segmentation run: [source/experiments/segmentation.py](experiments/segmentation.py)
- Ablation sweeps (Hydra multirun): [source/experiments/Ablation_Sweep.py](experiments/Ablation_Sweep.py) (uses [`source.utils.SweepUtils.set_sweep_config`](utils/SweepUtils.py))

These scripts typically call:
1. [`source.setup.setup_classification`](setup.py) / [`source.setup.setup_segmentation`](setup.py)
2. [`source.train.train`](train.py)
3. Optionally [`source.evaluate.evaluate_classification`](evaluate.py) / [`source.evaluate.evaluate_segmentation`](evaluate.py)