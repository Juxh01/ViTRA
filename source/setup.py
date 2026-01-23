# Adopted from https://github.com/schneiderkamplab/DeToNATION/blob/main/benchmarks/ViT/train.py

from typing import Any, Dict

import functools
import os
import random

import numpy as np
import torch
from detonation import (
    DeMoReplicator,
    FullReplicator,
    NoReplicator,
    Optimizers,
    RandomReplicator,
    SlicingReplicator,
    StridingReplicator,
    prepare_detonation,
)
from torch import distributed as dist
from torch.distributed.fsdp import (
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim.lr_scheduler import LinearLR, PolynomialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, tv_tensors
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as T
from transformers import (
    DPTConfig,
    DPTForSemanticSegmentation,
    ViTForImageClassification,
    ViTModel,
)
from transformers.models.dpt.modeling_dpt import (
    DPTAuxiliaryHead,
    DPTFeatureFusionLayer,
    DPTReassembleLayer,
    DPTSemanticSegmentationHead,
    DPTViTEmbeddings,
    DPTViTIntermediate,
    DPTViTLayer,
    DPTViTOutput,
)
from transformers.models.vit.modeling_vit import ViTLayer

from source.utils.SBDatasetMutlilabel import SBDatasetMultiLabel

# TODO: Actiavate checkpointing based on config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.mps.is_available():
        torch.mps.manual_seed(seed)


def steup_lr_scheduler(optim, config: Dict[str, Any]):
    # TODO: Move to step based scheduler?
    optim_config = config["optimizer"]
    # task = config["general"]["task"]
    # if task == "classification":
    #     scheduler = StepLR(optim, step_size=1, gamma=optim_config["lr_gamma"])
    # elif task == "segmentation":
    total_epochs = optim_config["epochs"]
    warmup_epochs = optim_config["warmup_epochs"]
    main_scheduler = PolynomialLR(
        optim,
        total_iters=total_epochs - warmup_epochs,
        power=optim_config["lr_power"],
        last_epoch=-1,
    )
    warmup_scheduler = LinearLR(
        optim,
        start_factor=0.001,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optim,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )
    # else:
    #     raise ValueError(f"Unknown task: {task}")
    return scheduler


def get_dataset(config: Dict[str, Any], split: str, transforms=None):
    data_dir = config["general"]["data_dir"]
    dataset_name = config["general"]["dataset_name"]
    task = config["general"]["task"]
    try:
        if dataset_name == "Country211":
            dataset = datasets.Country211(
                root=data_dir,
                split=split,
                download=False,
                transform=transforms,
            )
        elif dataset_name == "SBDataset":
            if split == "valid":
                split = "val"

            if task == "classification":
                dataset = SBDatasetMultiLabel(
                    root=data_dir,
                    image_set=split,
                    download=False,
                    transforms=transforms,
                    num_classes=config["general"]["num_classes"],  # sollte 20 sein
                )
            else:
                dataset = datasets.SBDataset(
                    root=data_dir,
                    image_set=split,
                    mode="segmentation",
                    download=False,
                    transforms=transforms,
                )
        elif dataset_name == "CIFAR100":
            dataset = datasets.CIFAR100(
                root=data_dir,
                train=(split == "train"),
                download=False,
                transform=transforms,
            )
        elif dataset_name == "VOCSegmentation":
            if split == "valid":
                split = "val"
            dataset = datasets.VOCSegmentation(
                root=data_dir,
                year="2012",
                image_set=split,
                download=False,
                transforms=transforms,
            )
        elif dataset_name == "FGVCAircraft":
            if split == "train":
                split = "trainval"
            else:
                split = "test"
            dataset = datasets.FGVCAircraft(
                root=data_dir,
                split=split,
                download=False,
                transform=transforms,
                annotation_level="variant",
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    except Exception as e:
        raise RuntimeError(f"Datasets must be downloaded before setup. Error: {e}")
    # Manual wrapping for SBDataset with Multi-Label Classification
    if task == "segmentation":
        dataset = datasets.wrap_dataset_for_transforms_v2(dataset)
    return dataset


def get_transform(config: Dict[str, Any], split: str, add_normalize: bool = True):
    task = config["general"]["task"]
    dataset = config["general"]["dataset_name"]
    # if task == "classification":
    #     if split == "train":
    #         transforms = T.Compose(
    #             [
    #                 T.RandomResizedCrop(size=(224, 224), scale=(0.5, 2.0)),
    #                 T.RandomHorizontalFlip(),
    #                 T.ToImage(),
    #                 T.ToDtype(
    #                     dtype={
    #                         tv_tensors.Image: torch.float32,
    #                         tv_tensors.Mask: torch.int64,
    #                         "others": None,
    #                     },
    #                     scale=True,
    #                 ),
    #             ]
    #         )
    #     else:
    #         transforms = T.Compose(
    #             [
    #                 T.Resize(size=(224, 224)),
    #                 T.ToImage(),
    #                 T.ToDtype(
    #                     dtype={
    #                         tv_tensors.Image: torch.float32,
    #                         tv_tensors.Mask: torch.int64,
    #                         "others": None,
    #                     },
    #                     scale=True,
    #                 ),
    #             ]
    #         )

    # elif task == "segmentation":
    #     if split == "train":
    #         transforms = T.Compose(
    #             [
    #                 T.RandomShortestSize(
    #                     min_size=int(224 * 0.5),
    #                     max_size=int(224 * 2.0),  # Change 384
    #                 ),
    #                 T.RandomCrop(
    #                     size=(224, 224),  # Change 384
    #                     pad_if_needed=True,
    #                     fill=0,
    #                     padding_mode="constant",
    #                 ),
    #                 T.RandomRotation(degrees=(-15, 15)),
    #                 T.RandomHorizontalFlip(p=0.5),
    #                 T.RandomGrayscale(p=0.05),
    #                 T.ColorJitter(
    #                     brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1
    #                 ),
    #                 # T.RandomApply(
    #                 #     [T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.0))], p=0.2
    #                 # ),
    #                 T.ToImage(),
    #                 T.ToDtype(
    #                     dtype={
    #                         tv_tensors.Image: torch.float32,
    #                         tv_tensors.Mask: torch.int64,
    #                         "others": None,
    #                     },
    #                     scale=True,
    #                 ),
    #             ]
    #         )
    #     else:
    #         transforms = T.Compose(
    #             [
    #                 T.Resize(size=(224, 224)),  # Change 384
    #                 T.ToImage(),
    #                 T.ToDtype(
    #                     dtype={
    #                         tv_tensors.Image: torch.float32,
    #                         tv_tensors.Mask: torch.int64,
    #                         "others": None,
    #                     },
    #                     scale=True,
    #                 ),
    #             ],
    #         )
    # else:
    #     raise ValueError(f"Unsupported task: {task}")
    if dataset == "FGVCAircraft":
        if split == "train":
            transforms = T.Compose(
                [
                    # Use moderate crop to keep aircraft details
                    T.RandomResizedCrop(
                        size=(224, 224),
                        scale=(0.5, 1.0),
                        ratio=(0.75, 1.33),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    T.RandomHorizontalFlip(p=0.5),
                    # Use moderate RandAugment for aircraft details, but small dataset
                    T.RandAugment(
                        num_ops=2, magnitude=9, interpolation=InterpolationMode.BICUBIC
                    ),
                    T.ToTensor(),
                    T.ToDtype(
                        dtype=torch.float32,
                        scale=True,
                    ),
                ]
            )
        else:
            transforms = T.Compose(
                [
                    T.Resize(int(224 * 1.14), interpolation=InterpolationMode.BICUBIC),
                    T.CenterCrop((224, 224)),
                    T.ToTensor(),
                    T.ToDtype(
                        dtype=torch.float32,
                        scale=True,
                    ),
                ]
            )
    else:
        if split == "train":
            transforms = T.Compose(
                [
                    T.RandomShortestSize(
                        min_size=int(224 * 0.5),
                        max_size=int(224 * 2.0),  # Change 384
                    ),
                    T.RandomCrop(
                        size=(224, 224),  # Change 384
                        pad_if_needed=True,
                        fill=0,
                        padding_mode="constant",
                    ),
                    T.RandomRotation(degrees=(-15, 15)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomGrayscale(p=0.05),
                    T.ColorJitter(
                        brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1
                    ),
                    # T.RandomApply(
                    #     [T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.0))], p=0.2
                    # ),
                    T.ToImage(),
                    T.ToDtype(
                        dtype={
                            tv_tensors.Image: torch.float32,
                            tv_tensors.Mask: torch.int64,
                            "others": None,
                        },
                        scale=True,
                    ),
                ]
            )
        else:
            transforms = T.Compose(
                [
                    T.Resize(size=(224, 224)),  # Change 384
                    T.ToImage(),
                    T.ToDtype(
                        dtype={
                            tv_tensors.Image: torch.float32,
                            tv_tensors.Mask: torch.int64,
                            "others": None,
                        },
                        scale=True,
                    ),
                ],
            )
    if add_normalize:
        if task == "classification":
            normalize = T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        elif task == "segmentation":
            normalize = T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        else:
            raise ValueError(f"Unsupported task: {task}")
        transforms = T.Compose([transforms, normalize])
    return transforms


def get_ViT(config: Dict[str, Any]):
    task = config["general"]["task"]
    backbone_name = config["general"].get("backbone_name", None)
    if task == "classification":
        if backbone_name:
            # Check for multi-label classification
            dataset_name = config["general"]["dataset_name"]
            problem_type = (
                "multi_label_classification" if dataset_name == "SBDataset" else None
            )
            print(f"Initializing ViT with pretrained backbone: {backbone_name}")
            model = ViTForImageClassification.from_pretrained(
                backbone_name,
                num_labels=config["general"]["num_classes"],
                image_size=224,
                ignore_mismatched_sizes=True,
                problem_type=problem_type,
            )
        else:
            raise ValueError("Backbone name must be provided for classification task.")
    elif task == "segmentation":
        if backbone_name:
            pretrained_vit = ViTModel.from_pretrained(
                backbone_name, add_pooling_layer=False
            )
        else:
            raise ValueError("Backbone name must be provided for segmentation task.")

        # TODO: Reduce complexity of CNN head -> But check performance first
        dpt_base_config = DPTConfig(
            # --- ViT-Base Configuration (mostly default) ---
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            image_size=224,  # Change 384
            patch_size=16,
            num_channels=3,
            qkv_bias=True,
            # --- DPT Configuration ---
            is_hybrid=False,  # ViT-Base Backbone
            backbone_out_indices=[2, 5, 8, 11],  # Like in DPT-Paper
            readout_type="project",  # Default in DPT-Paper
            num_labels=config["general"]["num_classes"],  # VOC has 21 classes
            # --- Decoder Configuration (default configuration) ---
            reassemble_factors=[4, 2, 1, 0.5],
            # neck_hidden_sizes=[96, 192, 384, 768],
            neck_hidden_sizes=[48, 96, 192, 384],
            # fusion_hidden_size=256,
            fusion_hidden_size=128,
            head_in_index=-1,
            use_batch_norm_in_fusion_residual=False,
            use_bias_in_fusion_residual=True,
            add_projection=False,
            # --- Auxiliary Head (default configuration) ---
            use_auxiliary_head=True,
            auxiliary_loss_weight=0.4,
            semantic_loss_ignore_index=255,
            semantic_classifier_dropout=0.1,
            # --- Backbone-Configuration (no pretrained version) ---
            backbone_config=None,
            use_pretrained_backbone=False,
            use_timm_backbone=False,
            # --- Pooler-Configuration ---
            pooler_output_size=None,  # Defaults to hidden_size
            pooler_act="tanh",
        )

        model = DPTForSemanticSegmentation(dpt_base_config)
        print(f"Initializing DPT with pretrained backbone: {backbone_name}")

        # Manually load weights as AutoBackbone does not support DPT yet
        model.dpt.embeddings.load_state_dict(
            pretrained_vit.embeddings.state_dict(), strict=False
        )
        model.dpt.encoder.load_state_dict(
            pretrained_vit.encoder.state_dict(), strict=False
        )
        model.dpt.layernorm.load_state_dict(
            pretrained_vit.layernorm.state_dict(), strict=False
        )
    else:
        raise ValueError(f"Unsupported task: {task}")
    return model


def setup_process_group(device: str, config: Dict[str, Any] = None):
    """
    Initialize a torch.distributed process group.

    Detects SLURM and sets MASTER_ADDR/PORT, RANK, WORLD_SIZE, LOCAL_RANK.
    Respects existing torchrun env vars. Sets CUDA device when device == "cuda".
    """
    # Check for SLURM Environment (Hydra/Submitit)
    if "SLURM_PROCID" in os.environ:
        # Resolve Master Address from SLURM Nodelist
        os.environ["MASTER_ADDR"] = config["distributed"]["master_addr"]

        os.environ["MASTER_PORT"] = str(config["distributed"]["master_port"])

        # Map SLURM variables to Torch variables
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])

        nnodes = int(os.environ.get("SLURM_NNODES", "1"))
        local_world_size = world_size // nnodes if nnodes > 0 else 1
        os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)

        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)

        print(
            f"SLURM Setup: Rank {rank}/{world_size}, Local {local_rank}, Master {os.environ['MASTER_ADDR']}"
        )

    # Handle Torchrun
    elif "LOCAL_RANK" in os.environ:
        pass

    # Initialize process group
    if device == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        # Initialize process group based on set environment variables
        dist.init_process_group(backend="nccl")


def setup_distributed_training(
    model,
    transformer_layer_cls,
    seed: int,
    config: Dict[str, Any],
):
    optim_config = config["optimizer"]

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls=transformer_layer_cls
    )
    mixed_precision = (
        MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        if torch.cuda.is_bf16_supported()
        else None
    )

    if mixed_precision is None:
        print("Mixed precision with bfloat16 is not supported on this device.")

    if optim_config["repl"] == "deto-demo":
        replicator = DeMoReplicator(
            compression_topk=optim_config["compression_topk"],
            compression_chunk=optim_config["compression_chunk"],
        )
    elif optim_config["repl"] == "deto-random":
        replicator = RandomReplicator(
            compression_rate=optim_config["compression_rate"],
            seed=seed,
        )
    elif optim_config["repl"] == "deto-full":
        replicator = FullReplicator()
    elif optim_config["repl"] == "deto-stride":
        replicator = StridingReplicator(
            compression_rate=optim_config["compression_rate"],
            compression_chunk=optim_config["compression_chunk"],
        )
    elif optim_config["repl"] == "deto-slice":
        replicator = SlicingReplicator(
            compression_rate=optim_config["compression_rate"],
            compression_chunk=optim_config["compression_chunk"],
        )
    else:
        replicator = NoReplicator()
        raise NotImplementedError("NoReplicator is not wanted yet.")
    opt_enum = Optimizers(optim_config["optimizer_str"].lower())
    model, optimizer = prepare_detonation(
        model,
        opt_enum,
        replicator,
        fsdp_kwargs={
            "auto_wrap_policy": auto_wrap_policy,
            "mixed_precision": mixed_precision,
        },
        replicate_every=optim_config["replicate_every"],
        skip_every=optim_config["skip_every"],
        sharding_group_size=optim_config["shards"],
        momentum=optim_config["momentum"],
    )
    optim = optimizer._optimizer if hasattr(optimizer, "_optimizer") else optimizer
    for param_group in optim.param_groups:
        param_group["lr"] = optim_config["lr"]

    return model, optimizer


def setup_classification(device: str, config: Dict[str, Any]):
    setup_process_group(device, config)
    batch_size = config["optimizer"]["batch_size_per_device"]
    seed = config["general"]["seed"]
    set_seed(seed)

    # vit_base_config = ViTConfig(
    #     hidden_size=768,
    #     num_hidden_layers=12,
    #     num_attention_heads=12,
    #     intermediate_size=3072,
    #     hidden_act="gelu",
    #     hidden_dropout_prob=0.0,
    #     attention_probs_dropout_prob=0.0,
    #     initializer_range=0.02,
    #     layer_norm_eps=1e-12,
    #     image_size=224,
    #     patch_size=16,
    #     num_channels=3,
    #     qkv_bias=True,
    #     encoder_stride=16,
    #     pooler_output_size=None,  # Defaults to hidden_size
    #     pooler_act="tanh",
    #     num_labels=211,
    # )
    model = get_ViT(config)
    model.to(device)

    train_transforms = get_transform(config, split="train", add_normalize=True)
    val_transforms = get_transform(config, split="valid", add_normalize=True)

    train_dataset = get_dataset(config, split="train", transforms=train_transforms)

    val_dataset = get_dataset(config, split="valid", transforms=val_transforms)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=int(
            os.environ.get("SLURM_CPUS_PER_TASK", 8)
        ),  # TODO: dynamic value?
        shuffle=False,  # Shuffle is done by sampler
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=int(
            os.environ.get("SLURM_CPUS_PER_TASK", 8)
        ),  # TODO: dynamic value?
        shuffle=False,
    )

    model, optimizer = setup_distributed_training(
        model=model,
        transformer_layer_cls={ViTLayer},
        seed=seed,
        config=config,
    )
    scheduler = steup_lr_scheduler(optimizer, config)

    return model, train_loader, val_loader, train_sampler, optimizer, scheduler


def setup_segmentation(device: str, config: Dict[str, Any]):
    setup_process_group(device, config)
    batch_size = config["optimizer"]["batch_size_per_device"]
    seed = config["general"]["seed"]
    set_seed(seed)

    model = get_ViT(config)
    model.to(device)

    train_transforms = get_transform(config, split="train", add_normalize=True)
    val_transforms = get_transform(config, split="valid", add_normalize=True)

    train_dataset = get_dataset(config, split="train", transforms=train_transforms)
    val_dataset = get_dataset(config, split="valid", transforms=val_transforms)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=int(
            os.environ.get("SLURM_CPUS_PER_TASK", 8)
        ),  # TODO: dynamic value?
        shuffle=False,  # Shuffle is done by sampler
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=int(
            os.environ.get("SLURM_CPUS_PER_TASK", 8)
        ),  # TODO: dynamic value?
        shuffle=False,
    )

    # Distribute the model in small parts
    model, optimizer = setup_distributed_training(
        model=model,
        transformer_layer_cls=(
            DPTViTLayer,
            DPTViTIntermediate,
            DPTViTOutput,
            DPTFeatureFusionLayer,
            DPTReassembleLayer,
            DPTSemanticSegmentationHead,
            DPTAuxiliaryHead,
            DPTViTEmbeddings,
        ),
        seed=seed,
        config=config,
    )
    optim_config = config["optimizer"]

    # Set different learning rates for backbone and new layers
    backbone_name = config["general"].get("backbone_name", None)
    if backbone_name:
        base_lr = optim_config["lr"]
        head_lr = optim_config["lr_head"]
        backbone_params = []
        new_params = []

        # Recreate param groups
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Weights at dpt.* belong to the ViT backbone
            if name.startswith("dpt.") or "dpt." in name:
                backbone_params.append(param)
            else:
                new_params.append(param)

        assert len(backbone_params) + len(new_params) == len(
            list(model.parameters())
        ), "Parameter count mismatch!"
        # Remove all param groups
        optimizer.param_groups = []
        if backbone_params:
            optimizer.add_param_group(
                {"params": backbone_params, "lr": base_lr, "name": "backbone"}
            )

        # Gruppe 2: Random Init Layers (aggressive LR)
        if new_params:
            optimizer.add_param_group(
                {
                    "params": new_params,
                    "lr": head_lr,
                    "name": "head",
                }
            )
        print(
            f"LR Setup Complete: Pretrained params at {base_lr}, new params at {head_lr}"
        )

    scheduler = steup_lr_scheduler(optimizer, config)

    return model, train_loader, val_loader, train_sampler, optimizer, scheduler
