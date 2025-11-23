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
    prepare_detonation,
)
from torch import distributed as dist
from torch.distributed.fsdp import (
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, tv_tensors
from torchvision.transforms import v2 as T
from transformers import (
    DPTConfig,
    DPTForSemanticSegmentation,
    ViTConfig,
    ViTForImageClassification,
)
from transformers.models.dpt.modeling_dpt import (
    DPTAuxiliaryHead,
    DPTFeatureFusionLayer,
    DPTPreActResidualLayer,
    DPTReassembleLayer,
    DPTSemanticSegmentationHead,
    DPTViTLayer,
)
from transformers.models.vit.modeling_vit import ViTLayer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.mps.is_available():
        torch.mps.manual_seed(seed)


def setup_process_group(device: str):
    # Set the device for the current process
    if device == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    # Initialize the process group for distributed training
    if not dist.is_initialized():
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
    )
    optim = optimizer._optimizer if hasattr(optimizer, "_optimizer") else optimizer
    for param_group in optim.param_groups:
        param_group["lr"] = optim_config["lr"]
    scheduler = StepLR(optim, step_size=1, gamma=optim_config["lr_gamma"])

    return model, optimizer, scheduler


def setup_classification(device: str, config: Dict[str, Any]):
    setup_process_group(device)
    batch_size = config["optimizer"]["batch_size_per_device"]
    seed = config["general"]["seed"]
    set_seed(seed)

    # Get rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    data_dir = config["general"].get("data_dir", "./data")

    vit_base_config = ViTConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=224,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        encoder_stride=16,
        pooler_output_size=None,  # Defaults to hidden_size
        pooler_act="tanh",
        num_labels=100,
    )
    backbone_name = config["general"].get("backbone_name", None)
    if backbone_name:
        print(f"Initializing ViT with pretrained backbone: {backbone_name}")
        model = ViTForImageClassification.from_pretrained(
            backbone_name, num_labels=100, image_size=224, ignore_mismatched_sizes=True
        )
    else:
        print("Initializing ViT from scratch (Random Weights)")
        model = ViTForImageClassification(vit_base_config)
    model.to(device)

    train_transforms = T.Compose(
        [
            T.RandomResizedCrop(size=(224, 224), scale=(0.5, 2.0)),
            T.RandomHorizontalFlip(),
            T.ToImage(),
            T.ToDtype(
                dtype={
                    tv_tensors.Image: torch.float32,
                    "others": None,
                },
                scale=True,
            ),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transforms = T.Compose(
        [
            T.Resize(size=(224, 224)),
            T.ToImage(),
            T.ToDtype(
                dtype={
                    tv_tensors.Image: torch.float32,
                    "others": None,
                },
                scale=True,
            ),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if local_rank == 0:
        print("Downloading CIFAR100 dataset...")
        datasets.CIFAR100(
            root=data_dir,
            train=True,
            download=True,
        )
        datasets.CIFAR100(
            root=data_dir,
            train=False,
            download=True,
        )

    dist.barrier()  # Ensure that only one process downloads the dataset

    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transforms,
    )
    train_dataset = datasets.wrap_dataset_for_transforms_v2(train_dataset)
    val_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=val_transforms,
    )
    val_dataset = datasets.wrap_dataset_for_transforms_v2(val_dataset)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=int(
            os.environ.get("SLURM_CPUS_PER_TASK", 4)
        ),  # TODO: dynamic value?
        shuffle=False,  # Shuffle is done by sampler
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=int(
            os.environ.get("SLURM_CPUS_PER_TASK", 4)
        ),  # TODO: dynamic value?
        shuffle=False,
    )

    model, optimizer, scheduler = setup_distributed_training(
        model=model,
        transformer_layer_cls={ViTLayer},
        seed=seed,
        config=config,
    )

    return model, train_loader, val_loader, train_sampler, optimizer, scheduler


def setup_segmentation(device: str, config: Dict[str, Any]):
    setup_process_group(device)
    batch_size = config["optimizer"]["batch_size_per_device"]
    seed = config["general"]["seed"]
    set_seed(seed)

    # Get rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    data_dir = config["general"].get("data_dir", "./data")

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
        image_size=384,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        # --- DPT Configuration ---
        is_hybrid=False,  # ViT-Base Backbone
        backbone_out_indices=[2, 5, 8, 11],  # Like in DPT-Paper
        readout_type="project",  # Default in DPT-Paper
        num_labels=21,  # VOC has 21 classes
        # --- Decoder Configuration (default configuration) ---
        reassemble_factors=[4, 2, 1, 0.5],
        neck_hidden_sizes=[96, 192, 384, 768],
        fusion_hidden_size=256,
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
        backbone=None,
        use_pretrained_backbone=False,
        use_timm_backbone=False,
        backbone_kwargs=None,
        # --- Pooler-Configuration ---
        pooler_output_size=None,  # Defaults to hidden_size
        pooler_act="tanh",
    )
    backbone_name = config["general"].get("backbone_name", None)
    if backbone_name:
        print(f"Initializing DPT with pretrained backbone: {backbone_name}")
        dpt_base_config.use_pretrained_backbone = True
        dpt_base_config.backbone = backbone_name
    else:
        print("Initializing DPT from scratch (Random Weights)")

    model = DPTForSemanticSegmentation(dpt_base_config)
    model.to(device)

    train_transforms = T.Compose(
        [
            T.RandomResizedCrop(size=(384, 384), scale=(0.5, 2.0)),
            T.RandomHorizontalFlip(),
            T.ToImage(),
            T.ToDtype(
                dtype={
                    tv_tensors.Image: torch.float32,
                    tv_tensors.Mask: torch.int64,
                    "others": None,
                },
                scale=True,
            ),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transforms = T.Compose(
        [
            T.Resize(size=(384, 384)),
            T.ToImage(),
            T.ToDtype(
                dtype={
                    tv_tensors.Image: torch.float32,
                    tv_tensors.Mask: torch.int64,
                    "others": None,
                },
                scale=True,
            ),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )

    if local_rank == 0:
        print("Downloading VOCSegmentation dataset...")
        datasets.VOCSegmentation(
            root=data_dir,
            year="2012",
            image_set="train",
            download=True,
        )
        datasets.VOCSegmentation(
            root=data_dir,
            year="2012",
            image_set="val",
            download=True,
        )

    dist.barrier()  # Ensure that only one process downloads the dataset

    train_dataset = datasets.VOCSegmentation(
        root=data_dir,
        year="2012",
        image_set="train",
        download=True,
        transform=train_transforms,
    )
    train_dataset = datasets.wrap_dataset_for_transforms_v2(train_dataset)
    val_dataset = datasets.VOCSegmentation(
        root=data_dir,
        year="2012",
        image_set="val",
        download=True,
        transform=val_transforms,
    )
    val_dataset = datasets.wrap_dataset_for_transforms_v2(val_dataset)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=int(
            os.environ.get("SLURM_CPUS_PER_TASK", 4)
        ),  # TODO: dynamic value?
        shuffle=False,  # Shuffle is done by sampler
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=int(
            os.environ.get("SLURM_CPUS_PER_TASK", 4)
        ),  # TODO: dynamic value?
        shuffle=False,
    )
    model, optimizer, scheduler = setup_distributed_training(
        model=model,
        transformer_layer_cls=(
            DPTViTLayer,
            DPTFeatureFusionLayer,
            DPTAuxiliaryHead,
            DPTPreActResidualLayer,
            DPTReassembleLayer,
            DPTSemanticSegmentationHead,
        ),
        seed=seed,
        config=config,
    )

    return model, train_loader, val_loader, train_sampler, optimizer, scheduler
