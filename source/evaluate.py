import os

import torch
import torch.nn as nn
from autoattack import AutoAttack
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, tv_tensors
from torchvision.transforms import v2 as T
from tqdm import tqdm
from transformers import ViTConfig, ViTForImageClassification

from source.train import get_metrics
from source.utils.AdvAttack import apgd_largereps

# Obtained from https://github.com/nmndeep/Robust-Segmentation/blob/f8ff8e6be040daf53cce745b6be94083f98e2a5a/semseg/utils/utils.py
VOC_WTS = [
    0.0007,
    0.0531,
    0.1394,
    0.0500,
    0.0814,
    0.0575,
    0.0256,
    0.0312,
    0.0198,
    0.0626,
    0.0382,
    0.0457,
    0.0212,
    0.0404,
    0.0421,
    0.0089,
    0.0915,
    0.0585,
    0.0366,
    0.0279,
    0.0677,
]


# Wrapper to include normalization in the model for evaluation
class NormalizationWrapper(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        x_norm = (x - self.mean) / self.std
        return self.model(x_norm).logits


def evaluate_segmentation(model, device, config, run):
    rank = int(os.environ["RANK"])
    model.eval()
    _, val_metrics = get_metrics("segmentation", device)
    val_metrics.prefix = "best/val/"
    adv_metrics = val_metrics.clone(prefix="best/adv/").to(device)

    results_dict = {}
    val_metrics.reset()
    adv_metrics.reset()
    state_dict = torch.load("best_model.pt", map_location="cpu")
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    set_model_state_dict(model, model_state_dict=state_dict, options=options)
    for param in model.parameters():
        param.requires_grad = False

    # Get new validation dataloader with no normalization for adversarial attack
    data_dir = config["general"].get("data_dir", "./data")
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
            # No normalization for adversarial attack
        ],
    )
    val_dataset = datasets.SBDataset(
        root=data_dir,
        mode="segmentation",
        image_set="val",
        download=False,
        transforms=val_transforms,
    )
    val_dataset = datasets.wrap_dataset_for_transforms_v2(val_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["adversarial"]["batch_size_per_device"],
        sampler=val_sampler,
        num_workers=int(
            os.environ.get("SLURM_CPUS_PER_TASK", 4)
        ),  # TODO: dynamic value?
        shuffle=False,
    )

    # Wrap Model
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    wrapped_model = NormalizationWrapper(model, mean, std).to(device).eval()

    ### Evaluation ###
    for images, targets in tqdm(
        val_loader, desc="Evaluating", disable=rank > 0, colour="green", ncols=150
    ):
        images = images.to(device)
        targets = targets.to(device)
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        with torch.no_grad():
            clean_outputs = wrapped_model(images.clone())
        adv_images, _, _ = apgd_largereps(
            model=wrapped_model,
            x=images,
            y=targets,
            weights=torch.tensor(VOC_WTS).to(images.device, dtype=clean_outputs.dtype),
            norm="Linf",
            eps=config["adversarial"]["epsilon"],
            n_iter=config["adversarial"]["steps"],
            loss=config["adversarial"]["loss"],
            verbose=False,
            n_restarts=config["adversarial"]["n_restarts"],
            track_loss=config["adversarial"]["loss"],
            early_stop=False,
            ignore_index=255,
            num_classes=21,
        )
        with torch.no_grad():
            adv_outputs = wrapped_model(adv_images)
        val_metrics.update(clean_outputs, targets)
        adv_metrics.update(adv_outputs, targets)

    val_metrics_dict = val_metrics.compute()
    adv_metrics_dict = adv_metrics.compute()
    if rank == 0:
        results_dict.update({k: v.item() for k, v in val_metrics_dict.items()})
        results_dict.update({k: v.item() for k, v in adv_metrics_dict.items()})
        run.log(results_dict)

    return val_metrics, adv_metrics


def evaluate_classification(device, config, run):
    rank = int(os.environ["RANK"])
    if rank == 0:
        print(f"Evaluate: Initializing fresh model for evaluation on {device}...")

    # --- RECREATE MODEL FROM CONFIG  ---
    # AutoAttack seems not to work in a ddp setting
    # TODO: Refactor both: model setup and datasets
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
        pooler_output_size=None,
        pooler_act="tanh",
        num_labels=100,
    )

    backbone_name = config["general"].get("backbone_name", None)
    if backbone_name:
        if rank == 0:
            print(f"Initializing ViT with pretrained backbone: {backbone_name}")
        model = ViTForImageClassification.from_pretrained(
            backbone_name, num_labels=100, image_size=224, ignore_mismatched_sizes=True
        )
    else:
        if rank == 0:
            print("Initializing ViT from scratch (Random Weights)")
        model = ViTForImageClassification(vit_base_config)

    # --- LOAD WEIGHTS LOCALLY ---
    if rank == 0:
        print("Evaluate: Loading weights from best_model.pt...")
    # These weights were saved with offload_to_cpu=True, so they are clean state dicts
    state_dict = torch.load("best_model.pt", map_location="cpu")
    model.load_state_dict(state_dict)

    for param in model.parameters():
        param.requires_grad = False

    model.to(device)
    model.eval()
    _, val_metrics = get_metrics("classification", device)
    val_metrics.prefix = "best/val/"
    adv_metrics = val_metrics.clone(prefix="best/adv/").to(device)

    results_dict = {}
    val_metrics.reset()
    adv_metrics.reset()

    # Wrap for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    wrapped_model = NormalizationWrapper(model, mean, std).to(device).eval()

    data_dir = config["general"].get("data_dir", "./data")
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
        ]
    )
    val_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=val_transforms,
    )
    val_dataset = datasets.wrap_dataset_for_transforms_v2(val_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["adversarial"]["batch_size_per_device"],
        sampler=val_sampler,
        num_workers=int(
            os.environ.get("SLURM_CPUS_PER_TASK", 8)
        ),  # TODO: dynamic value?
        shuffle=False,
    )

    # --- Collect Data ---
    if rank == 0:
        print("Evaluate: Collecting validation data...")
    all_images = []
    all_targets = []
    for images, targets in tqdm(
        val_loader, desc="Collecting Data", colour="blue", ncols=150
    ):
        all_images.append(images.cpu())
        all_targets.append(targets.cpu())

    x_test = torch.cat(all_images, dim=0)
    y_test = torch.cat(all_targets, dim=0)

    # Setup AutoAttack
    if rank == 0:
        adversary = AutoAttack(
            model=wrapped_model,
            norm=config["adversarial"]["norm"],
            eps=config["adversarial"]["epsilon"],
            version="custom",
            seed=config["general"]["seed"],
            device=device,
            attacks_to_run=["apgd-ce"],
            verbose=True,
        )
    else:
        adversary = AutoAttack(
            model=wrapped_model,
            norm=config["adversarial"]["norm"],
            eps=config["adversarial"]["epsilon"],
            version="custom",
            seed=config["general"]["seed"],
            device=device,
            attacks_to_run=["apgd-ce"],
            verbose=False,
        )

    if rank == 0:
        print(f"Evaluate: Running AutoAttack on {len(x_test)} images...")

    # --- Clean Evaluation (Batched) ---
    with torch.no_grad():
        bs_eval = config["adversarial"]["batch_size_per_device"]
        num_batches = (len(x_test) + bs_eval - 1) // bs_eval

        for i in range(num_batches):
            start_idx = i * bs_eval
            end_idx = min((i + 1) * bs_eval, len(x_test))
            batch_x = x_test[start_idx:end_idx].to(device)
            batch_y = y_test[start_idx:end_idx].to(device)
            clean_out = wrapped_model(batch_x)
            val_metrics.update(clean_out, batch_y)

    # --- Run Attack ---
    adv_images, adv_labels = adversary.run_standard_evaluation(
        x_test,
        y_test,
        bs=config["adversarial"]["batch_size_per_device"],
        return_labels=True,
    )

    # --- Adversarial Evaluation (Batched) ---
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * bs_eval
            end_idx = min((i + 1) * bs_eval, len(x_test))
            batch_adv = adv_images[start_idx:end_idx].to(device)
            batch_y = adv_labels[start_idx:end_idx].to(device)
            adv_out = wrapped_model(batch_adv)
            adv_metrics.update(adv_out, batch_y)

    val_metrics_dict = val_metrics.compute()
    adv_metrics_dict = adv_metrics.compute()

    results_dict.update({k: v.item() for k, v in val_metrics_dict.items()})
    results_dict.update({k: v.item() for k, v in adv_metrics_dict.items()})

    if run is not None:
        run.log(results_dict)

    return val_metrics, adv_metrics
