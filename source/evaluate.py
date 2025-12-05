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
    val_dataset = datasets.VOCSegmentation(
        root=data_dir,
        year="2012",
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


def evaluate_classification(model, device, config, run):
    rank = int(os.environ["RANK"])
    model.eval()
    _, val_metrics = get_metrics("classification", device)
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

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    wrapped_model = NormalizationWrapper(model, mean, std).to(device).eval()

    adversary = AutoAttack(
        model=wrapped_model,
        norm=config["adversarial"]["norm"],
        eps=config["adversarial"]["epsilon"],
        version="standard",
        seed=config["general"]["seed"],
        device=device,
    )

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
            os.environ.get("SLURM_CPUS_PER_TASK", 4)
        ),  # TODO: dynamic value?
        shuffle=False,
    )

    ### Evaluation ###
    for images, targets in tqdm(
        val_loader, desc="Evaluating", disable=rank > 0, colour="green", ncols=150
    ):
        images = images.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            clean_outputs = wrapped_model(images.clone())
        adv_images = adversary.run_standard_evaluation(
            images, targets, bs=images.size(0)
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
