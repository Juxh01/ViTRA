# Adopted from https://github.com/schneiderkamplab/DeToNATION/blob/main/benchmarks/ViT/train.py

import os

import torch
import wandb
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAveragePrecision,
    MulticlassCalibrationError,
    MulticlassJaccardIndex,
)
from tqdm import tqdm

from source.utils.BestModelLogger import BestModelLogger
from source.utils.BoundaryIoU import BoundaryIoU
from source.utils.HausdorffDistance95 import HausdorffDistance95


class MinClassAccuracy(MulticlassAccuracy):
    def compute(self):
        per_class_acc = super().compute()
        return torch.min(per_class_acc)


def get_metrics(task: str, device: str):
    """
    Get the appropriate metrics for the given task.
    Sets the metrics to complement each other well.
    :param task: The task type ("classification" or "segmentation").
    :param device: The device to move the metrics to.
    :return: A tuple of (train_metrics, val_metrics).
    """
    if task == "classification":
        metrics = MetricCollection(
            {
                "acc": MulticlassAccuracy(
                    num_classes=211,
                ),
                "acc_top5": MulticlassAccuracy(num_classes=211, top_k=5),
                "ece": MulticlassCalibrationError(
                    num_classes=211, n_bins=15, norm="l1"
                ),  # from "On Calibration of Modern Neural Networks"
                "map": MulticlassAveragePrecision(num_classes=211, average="macro"),
                "min_acc": MinClassAccuracy(num_classes=211, average=None),
            }
        )
    elif task == "segmentation":
        metrics = MetricCollection(
            {
                "mIoU": MulticlassJaccardIndex(num_classes=21, ignore_index=255),
            }
        )
    else:
        metrics = None
        raise ValueError(f"Unsupported task: {task}")

    train_metrics = metrics.clone(prefix="train/").to(device)
    val_metrics = metrics.clone(prefix="val/").to(device)
    if task == "segmentation":
        val_metrics.add_metrics(
            {
                "hd95": HausdorffDistance95(num_classes=21, ignore_index=255),
                "bIoU": BoundaryIoU(
                    num_classes=21,
                    ignore_index=255,
                    boundary_scale=0.02,
                    min_pixel_dist=1,
                ),
                "ece": MulticlassCalibrationError(
                    num_classes=21, n_bins=15, norm="l1", ignore_index=255
                ),
            }
        )
        val_metrics = val_metrics.to(device)
    return train_metrics, val_metrics


def train(
    device: str,
    config: dict,
    model,
    train_loader,
    val_loader,
    train_sampler,
    optimizer,
    scheduler,
    run,
) -> None:
    """
    Train the model model using the provided data loaders, optimizer, and scheduler.
    This method assumes that distributed training
    :param device: The device to use for training (e.g., "cuda" or "cpu").
    :param config: Configuration dictionary.
    :param model: The model to train.
    :param train_loader: DataLoader for the training data.
    :param val_loader: DataLoader for the validation data.
    :param train_sampler: Sampler for the training data (for distributed training).
    :param optimizer: The optimizer to use for training.
    :param scheduler: The learning rate scheduler.
    """
    rank = int(os.environ["RANK"])
    num_epochs = config["optimizer"]["epochs"]
    task = config["general"]["task"]

    train_metrics, val_metrics = get_metrics(task, device)
    train_loss_metric = MeanMetric().to(device)
    val_loss_metric = MeanMetric().to(device)

    best_model_logger = BestModelLogger(
        config=config,
        val_dataset=val_loader.dataset,
        device=device,
        num_images=8,
    )

    for epoch in range(1, num_epochs + 1):
        ### Training loop ###
        model.train()
        train_sampler.set_epoch(epoch)
        # Reset metrics and loss
        train_metrics.reset()
        train_loss_metric.reset()

        val_metrics.reset()
        val_loss_metric.reset()

        for inputs, targets in tqdm(
            train_loader,
            desc=f"Training epoch {epoch}",
            disable=rank > 0,
            colour="blue",
            ncols=150,
        ):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            if task == "segmentation" and targets.dim() == 4:
                targets = targets.squeeze(1)
            with model.no_sync():
                outputs = model(
                    inputs,
                    labels=targets,
                )
                loss = outputs.loss
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update loss and metrics
            train_loss_metric.update(value=loss.item(), weight=inputs.size(0))
            preds = outputs.logits
            if task == "segmentation":
                # Für Seg müssen wir oft interpolieren, da Output kleiner als Input sein kann
                if preds.shape[-2:] != targets.shape[-2:]:
                    preds = torch.nn.functional.interpolate(
                        preds,
                        size=targets.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

            train_metrics.update(preds, targets)

        ### Validation loop ###
        with torch.no_grad():
            model.eval()

            for inputs, targets in tqdm(
                val_loader,
                desc=f"Validation epoch {epoch}",
                disable=rank > 0,
                colour="green",
                ncols=150,
            ):
                inputs, targets = inputs.to(device), targets.to(device)
                if task == "segmentation" and targets.dim() == 4:
                    targets = targets.squeeze(1)
                outputs = model(
                    inputs,
                    labels=targets,
                )
                loss = outputs.loss
                # Metrics Update
                val_loss_metric.update(loss, weight=inputs.size(0))

                preds = outputs.logits
                if task == "segmentation":
                    if preds.shape[-2:] != targets.shape[-2:]:
                        preds = torch.nn.functional.interpolate(
                            preds,
                            size=targets.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )

                val_metrics.update(preds, targets)

        # Compute loss and metrics across all processes
        train_loss = train_loss_metric.compute().item()
        val_loss = val_loss_metric.compute().item()
        train_metrics_dict = train_metrics.compute()
        val_metrics_dict = val_metrics.compute()

        ### Check for best model and log images if needed ###
        best_model_logger.check_and_log(
            current_metric=(
                val_metrics_dict["val/mIoU"]
                if task == "segmentation"
                else val_metrics_dict["val/acc"]
            ),
            model=model,
            epoch=epoch,
            run=run,
            rank=rank,
        )
        if rank == 0:
            print(
                f"Epoch {epoch}/{num_epochs}, "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
            )
            results_dict = {
                "epoch": epoch,
                "lr": scheduler.get_last_lr()[0],
                "train/loss": train_loss,
                "val/loss": val_loss,
            }
            results_dict.update({k: v.item() for k, v in train_metrics_dict.items()})
            results_dict.update({k: v.item() for k, v in val_metrics_dict.items()})
            run.log(results_dict)
        scheduler.step()

    ### Save final model and best model ###
    best_model_logger.upload_final_artifact(run=run, rank=rank)
    # Collect the full state dict on CPU
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()

    if rank == 0:
        print("Saving final model to WandB...")
        model_filename = f"model-epoch_{num_epochs}.pt"
        torch.save(cpu_state, model_filename)

        artifact = wandb.Artifact(
            name=f"model-{run.id}",
            type="model",
            description=f"Trained model state_dict after {num_epochs} epochs",
        )

        artifact.add_file(model_filename)
        run.log_artifact(artifact)

        print("Model saved and uploaded.")
