# Adopted from https://github.com/schneiderkamplab/DeToNATION/blob/main/benchmarks/ViT/train.py

import os

import torch
from torch import distributed as dist
from tqdm import tqdm


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

    for epoch in range(1, num_epochs + 1):
        ### Training loop ###
        train_correct = 0
        model.train()
        train_sampler.set_epoch(epoch)
        train_loss_sample = torch.zeros(2).to(device)

        for inputs, targets in tqdm(
            train_loader,
            desc=f"Training epoch {epoch}",
            disable=rank > 0,
            colour="blue",
            ncols=150,
        ):
            optimizer.zero_grad()
            with model.no_sync():
                outputs = model(
                    inputs,
                    labels=targets,
                )
                loss = outputs.loss
                loss.backward()
            optimizer.step()
            # Accumulate loss and correct predictions
            train_loss_sample[0] += loss.item()
            train_loss_sample[1] += inputs.size(0)
            _, predicted = outputs.logits.max(1)
            train_correct += predicted.eq(targets).sum().item()

        # Calculate train metrics across all processes
        train_correct_tensor = torch.tensor(train_correct).to(device)
        dist.all_reduce(train_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_loss_sample, op=dist.ReduceOp.SUM)
        if rank == 0:
            global_total = train_loss_sample[1].item()
            avg_train_loss = train_loss_sample[0] / global_total
            avg_train_acc = 100.0 * train_correct_tensor.item() / global_total

        ### Validation loop ###
        val_correct = 0
        with torch.no_grad():
            model.eval()
            val_loss_sample = torch.zeros(2).to(device)

            for inputs, targets in tqdm(
                val_loader,
                desc=f"Validation epoch {epoch}",
                disable=rank > 0,
                colour="green",
                ncols=150,
            ):
                outputs = model(
                    inputs,
                    labels=targets,
                )
                loss = outputs.loss
                # Accumulate loss and correct predictions
                val_loss_sample[0] += loss.item()
                val_loss_sample[1] += inputs.size(0)
                _, predicted = outputs.logits.max(1)
                val_correct += predicted.eq(targets).sum().item()

        # Calculate val metrics across all processes
        val_correct_tensor = torch.tensor(val_correct).to(device)
        dist.all_reduce(val_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_loss_sample, op=dist.ReduceOp.SUM)
        if rank == 0:
            global_total = val_loss_sample[1].item()
            avg_val_loss = val_loss_sample[0] / global_total
            avg_val_acc = 100.0 * val_correct_tensor.item() / global_total
            print(
                f"Epoch {epoch}/{num_epochs}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Train Acc: {avg_train_acc:.2f}%, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Val Acc: {avg_val_acc:.2f}%"
            )
            run.log(
                {
                    "epoch": epoch,
                    "train/loss": avg_train_loss,
                    "train/accuracy": avg_train_acc,
                    "val/loss": avg_val_loss,
                    "val/accuracy": avg_val_acc,
                    "learning_rate": scheduler.get_last_lr()[0],
                }
            )
        scheduler.step()
    if rank == 0:
        run.finish()
    dist.destroy_process_group()
