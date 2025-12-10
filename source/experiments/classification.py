import os
import warnings

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist

from source.evaluate import evaluate_classification
from source.setup import setup_classification
from source.train import train


@hydra.main(
    config_path="../../configs", config_name="classification", version_base="1.1"
)
def main(cfg: DictConfig) -> None:
    warnings.filterwarnings("ignore", message=".*Using a non-tuple sequence*")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    # Setup the classification environment
    model, train_loader, val_loader, train_sampler, optimizer, scheduler = (
        setup_classification(device=device, config=config_dict)
    )

    rank = int(os.environ["RANK"])
    run = None
    if rank == 0:
        # Initialize Weights & Biases logging
        run = wandb.init(
            project=cfg.general.wandb_project,
            config=config_dict,
            name=cfg.general.experiment_name,
        )

    _, _, _, _ = train(
        device=device,
        config=config_dict,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_sampler=train_sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        run=run,
    )
    dist.barrier()
    if config_dict["adversarial"]["enabled"]:
        evaluate_classification(
            device=device,
            config=config_dict,
            run=run,
        )
    dist.barrier()
    if rank == 0:
        run.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
