import json
import os
import warnings

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist

from source.setup import setup_classification
from source.train import train
from source.utils.SweepUtils import set_sweep_config


@hydra.main(
    config_path="../../configs",
    # config_name="DeMo_GP_classification",
    config_name="Classification_SBDataset",
    version_base="1.1",
)
def main(cfg: DictConfig) -> dict:
    warnings.filterwarnings("ignore", message=".*Using a non-tuple sequence*")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # Set sweep specific configuration
    config_dict = set_sweep_config(config_dict)

    # Setup
    model, train_loader, val_loader, train_sampler, optimizer, scheduler = (
        setup_classification(device=device, config=config_dict)
    )

    rank = int(os.environ["RANK"])
    run = None

    # Offline as wandb does not work currently on ucloud
    if cfg.general.log_offline:
        os.environ["WANDB_MODE"] = "offline"

    # Wandb on Rank 0
    if rank == 0:
        run = wandb.init(
            project=cfg.general.wandb_project,
            config=config_dict,
            name=cfg.general.experiment_name,
            reinit=True,  # For sweeps
            group="DeMo_GP_classification",
        )

    # Train with configuration
    acc1, avg_epoch_time, avg_aulc = train(
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

    # Result Dictionary
    result_dict = {
        "classification_error": 1.0 - acc1,
        "aulc_error": 1.0 - avg_aulc,
        "avg_epoch_time": avg_epoch_time,
    }

    if rank == 0:
        # Write result
        with open("trial_result.json", "w") as f:
            json.dump(result_dict, f)
        run.finish()

    dist.destroy_process_group()
    return result_dict


if __name__ == "__main__":
    main()
