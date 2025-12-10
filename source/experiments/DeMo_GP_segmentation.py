import json
import os
import subprocess
import sys
import warnings

import hydra
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist

from source.setup import setup_segmentation
from source.train import train


def run_worker(cfg: DictConfig) -> dict:
    """
    WORKER-MODE: Run by torchrun for distributed training.
    Performs the actual training and calculates the 4 objectives.
    """
    warnings.filterwarnings("ignore", message=".*Using a non-tuple sequence*")

    # Calculates topk based on rate * chunk, guaranteed >= 1

    cfg.optimizer.compression_topk = max(
        1, int(cfg.optimizer.compression_chunk * cfg.optimizer.compression_rate)
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # Setup
    model, train_loader, val_loader, train_sampler, optimizer, scheduler = (
        setup_segmentation(device=device, config=config_dict)
    )

    rank = int(os.environ["RANK"])
    run = None

    # Wandb on Rank 0
    if rank == 0:
        run = wandb.init(
            project=cfg.general.wandb_project,
            config=config_dict,
            name=cfg.general.experiment_name,
            reinit=True,  # For sweeps
            group="DeMo_GP_segmentation",
        )

    # Train with configuration
    mIoU, biou, avg_epoch_time, avg_aulc = train(
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

    # 6. Result Dictionary (SMAC minimizes all objectives)
    result_dict = {
        "segmentation_error": 1.0 - mIoU,
        "boundary_error": 1.0 - biou,
        "aulc_error": 1.0 - avg_aulc,
        "avg_epoch_time": avg_epoch_time,
    }

    if rank == 0:
        # Write result to file so the driver can read it
        with open("trial_result.json", "w") as f:
            json.dump(result_dict, f)
        run.finish()

    dist.destroy_process_group()
    return result_dict


def launch_worker(cfg: DictConfig) -> dict:
    """
    DRIVER-MODE: Called by hydra main when no RANK is set.
    Starts torchrun as a subprocess and returns the result to SMAC.
    """

    dist_cfg = cfg.distributed

    # Path to current script
    script_path = sys.argv[0]

    # Build torchrun command
    cmd = [
        "torchrun",
        f"--nproc_per_node={dist_cfg.nproc_per_node}",
        f"--nnodes={dist_cfg.nnodes}",
        "--rdzv-backend=c10d",
        script_path,
    ]

    # Get hydra overrides from current run
    hydra_overrides = HydraConfig.get().overrides.task
    cmd.extend(hydra_overrides)

    # Start process
    # capture_output=False lets us see the training output live
    result = subprocess.run(cmd, capture_output=False)

    # Crash Handling (Worst-Case Kosten)
    crash_result = {
        "segmentation_error": 1.0,
        "boundary_error": 1.0,
        "neg_aulc_score": 1.0,
        "avg_epoch_time": 99999.0,
    }

    if result.returncode != 0:
        print("[Driver] Trial failed via torchrun return code.")
        return crash_result

    # Read result file
    try:
        if os.path.exists("trial_result.json"):
            with open("trial_result.json", "r") as f:
                data = json.load(f)
            # Remove result file after reading
            os.remove("trial_result.json")
            return data
        else:
            print("[Driver] Error: trial_result.json not found.")
            return crash_result
    except Exception as e:
        print(f"[Driver] Error reading result: {e}")
        return crash_result


@hydra.main(config_path="../../configs", config_name="segmentation", version_base="1.1")
def main(cfg: DictConfig) -> float:
    # Decide mode based on presence of RANK environment variable
    if "RANK" in os.environ:
        return run_worker(cfg)
    else:
        return launch_worker(cfg)


if __name__ == "__main__":
    main()
