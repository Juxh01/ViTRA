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

from source.setup import setup_classification
from source.train import train


def run_worker(cfg: DictConfig) -> dict:
    """
    WORKER-MODE: Run by torchrun for distributed training.
    Performs the actual training and calculates the 3 objectives.
    """
    warnings.filterwarnings("ignore", message=".*Using a non-tuple sequence*")

    # Calculates topk based on rate * chunk, guaranteed >= 1
    cfg.optimizer.compression_topk = max(
        1, int(cfg.optimizer.compression_chunk * cfg.optimizer.compression_rate)
    )

    cfg.optimizer.compression_chunk = 2 ** int(cfg.optimizer.compression_chunk_factor)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # Setup
    model, train_loader, val_loader, train_sampler, optimizer, scheduler = (
        setup_classification(device=device, config=config_dict)
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

    # 6. Result Dictionary (SMAC minimizes all objectives)
    result_dict = {
        "classification_error": 1.0 - acc1,
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
    DRIVER-MODE: Starts torchrun locally AND on the remote node via SSH (if nnodes > 1).
    """
    dist_cfg = cfg.distributed

    # Prepare Command Base
    # We must explicitly set the rendezvous endpoint for multi-node
    rdzv_endpoint = f"{dist_cfg.master_addr}:{dist_cfg.master_port}"

    base_cmd = [
        "torchrun",
        f"--nproc_per_node={dist_cfg.nproc_per_node}",
        f"--nnodes={dist_cfg.nnodes}",
        "--rdzv-backend=c10d",
        f"--rdzv-endpoint={rdzv_endpoint}",
        sys.argv[0],  # The script path
    ]

    # Append all Hydra overrides (the SMAC params)
    hydra_overrides = HydraConfig.get().overrides.task
    base_cmd.extend(hydra_overrides)

    # Construct Local Command (Rank 0)
    cmd_rank0 = base_cmd.copy()
    cmd_rank0.insert(1, "--node_rank=0")  # Insert rank before script path

    print(f"[Driver] Launching Local Rank 0: {' '.join(cmd_rank0)}")
    proc_local = subprocess.Popen(cmd_rank0)
    proc_remote = None

    # Construct Remote Command (Rank 1) via SSH
    node1_address = cfg.distributed.node1_addr
    # Ensure the python env is correct on remote.
    remote_python_cmd = " ".join(base_cmd)

    # Insert rank 1 for remote
    remote_python_cmd = remote_python_cmd.replace("torchrun", "torchrun --node_rank=1")

    cmd_rank1 = [
        "ssh",
        node1_address,
        f"cd {os.getcwd()} && source .venv/bin/activate && {remote_python_cmd}",
    ]

    print(f"[Driver] Launching Remote Rank 1: {' '.join(cmd_rank1)}")
    proc_remote = subprocess.Popen(cmd_rank1)

    # Wait for completion
    proc_local.wait()
    if proc_remote:
        proc_remote.wait()

    # Crash Handling & Result Retrieval
    crash_result = {
        "classification_error": 1.0,  # CHANGED: Key to match task
        "aulc_error": 1.0,
        "avg_epoch_time": 99999.0,
    }

    if proc_local.returncode != 0:
        print("[Driver] Local trial failed.")
        return crash_result

    # Note: If Node 1 fails but Node 0 succeeds, PyTorch DDP usually crashes Node 0 too.
    # If Node 0 finishes writing the json, we assume success.

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


# CHANGED: Config name points to classification yaml
@hydra.main(
    config_path="../../configs",
    config_name="DeMo_GP_classification",
    version_base="1.1",
)
def main(cfg: DictConfig) -> float:
    # Decide mode based on presence of RANK environment variable
    if "RANK" in os.environ:
        return run_worker(cfg)
    else:
        return launch_worker(cfg)


if __name__ == "__main__":
    main()
