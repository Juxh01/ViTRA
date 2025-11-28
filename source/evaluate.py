import os

import torch
from torch import distributed as dist
from torch.distributed.checkpoint.state_dict import set_model_state_dict
from tqdm import tqdm

from source.train import get_metrics
from source.utils.AdvAttack import apgd_largereps

VOC_WTS = [1.0] * 21


def evaluate_segmentation(model, dataloader, device, config, run):
    rank = int(os.environ["RANK"])
    model.eval()
    _, val_metrics = get_metrics("segmentation", device)
    val_metrics.prefix = "best/val/"
    adv_metrics = val_metrics.clone(prefix="best/adv/").to(device)

    results_dict = {}
    val_metrics.reset()
    adv_metrics.reset()
    if rank == 0:
        state_dict = torch.load(os.path.join("best_model.pt"))
    dist.barrier()
    set_model_state_dict(model, model_state_dict=state_dict)
    with torch.no_grad():
        for images, targets in tqdm(
            dataloader, desc="Evaluating", disable=rank > 0, colour="green", ncols=150
        ):
            images = images.to(device)
            targets = targets.to(device)
            clean_outputs = model(images.clone())
            adv_images, _, _ = apgd_largereps(
                model=model,
                x=images,
                y=targets,
                weights=torch.tensor(VOC_WTS).to(images.device),
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
            adv_outputs = model(adv_images)
            val_metrics.update(clean_outputs.logits, targets)
            adv_metrics.update(adv_outputs.logits, targets)

    val_metrics_dict = val_metrics.compute()
    adv_metrics_dict = adv_metrics.compute()
    if rank == 0:
        results_dict.update({k: v.item() for k, v in val_metrics_dict.items()})
        results_dict.update({k: v.item() for k, v in adv_metrics_dict.items()})
        run.log(results_dict)

    return val_metrics, adv_metrics
