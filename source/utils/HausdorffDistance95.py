# Several implementations of Hausdorff Distance 95 exist, but they usually do not handle ignore_index properly or support batch processing or provide percentile calculation.
# This implementation uses the DeepMind surface-distance library to compute the HD95 while properly handling ignore_index and batch processing.

import numpy as np
import torch
from joblib import Parallel, delayed
from surface_distance import metrics as sd_metrics
from torchmetrics import Metric


def compute_sample_hd95(p_img, t_img, num_classes, ignore_index, spacing_mm):
    """
    Helper function to compute HD95 for a single sample.
    Designed to be picklable for joblib parallelization.
    """
    # Handle ignore_index
    if ignore_index is not None:
        valid_mask = t_img != ignore_index
        p_curr = np.zeros_like(p_img)
        t_curr = np.zeros_like(t_img)
        p_curr[valid_mask] = p_img[valid_mask]
        t_curr[valid_mask] = t_img[valid_mask]
    else:
        p_curr, t_curr = p_img, t_img

    sample_sum = 0.0
    sample_count = 0

    for c in range(num_classes):
        p_bool = p_curr == c
        t_bool = t_curr == c

        has_pred = np.any(p_bool)
        has_target = np.any(t_bool)

        if not has_pred and not has_target:
            # Both empty -> Perfect match
            continue

        if not has_pred or not has_target:
            # One is empty -> Max distance
            max_dist = np.sqrt(p_bool.shape[0] ** 2 + p_bool.shape[1] ** 2)
            sample_sum += max_dist
            sample_count += 1
            continue

        try:
            # DeepMind library call
            surface_dists = sd_metrics.compute_surface_distances(
                t_bool, p_bool, spacing_mm=spacing_mm
            )
            hd95 = sd_metrics.compute_robust_hausdorff(surface_dists, percent=95.0)

            # If 'inf' is returned
            if not np.isfinite(hd95):
                max_dist = np.sqrt(p_bool.shape[0] ** 2 + p_bool.shape[1] ** 2)
                sample_sum += max_dist
            else:
                sample_sum += hd95
            sample_count += 1
        except Exception:
            continue

    return sample_sum, sample_count


class HausdorffDistance95(Metric):
    """
    Computes the 95th percentile Hausdorff Distance (HD95) between predicted and target segmentation masks,
    ignoring a specified index (e.g., 255 for void class). Utilizes the DeepMind surface-distance library.
    """

    # Prevents TorchMetrics from trying to save the inputs
    full_state_update = False

    def __init__(
        self, num_classes: int, ignore_index: int = 255, spacing_mm=(1.0, 1.0), **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.spacing_mm = spacing_mm

        # States to accumulate total distance and count
        self.add_state(
            "total_distance",
            default=torch.tensor(0.0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "num_samples",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Move tensors to CPU for processing
        if preds.ndim == 4:
            preds = preds.argmax(dim=1)
        if target.ndim == 4:
            target = target.squeeze(1)

        preds_np = preds.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        # Parallelize the batch loop using joblib
        # n_jobs=-1 uses all available cores. Adjust if CPU oversubscription occurs.
        results = Parallel(n_jobs=8)(
            delayed(compute_sample_hd95)(
                preds_np[i],
                target_np[i],
                self.num_classes,
                self.ignore_index,
                self.spacing_mm,
            )
            for i in range(len(preds_np))
        )

        # Aggregate results from parallel execution
        batch_sum = sum(res[0] for res in results)
        batch_count = sum(res[1] for res in results)

        # Add to global state (tensor on correct device)
        self.total_distance += torch.tensor(batch_sum, device=self.device)
        self.num_samples += torch.tensor(batch_count, device=self.device)

    def compute(self):
        if self.num_samples == 0:
            return torch.tensor(0.0, device=self.device)
        return self.total_distance / self.num_samples
