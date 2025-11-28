import numpy as np
import torch
from surface_distance import metrics as sd_metrics
from torchmetrics import Metric


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

        batch_sum = 0.0
        batch_count = 0

        # Loop over the batch (DeepMind Lib processes only single images)
        for i in range(len(preds_np)):
            p_img = preds_np[i]
            t_img = target_np[i]

            # Create masked versions for ignore_index
            if self.ignore_index is not None:
                valid_mask = t_img != self.ignore_index
                # Create copies to avoid modifying original data
                p_curr = np.zeros_like(p_img)
                t_curr = np.zeros_like(t_img)

                p_curr[valid_mask] = p_img[valid_mask]
                t_curr[valid_mask] = t_img[valid_mask]
            else:
                p_curr, t_curr = p_img, t_img

            # Calculate HD95 per class
            for c in range(self.num_classes):
                # Create boolean masks for the current class
                p_bool = p_curr == c
                t_bool = t_curr == c

                # Check for empty sets (prevents library crash)
                has_pred = np.any(p_bool)
                has_target = np.any(t_bool)

                if not has_pred and not has_target:
                    # Both empty -> Perfect match (distance 0)
                    batch_sum += 0.0
                    batch_count += 1
                    continue

                if not has_pred or not has_target:
                    # One is empty -> Max distance
                    max_dist = np.sqrt(p_bool.shape[0] ** 2 + p_bool.shape[1] ** 2)
                    batch_sum += max_dist
                    batch_count += 1
                    continue

                # Actual calculation via DeepMind Library
                try:
                    surface_dists = sd_metrics.compute_surface_distances(
                        t_bool, p_bool, spacing_mm=self.spacing_mm
                    )
                    hd95 = sd_metrics.compute_robust_hausdorff(
                        surface_dists, percent=95.0
                    )

                    # If 'inf' is returned
                    if not np.isfinite(hd95):
                        max_dist = np.sqrt(p_bool.shape[0] ** 2 + p_bool.shape[1] ** 2)
                        batch_sum += max_dist
                    else:
                        batch_sum += hd95

                    batch_count += 1
                except Exception as e:
                    print("Error computing HD95 for class", c, ":", e)
                    continue

        # Add to global state (tensor on correct device)
        self.total_distance += torch.tensor(batch_sum, device=self.device)
        self.num_samples += torch.tensor(batch_count, device=self.device)

    def compute(self):
        if self.num_samples == 0:
            return torch.tensor(0.0, device=self.device)
        return self.total_distance / self.num_samples
