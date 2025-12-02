import warnings

import numpy as np
import torch
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
from torchmetrics import Metric


class HausdorffDistance95(Metric):
    """
    Computes HD95 using MONAI's optimized GPU implementation.
    - Handles ignore_index by zeroing it out (background).
    - Returns image diagonal for empty predictions.
    """

    full_state_update = False

    def __init__(
        self, num_classes: int, ignore_index: int = 255, spacing_mm=(1.0, 1.0), **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.spacing_mm = spacing_mm

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
        # Standardize Input Shapes to (B, H, W)
        if preds.ndim == 4:
            preds = preds.argmax(dim=1)
        if target.ndim == 4:
            target = target.squeeze(1)

        # Handle ignore_index (Force to class 0/Background)
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            preds = preds * mask
            target = target * mask

        # Iterate over batch to save memory
        batch_size = preds.shape[0]

        # Pre-calculate Max Dist (Diagonal) for penalties
        # spatial dims are (H, W) -> indices 1 and 2 of preds (B, H, W)
        max_dist = np.sqrt(preds.shape[1] ** 2 + preds.shape[2] ** 2)

        # Suppress MONAI warnings about empty classes (expected in Segmentation)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            for i in range(batch_size):
                # Expand dims to (1, H, W) for one_hot conversion
                curr_pred = preds[i : i + 1]
                curr_target = target[i : i + 1]

                # Shape: (1, C, H, W)
                p_one_hot = torch.nn.functional.one_hot(
                    curr_pred.long(), num_classes=self.num_classes
                ).permute(0, 3, 1, 2)

                t_one_hot = torch.nn.functional.one_hot(
                    curr_target.long(), num_classes=self.num_classes
                ).permute(0, 3, 1, 2)

                # Compute HD95 (Returns shape (1, C))
                hd95_scores = compute_hausdorff_distance(
                    y_pred=p_one_hot,
                    y=t_one_hot,
                    include_background=True,
                    percentile=95.0,
                    spacing=self.spacing_mm,
                )

                # Replace Infinity (one mask empty) with max_dist
                hd95_scores[torch.isinf(hd95_scores)] = max_dist

                # NaNs occur when BOTH masks are empty for a class.
                valid_mask = ~torch.isnan(hd95_scores)

                self.total_distance += hd95_scores[valid_mask].sum()
                self.num_samples += valid_mask.sum()

    def compute(self):
        if self.num_samples == 0:
            return torch.tensor(0.0, device=self.device)
        return self.total_distance / self.num_samples
