import numpy as np
import torch
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
from torchmetrics import Metric


class FastHausdorffDistance95(Metric):
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

        # Initialize MONAI metric
        self.monai_hd95 = HausdorffDistanceMetric(
            include_background=True,
            percentile=95.0,
            reduction="none",
        )

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
        # Prepare Inputs
        if preds.ndim == 4:
            preds = preds.argmax(dim=1)
        if target.ndim == 4:
            target = target.squeeze(1)

        # Handle ignore_index (Force to class 0/Background)
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            preds = preds * mask
            target = target * mask

        # Convert to One-Hot (Required by MONAI)
        # Shape: (B, C, H, W)
        preds_one_hot = torch.nn.functional.one_hot(
            preds.long(), num_classes=self.num_classes
        ).permute(0, 3, 1, 2)
        target_one_hot = torch.nn.functional.one_hot(
            target.long(), num_classes=self.num_classes
        ).permute(0, 3, 1, 2)

        # Compute HD95 using MONAI (on GPU)
        # Output shape: (B, C)
        batch_hd95 = self.monai_hd95(
            y_pred=preds_one_hot, y=target_one_hot, spacing=self.spacing_mm
        )

        # Handle Edge Cases (Inf/NaN)
        # Calculate max possible distance (image diagonal) for penalties
        max_dist = np.sqrt(preds.shape[1] ** 2 + preds.shape[2] ** 2)

        # Replace Infinity (one empty) with max_dist
        batch_hd95[torch.isinf(batch_hd95)] = max_dist

        # NaNs occur when BOTH are empty. We ignore these in averaging.
        valid_mask = ~torch.isnan(batch_hd95)

        self.total_distance += batch_hd95[valid_mask].sum()
        self.num_samples += valid_mask.sum()

    def compute(self):
        if self.num_samples == 0:
            return torch.tensor(0.0, device=self.device)
        return self.total_distance / self.num_samples
