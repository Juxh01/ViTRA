import torch
import torch.nn.functional as F
from torchmetrics.segmentation import HausdorffDistance


class MaskedHausdorffDistance(HausdorffDistance):
    def __init__(self, ignore_index: int, **kwargs):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets, applying the ignore_index mask.
        """
        # Create segmentation predictions from logits
        if preds.ndim == 4 and preds.shape[1] == self.num_classes:
            preds = preds.argmax(dim=1)

        # Create a mask for valid pixels
        valid_mask = target != self.ignore_index

        # --- Process Target ---
        # Replace ignore_index with a valid class (0) temporarily for one_hot encoding
        target_clean = target.clone()
        target_clean[~valid_mask] = 0

        # Convert to one-hot (B, C, H, W)
        target_one_hot = F.one_hot(
            target_clean.long(), num_classes=self.num_classes
        ).permute(0, 3, 1, 2)

        # Zero out the one-hot vector at ignored positions
        # This removes these pixels from the 'set' of any class
        target_one_hot = target_one_hot * valid_mask.unsqueeze(1)

        # --- Process Preds ---
        # Convert preds to one-hot
        preds_one_hot = F.one_hot(preds.long(), num_classes=self.num_classes).permute(
            0, 3, 1, 2
        )

        # Also mask predictions in the ignore region.
        # This prevents the metric from penalizing the model for predictions in void regions.
        preds_one_hot = preds_one_hot * valid_mask.unsqueeze(1)

        # Pass the processed one-hot tensors to the parent update
        super().update(preds_one_hot, target_one_hot)
