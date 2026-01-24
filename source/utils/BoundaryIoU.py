# Boundary IoU Metric is not implemented in TorchMetrics, so we implement it here.
import math

import torch
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.classification import MulticlassJaccardIndex


class BoundaryIoU(Metric):
    """
    Implements Boundary IoU with dynamic scaling as defined in:
    "Boundary IoU: Improving Object-Centric Image Segmentation Evaluation"
    Cheng et al., CVPR 2021.
    """

    full_state_update = False

    def __init__(
        self,
        num_classes: int,
        boundary_scale: float = 0.02,  # Default 2% per paper
        min_pixel_dist: int = 1,
        ignore_index: int = 255,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.boundary_scale = boundary_scale
        self.min_pixel_dist = min_pixel_dist
        self.ignore_index = ignore_index

        # Accumulators for Intersection and Union
        self.add_state(
            "intersection", default=torch.zeros(num_classes), dist_reduce_fx="sum"
        )
        self.add_state("union", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.zeros(1), dist_reduce_fx="sum")

        self.mIoU = MulticlassJaccardIndex(
            num_classes=num_classes, ignore_index=ignore_index, average=None
        )

    def _get_circular_kernel(self, radius: int, device):
        """
        Generates a circular kernel for Euclidean distance morphological operations.
        Paper defines boundary as pixels within distance 'd'.
        """
        kernel_size = 2 * radius + 1
        Y, X = torch.meshgrid(
            torch.arange(kernel_size, device=device),
            torch.arange(kernel_size, device=device),
            indexing="ij",
        )
        center = radius
        dist_sq = (X - center) ** 2 + (Y - center) ** 2
        # Create binary mask where distance <= radius
        kernel = (dist_sq <= radius**2).float()
        return kernel

    def _depthwise_erosion(self, mask_oh: torch.Tensor, kernel: torch.Tensor):
        """
        Manual implementation of binary erosion using Depthwise Convolution.

        Args:
            mask_oh: (B, C, H, W) One-hot mask
            kernel: (K, K) Binary kernel
        """
        C = mask_oh.shape[1]
        weight = kernel.expand(C, 1, kernel.shape[0], kernel.shape[1])

        # Padding to maintain spatial resolution
        padding = kernel.shape[0] // 2

        # groups=C ensures each channel is convolved ONLY with its own kernel
        neighbor_counts = F.conv2d(mask_oh, weight, padding=padding, groups=C)

        # A pixel is kept only if ALL neighbors defined by the kernel are 1.
        # Sum of neighbors must equal sum of kernel.
        kernel_area = kernel.sum()

        # Use a small epsilon for float stability
        return (neighbor_counts >= (kernel_area - 1e-4)).float()

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        preds: (N, H, W) integer tensor of class labels
        target: (N, H, W) integer tensor of ground truth labels (with ignore_index)
        """
        preds = preds.detach()
        target = target.detach()

        self.mIoU.update(preds, target)
        # Ensure inputs are on the same device and correct shape
        if preds.ndim == 4:  # If (N, C, H, W) logits/probs
            preds = preds.argmax(dim=1)

        # Calculate Dynamic 'd' (Pixel Distance)
        h, w = preds.shape[-2], preds.shape[-1]
        img_diagonal = math.sqrt(h**2 + w**2)

        # d is the radius of the morphological kernel
        d = int(math.ceil(img_diagonal * self.boundary_scale))
        d = max(d, self.min_pixel_dist)

        # Create kernel based on dynamic size
        kernel = self._get_circular_kernel(radius=d, device=preds.device)

        # Preparation (One-Hot & Masking)
        # Handle Ignore Index
        valid_mask = (target != self.ignore_index).float()

        target_safe = target.clone()
        target_safe[target == self.ignore_index] = 0

        # One-Hot Encoding -> (N, C, H, W)
        preds_oh = (
            torch.nn.functional.one_hot(preds, num_classes=self.num_classes)
            .permute(0, 3, 1, 2)
            .contiguous()
            .float()
        )
        target_oh = (
            torch.nn.functional.one_hot(target_safe, num_classes=self.num_classes)
            .permute(0, 3, 1, 2)
            .contiguous()
            .float()
        )

        # Apply Valid Mask to ignore void regions
        valid_mask_expanded = valid_mask.unsqueeze(1)
        target_oh = target_oh * valid_mask_expanded
        preds_oh = preds_oh * valid_mask_expanded

        # Compute Inner Boundary Regions
        # P_d \cap P
        pred_eroded = self._depthwise_erosion(preds_oh, kernel)
        pred_boundary = preds_oh - pred_eroded

        # G_d \cap G
        target_eroded = self._depthwise_erosion(target_oh, kernel)
        target_boundary = target_oh - target_eroded

        pred_boundary = pred_boundary * valid_mask_expanded
        target_boundary = target_boundary * valid_mask_expanded

        # Compute Intersection and Union
        # Equation (1) in paper: Intersection of boundaries / Union of boundaries
        intersection = (pred_boundary * target_boundary).sum(dim=(0, 2, 3))
        union = (pred_boundary + target_boundary).clamp(0, 1).sum(dim=(0, 2, 3))

        self.intersection += intersection
        self.union += union
        self.total_samples += preds.shape[0]

    def compute(self):
        # Compute mIoU for reference
        mIoU_per_class = self.mIoU.compute()

        # Mean over classes (Macro Average)
        bIoU_per_class = self.intersection / (self.union + 1e-6)

        # Take minimum of mIoU and bIoU per class as per paper
        bIoU_mean = torch.min(mIoU_per_class, bIoU_per_class).mean()
        return bIoU_mean

    def reset(self):
        super().reset()
        self.mIoU.reset()
