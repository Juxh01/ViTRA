import numpy as np
import torch
from torchvision import datasets, tv_tensors
from torchvision.transforms import functional as F


class SBDatasetMultiLabel(datasets.SBDataset):
    def __init__(self, *args, num_classes=20, **kwargs):
        self._transforms = kwargs.pop("transforms", None)
        super().__init__(*args, mode="segmentation", transforms=None, **kwargs)
        self.num_classes = num_classes

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        if self._transforms is not None:
            # Wrap inputs: Image -> tv_tensors.Image, Target -> tv_tensors.Mask
            img = tv_tensors.Image(F.pil_to_tensor(img))
            target = tv_tensors.Mask(F.pil_to_tensor(target))

            # Apply transforms
            img, target = self._transforms(img, target)

        # Convert target to long tensor for bincount
        if isinstance(target, torch.Tensor):
            target_t = target.as_subclass(torch.Tensor).long()
        else:
            target_t = torch.as_tensor(np.array(target), dtype=torch.long)

        # Convert target to tensor
        target_t = torch.as_tensor(np.array(target), dtype=torch.long)

        # Count frequency of every class in the image
        counts = torch.bincount(target_t.flatten().long(), minlength=256)

        # Extract only the classes of interest (1 to 20).
        #    excludes background (0) and ignore (255)
        #    Shape becomes (num_classes,)
        valid_counts = counts[1 : self.num_classes + 1]

        # Create boolean presence vector and cast to float.
        label_vector = (valid_counts > 0).float()

        return img, label_vector
