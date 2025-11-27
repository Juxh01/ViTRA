import os

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torchvision import datasets


class BestModelLogger:
    def __init__(self, config, val_transform, device, num_images=10):
        """
        Logger to track the best model based on mIoU and visualize predictions using
        original, unprocessed images.

        Args:
            config (dict): The configuration dictionary containing data paths.
            val_transform (callable): The transformation pipeline used for validation
                                      (must output normalized tensors).
            device (str): The device to run inference on.
            num_images (int): Number of samples to visualize.
        """
        self.best_metrik = -1.0
        self.device = device
        self.num_images = num_images
        self.task = config["general"]["task"]

        # Retrieve data directory
        data_dir = config["general"].get("data_dir", "./data")

        # Initialize the dataset without transforms
        self.raw_dataset = datasets.VOCSegmentation(
            root=data_dir, year="2012", image_set="val", download=False, transform=None
        )

        self.fixed_raw_data = []
        self.fixed_model_inputs = []

        for i in range(min(num_images, len(self.raw_dataset))):
            # Get raw data (PIL Image, PIL Image/Map)
            raw_img, raw_target = self.raw_dataset[i]

            self.fixed_raw_data.append(
                {"raw_img": raw_img, "raw_target": np.array(raw_target)}
            )

            # Apply the validation transform to generate the model input
            img_tensor, _ = val_transform(raw_img, raw_target)

            self.fixed_model_inputs.append(img_tensor)

        # Stack inputs into a batch for efficient inference
        self.fixed_model_inputs = torch.stack(self.fixed_model_inputs).to(device)

    def check_and_log(self, current_metric, model, epoch, run, rank):
        """
        Checks if the current model is the best. If so:
        1. Saves the state dict locally.
        2. Performs inference on the fixed samples.
        3. Upscales predictions to match original image size.
        4. Logs the raw images with overlayed masks to WandB.
        """
        if current_metric > self.best_metric:
            self.best_metric = current_metric

            if self.task == "segmentation":
                # --- Inference ---
                model.eval()
                with torch.no_grad():
                    # Forward pass on normalized, resized tensors
                    outputs = model(self.fixed_model_inputs)
                    logits = outputs.logits

                if rank == 0:
                    print(
                        f"New best metric: {self.best_metric:.4f}. Logging images and saving model..."
                    )

                    wandb_images = []

                    # Iterate through each fixed sample
                    for i in range(len(self.fixed_raw_data)):
                        raw_img = self.fixed_raw_data[i]["raw_img"]
                        raw_target = self.fixed_raw_data[i]["raw_target"]

                        # Get the logits for this specific sample (C, H_model, W_model)
                        sample_logits = logits[i].unsqueeze(0)

                        # Resize logits back to the ORIGINAL raw image size
                        original_size = raw_img.size[
                            ::-1
                        ]  # PIL is (W, H), PyTorch needs (H, W)

                        if sample_logits.shape[-2:] != original_size:
                            sample_logits = F.interpolate(
                                sample_logits,
                                size=original_size,
                                mode="bilinear",
                                align_corners=False,
                            )

                        # Get prediction mask
                        pred_mask = sample_logits.argmax(dim=1).squeeze().cpu().numpy()

                        # PASCAL VOC class labels from https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/segexamples/index.html
                        VOC_CLASS_LABELS = {
                            0: "background",
                            1: "aeroplane",
                            2: "bicycle",
                            3: "bird",
                            4: "boat",
                            5: "bottle",
                            6: "bus",
                            7: "car",
                            8: "cat",
                            9: "chair",
                            10: "cow",
                            11: "diningtable",
                            12: "dog",
                            13: "horse",
                            14: "motorbike",
                            15: "person",
                            16: "potted plant",
                            17: "sheep",
                            18: "sofa",
                            19: "train",
                            20: "tv/monitor",
                        }

                        # Create overlayed WandB image
                        wandb_images.append(
                            wandb.Image(
                                raw_img,
                                masks={
                                    "predictions": {
                                        "mask_data": pred_mask,
                                        "class_labels": VOC_CLASS_LABELS,
                                    },
                                    "ground_truth": {
                                        "mask_data": raw_target,
                                        "class_labels": VOC_CLASS_LABELS,
                                    },
                                },
                                caption=f"Img {i} (Epoch {epoch}) | Original Size: {raw_img.size} | Metric: {self.best_metric:.4f}",
                            )
                        )

                    # Log to WandB
                    run.log(
                        {"val/best_model_predictions": wandb_images, "epoch": epoch},
                        commit=False,
                    )

            # --- Save Model State ---
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                cpu_state = model.state_dict()

            if rank == 0:
                torch.save(cpu_state, "best_model.pt")

    def upload_final_artifact(self, run, rank):
        """Uploads the locally saved best model to WandB at the end of training."""
        if rank == 0 and os.path.exists("best_model.pt"):
            print("Uploading best model to WandB...")
            artifact = wandb.Artifact(
                name=f"best-model-{run.id}",
                type="model",
                description=f"Best model based on val/mIoU or val/acc ({self.best_metric:.4f})",
            )
            artifact.add_file("best_model.pt")
            run.log_artifact(artifact)
