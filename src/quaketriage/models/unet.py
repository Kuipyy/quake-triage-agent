"""Early-fusion U-Net for binary change detection.

6-channel input (pre+post concatenated), ResNet34 encoder, ImageNet weights.
"""
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_unet(in_channels: int = 6, encoder: str = "resnet34") -> nn.Module:
    """Return an early-fusion U-Net with the given encoder."""
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=1,
        activation=None,  # raw logits; sigmoid applied in loss/metrics
    )
    return model


class DiceFocalLoss(nn.Module):
    """Dice + Focal loss for imbalanced binary segmentation."""

    def __init__(self, focal_alpha: float = 0.25, focal_gamma: float = 2.0) -> None:
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.focal = smp.losses.FocalLoss(
            mode="binary", alpha=focal_alpha, gamma=focal_gamma
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice(logits, targets) + self.focal(logits, targets)
