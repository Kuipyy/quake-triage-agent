"""Early-fusion U-Net for binary change detection.

6-channel input (pre+post concatenated), ResNet34 encoder, ImageNet weights.
"""
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_unet(
    in_channels: int = 6, encoder: str = "resnet34", encoder_weights: str | None = "imagenet"
) -> nn.Module:
    """Return an early-fusion U-Net with the given encoder."""
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
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


class SiameseUNet(nn.Module):
    """Siamese U-Net: shared encoder, |f_pre - f_post| skips fed to the decoder.

    Takes the same 6-channel input as the early-fusion model (pre = ch 0-2,
    post = ch 3-5), so it is a drop-in replacement in train/evaluate scripts.
    """

    def __init__(self, encoder: str = "resnet34", encoder_weights: str | None = "imagenet") -> None:
        super().__init__()
        base = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
        )
        self.encoder = base.encoder
        self.decoder = base.decoder
        self.segmentation_head = base.segmentation_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre, post = x[:, :3], x[:, 3:]
        feats_pre = self.encoder(pre)
        feats_post = self.encoder(post)
        diff = [torch.abs(a - b) for a, b in zip(feats_pre, feats_post)]
        return self.segmentation_head(self.decoder(diff))


def build_siamese_unet(
    encoder: str = "resnet34", encoder_weights: str | None = "imagenet"
) -> nn.Module:
    """Return a Siamese U-Net with the given encoder."""
    return SiameseUNet(encoder=encoder, encoder_weights=encoder_weights)
