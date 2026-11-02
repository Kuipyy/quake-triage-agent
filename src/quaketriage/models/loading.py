"""Checkpoint loading shared by inference / comparison scripts."""
from __future__ import annotations

from pathlib import Path

import torch

from quaketriage.config import get
from quaketriage.models.unet import build_siamese_unet, build_unet

MODEL_TAGS = {"early_fusion": "baseline", "siamese": "siamese"}


def load_model(
    kind: str, checkpoint: Path, encoder: str, device: torch.device
) -> tuple[torch.nn.Module, float]:
    """Build ``kind`` without downloading pretrained weights, load ``checkpoint``, eval mode.

    Returns (model, threshold stored in the checkpoint).
    """
    if kind not in MODEL_TAGS:
        raise ValueError(f"unknown model kind {kind!r}; expected one of {sorted(MODEL_TAGS)}")
    builder = build_unet if kind == "early_fusion" else build_siamese_unet
    model = builder(encoder=encoder, encoder_weights=None).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt.get("threshold", get("thresholds.damage_prob"))
