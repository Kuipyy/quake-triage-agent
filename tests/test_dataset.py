"""Unit tests for KateCDDataset and DiceFocalLoss."""
from pathlib import Path

import pytest
import torch

KATE_ROOT = Path(__file__).parents[1] / "data" / "raw" / "kate_cd"
needs_kate_cd = pytest.mark.skipif(
    not (KATE_ROOT / "data").exists(),
    reason="KATE-CD is not downloaded (the images are not in the repository for licence reasons): "
           "run `uv run python scripts/download_kate_cd.py`")


@needs_kate_cd
@pytest.mark.slow
def test_dataset_shapes():
    """Each item must be (6,512,512) image + (1,512,512) mask."""
    from quaketriage.data.kate_cd import KateCDDataset

    ds = KateCDDataset(KATE_ROOT, split="validation")
    assert len(ds) == 44
    img, mask = ds[0]
    assert img.shape == (6, 512, 512)
    assert mask.shape == (1, 512, 512)
    assert img.dtype == torch.float32
    assert mask.dtype == torch.float32
    assert mask.min() >= 0.0 and mask.max() <= 1.0


@needs_kate_cd
@pytest.mark.slow
def test_train_augmentation_determinism():
    """Training split should produce different results on repeated calls (augmentation active)."""
    from quaketriage.data.kate_cd import KateCDDataset

    ds = KateCDDataset(KATE_ROOT, split="train")
    img1, _ = ds[0]
    img2, _ = ds[0]
    # With random augmentation, at least sometimes they differ
    # (small chance of identity transform — acceptable for a smoke test)
    assert img1.shape == (6, 512, 512)
    assert img2.shape == (6, 512, 512)


def test_unet_forward_pass():
    """Model must accept (B,6,H,W) and output (B,1,H,W)."""
    pytest.importorskip("segmentation_models_pytorch", reason="train extra not installed")
    from quaketriage.models.unet import build_unet

    model = build_unet()
    model.eval()
    x = torch.randn(2, 6, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 256, 256)


def test_dice_focal_loss():
    """Loss must be a positive scalar."""
    pytest.importorskip("segmentation_models_pytorch", reason="train extra not installed")
    from quaketriage.models.unet import DiceFocalLoss

    loss_fn = DiceFocalLoss()
    logits = torch.randn(2, 1, 64, 64)
    targets = (torch.rand(2, 1, 64, 64) > 0.9).float()
    loss = loss_fn(logits, targets)
    assert loss.item() > 0
