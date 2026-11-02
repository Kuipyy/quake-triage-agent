"""Unit tests for SiameseUNet."""
import torch

from quaketriage.models.unet import build_siamese_unet


def test_siamese_output_shape():
    model = build_siamese_unet(encoder_weights=None).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 6, 64, 64))
    assert out.shape == (2, 1, 64, 64)


def test_identical_pre_post_gives_constant_diff_features():
    """pre == post → all skip diffs are zero → output must not depend on image content."""
    model = build_siamese_unet(encoder_weights=None).eval()
    a, b = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out_a = model(torch.cat([a, a], 1))
        out_b = model(torch.cat([b, b], 1))
    assert torch.allclose(out_a, out_b, atol=1e-5)
