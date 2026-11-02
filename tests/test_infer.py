"""Unit tests for sliding-window inference."""
import numpy as np

from quaketriage.models.infer import infer_area, window_starts


def test_window_starts_cover_edge():
    assert window_starts(100, 64, 48) == [0, 36]
    assert window_starts(64, 64, 48) == [0]
    assert window_starts(30, 64, 48) == [0]
    s = window_starts(1000, 512, 448)
    assert s[0] == 0 and s[-1] == 488


def _imgs(h, w, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(30, 220, (h, w, 3), dtype=np.uint8)
    return a, a.copy()


def test_constant_predictor_gives_constant_map_any_size():
    pre, post = _imgs(150, 200)
    res = infer_area(pre, post, lambda x: (np.full((64, 64), 0.7, np.float32), None),
                     tile=64, overlap=16)
    assert res.prob.shape == (150, 200)
    assert np.allclose(res.prob, 0.7)
    assert res.uncertainty is None


def test_uncertainty_is_averaged():
    pre, post = _imgs(100, 100)
    res = infer_area(pre, post, lambda x: (np.zeros((64, 64), np.float32),
                                           np.full((64, 64), 0.1, np.float32)),
                     tile=64, overlap=16)
    assert np.allclose(res.uncertainty, 0.1)


def test_guard_rejects_blank_tiles_as_nan():
    pre, post = _imgs(128, 128)
    pre[:, :64] = 0
    post[:, :64] = 0  # left half black
    res = infer_area(pre, post, lambda x: (np.ones((64, 64), np.float32), None),
                     tile=64, overlap=0)
    assert len(res.rejected) == 2
    assert np.isnan(res.prob[:, :64]).all() and np.allclose(res.prob[:, 64:], 1.0)


def test_shape_mismatch_raises():
    import pytest
    a, _ = _imgs(64, 64)
    with pytest.raises(ValueError):
        infer_area(a, a[:32], lambda x: (None, None), tile=64, overlap=0)


def test_predict_tile_tta_matches_plain_for_equivariant_model():
    import torch

    from quaketriage.models.infer import predict_tile

    class Pointwise(torch.nn.Module):
        def forward(self, x):
            return x[:, :1]

    x = torch.randn(1, 6, 32, 32)
    p0, s0 = predict_tile(Pointwise(), x, tta=False)
    p1, s1 = predict_tile(Pointwise(), x, tta=True)
    assert s0 is None and p0.shape == (32, 32)
    assert np.allclose(p0, p1, atol=1e-6) and s1.max() < 1e-6
