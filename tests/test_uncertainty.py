"""Unit tests for uncertainty estimators and calibration metrics."""
import pytest
import torch

from quaketriage.models.unet import build_unet
from quaketriage.uncertainty import (
    DecoderDropout,
    expected_calibration_error,
    predict_mc_dropout,
    predict_tta,
    uncertainty_error_ratio,
)


def _model():
    return build_unet(encoder_weights=None).eval()


def test_ece_perfect_and_worst():
    t = torch.cat([torch.ones(500), torch.zeros(500)])
    assert expected_calibration_error(t.clone(), t) == pytest.approx(0.0, abs=1e-6)
    assert expected_calibration_error(1 - t, t) == pytest.approx(1.0, abs=1e-6)
    # constant 0.5 on a balanced set is perfectly calibrated
    assert expected_calibration_error(torch.full((1000,), 0.5), t) == pytest.approx(0.0, abs=1e-6)


def test_dropout_inactive_is_deterministic_and_active_is_stochastic():
    m, x = _model(), torch.randn(1, 6, 64, 64)
    dd = DecoderDropout(m, p=0.3)
    with torch.no_grad():
        a, b = torch.sigmoid(m(x)), torch.sigmoid(m(x))
    assert torch.equal(a, b)
    mc = predict_mc_dropout(m, x, dd, T=4)
    assert mc.shape == (4, 1, 1, 64, 64)
    assert mc.std(0).max() > 0
    assert not dd.active  # restored afterwards


def test_dropout_does_not_change_state_dict_keys():
    m = _model()
    keys = set(m.state_dict())
    DecoderDropout(m, 0.1)
    assert set(m.state_dict()) == keys


def test_tta_inverse_maps_back():
    """A rotation/flip-equivariant 'model' must give zero TTA std."""
    class Equivariant(torch.nn.Module):
        def forward(self, x):
            return x[:, :1] * 3.0  # pointwise → equivariant to rot/flip

    x = torch.randn(1, 6, 32, 32)
    p = predict_tta(Equivariant(), x)
    assert p.shape[0] == 8 and p.std(0).max() < 1e-6


def test_uncertainty_error_ratio():
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
    probs = torch.tensor([0.9, 0.1, 0.1, 0.1])     # pixel 1 is wrong
    std = torch.tensor([0.01, 0.2, 0.01, 0.01])
    assert uncertainty_error_ratio(std, probs, targets) == pytest.approx(20.0)
