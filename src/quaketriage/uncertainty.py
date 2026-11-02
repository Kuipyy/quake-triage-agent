"""Uncertainty estimators (MC Dropout, TTA) and calibration metrics (ECE, reliability).

MC Dropout is implemented with forward hooks on the decoder blocks, so no layer is
inserted and existing checkpoints (state_dict keys) stay loadable. Dropout is only
active while ``DecoderDropout.active`` is True; BatchNorm stays in eval mode.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderDropout:
    """Channel dropout (Dropout2d) on the output of every decoder block, toggled by ``active``."""

    def __init__(self, model: nn.Module, p: float = 0.1) -> None:
        if not 0.0 < p < 1.0:
            raise ValueError("p must be in (0, 1)")
        self.p = p
        self.active = False
        self.handles = [b.register_forward_hook(self._hook) for b in model.decoder.blocks]

    def _hook(self, module, inputs, output):
        return F.dropout2d(output, self.p, training=True) if self.active else output

    def remove(self) -> None:
        for h in self.handles:
            h.remove()


@torch.no_grad()
def predict_mc_dropout(
    model: nn.Module, x: torch.Tensor, dropout: DecoderDropout, T: int = 10
) -> torch.Tensor:
    """T stochastic forward passes → probabilities [T, B, 1, H, W]."""
    model.eval()
    dropout.active = True
    try:
        return torch.stack([torch.sigmoid(model(x)) for _ in range(T)])
    finally:
        dropout.active = False


def _dihedral(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    x = torch.rot90(x, k, dims=(-2, -1))
    return x.flip(-1) if flip else x


def _dihedral_inv(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    x = x.flip(-1) if flip else x
    return torch.rot90(x, -k, dims=(-2, -1))


@torch.no_grad()
def predict_tta(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """8 dihedral views (rot90 × flip), each mapped back → probabilities [8, B, 1, H, W]."""
    model.eval()
    return torch.stack([
        _dihedral_inv(torch.sigmoid(model(_dihedral(x, k, f))), k, f)
        for k in range(4) for f in (False, True)
    ])


def expected_calibration_error(
    probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15
) -> float:
    """ECE with equal-width confidence bins over the positive-class probability."""
    return float(reliability_bins(probs, targets, n_bins)["ece"])


def reliability_bins(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> dict:
    """Per-bin mean predicted prob, observed positive rate and count, plus ECE."""
    p = probs.flatten().float()
    t = targets.flatten().float()
    idx = torch.clamp((p * n_bins).long(), max=n_bins - 1)
    count = torch.bincount(idx, minlength=n_bins).float()
    conf = torch.bincount(idx, weights=p, minlength=n_bins)
    pos = torch.bincount(idx, weights=t, minlength=n_bins)
    nz = count > 0
    mean_conf = torch.where(nz, conf / count.clamp(min=1), torch.zeros_like(conf))
    pos_rate = torch.where(nz, pos / count.clamp(min=1), torch.zeros_like(pos))
    ece = ((count / count.sum()) * (mean_conf - pos_rate).abs()).sum().item()
    return {"ece": ece, "mean_conf": mean_conf.tolist(), "pos_rate": pos_rate.tolist(),
            "count": count.tolist()}


def uncertainty_error_ratio(
    std: torch.Tensor, probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> float:
    """Mean uncertainty on wrong pixels ÷ on correct pixels (>1 means informative)."""
    wrong = ((probs > threshold).float() != targets).flatten()
    s = std.flatten()
    if wrong.sum() == 0 or (~wrong).sum() == 0:
        return float("nan")
    return float(s[wrong].mean() / s[~wrong].mean().clamp(min=1e-12))
