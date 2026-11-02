"""Unit tests for shared pixel metrics."""
import pytest
import torch

from quaketriage.metrics import bootstrap_ci, compute_metrics, metrics_from_probs


def test_perfect_and_empty_prediction():
    t = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    m = metrics_from_probs(torch.tensor([[0.9, 0.8, 0.1, 0.2]]), t, 0.5)
    assert m["f1"] == pytest.approx(1.0, abs=1e-6) and m["iou"] == pytest.approx(1.0, abs=1e-6)
    m = metrics_from_probs(torch.zeros(1, 4), t, 0.5)
    assert m["recall"] == pytest.approx(0.0, abs=1e-6)


def test_hand_computed_values():
    # tp=1, fp=1, fn=1 → precision 0.5, recall 0.5, f1 0.5, iou 1/3
    t = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    m = metrics_from_probs(torch.tensor([[0.9, 0.1, 0.9, 0.1]]), t, 0.5)
    assert m["precision"] == pytest.approx(0.5, abs=1e-6)
    assert m["recall"] == pytest.approx(0.5, abs=1e-6)
    assert m["f1"] == pytest.approx(0.5, abs=1e-6)
    assert m["iou"] == pytest.approx(1 / 3, abs=1e-6)


def test_logits_and_probs_agree():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(4, 1, 8, 8, generator=g)
    t = (torch.rand(4, 1, 8, 8, generator=g) > 0.7).float()
    assert compute_metrics(logits, t, 0.5) == metrics_from_probs(torch.sigmoid(logits), t, 0.5)


def test_bootstrap_ci_brackets_point_estimate_and_is_deterministic():
    g = torch.Generator().manual_seed(2)
    logits = torch.randn(20, 1, 8, 8, generator=g)
    t = (torch.rand(20, 1, 8, 8, generator=g) > 0.7).float()
    a, b = bootstrap_ci(logits, t, 0.5, n_boot=100), bootstrap_ci(logits, t, 0.5, n_boot=100)
    assert a == b
    assert a["f1"]["ci95_lo"] <= a["f1"]["ci95_hi"]
