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


def test_fbeta_weights_recall():
    from quaketriage.metrics import fbeta

    assert fbeta(0.5, 0.5, 1.0) == pytest.approx(0.5, abs=1e-6)
    # with recall > precision, F2 > F1; with precision > recall, F2 < F1
    assert fbeta(0.4, 0.8, 2.0) > fbeta(0.4, 0.8, 1.0)
    assert fbeta(0.8, 0.4, 2.0) < fbeta(0.8, 0.4, 1.0)


def test_best_threshold_finds_separating_threshold():
    from quaketriage.metrics import best_threshold

    # positives have prob ~0.3, negatives ~0.1 → best threshold must lie between them
    probs = torch.cat([torch.full((50,), 0.3), torch.full((50,), 0.1)])
    t = torch.cat([torch.ones(50), torch.zeros(50)])
    logits = torch.logit(probs)
    thr, score = best_threshold(logits, t, beta=1.0)
    assert 0.1 <= thr < 0.3 and score == pytest.approx(1.0, abs=1e-4)


def test_paired_bootstrap_detects_clear_improvement_and_null():
    from quaketriage.metrics import paired_bootstrap_diff

    g = torch.Generator().manual_seed(3)
    t = (torch.rand(30, 1, 16, 16, generator=g) > 0.8).float()
    good = (t * 4 - 2) + 0.5 * torch.randn(t.shape, generator=g)  # separates classes
    bad = torch.randn(t.shape, generator=g)                        # noise
    better = paired_bootstrap_diff(bad, good, t, 0.5, 0.5, n_boot=200)
    assert better["ci95_lo"] > 0 and better["p_not_better"] == 0.0
    null = paired_bootstrap_diff(good, good, t, 0.5, 0.5, n_boot=200)
    assert null["mean_diff"] == 0.0
