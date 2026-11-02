"""Pixel-level segmentation metrics shared by train / evaluate / eval_uncertainty."""
from __future__ import annotations

import numpy as np
import torch

METRIC_KEYS = ("precision", "recall", "f1", "iou")


def metrics_from_probs(probs: torch.Tensor, targets: torch.Tensor, threshold: float) -> dict:
    """Precision / recall / F1 / IoU for probabilities thresholded at ``threshold``."""
    preds = (probs > threshold).float()
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "iou": iou.item(),
    }


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> dict:
    """Same as :func:`metrics_from_probs` but takes raw logits."""
    return metrics_from_probs(torch.sigmoid(logits), targets, threshold)


def bootstrap_ci(logits: torch.Tensor, targets: torch.Tensor, threshold: float,
                 n_boot: int = 1000, seed: int = 42) -> dict:
    """95% bootstrap confidence intervals over samples (images)."""
    rng = np.random.default_rng(seed)
    n = logits.shape[0]
    samples = {k: [] for k in METRIC_KEYS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = compute_metrics(logits[idx], targets[idx], threshold)
        for k in METRIC_KEYS:
            samples[k].append(m[k])
    return {
        k: {
            "mean": float(np.mean(v)),
            "ci95_lo": float(np.percentile(v, 2.5)),
            "ci95_hi": float(np.percentile(v, 97.5)),
        }
        for k, v in samples.items()
    }


def fbeta(precision: float, recall: float, beta: float = 1.0) -> float:
    """F-beta score (beta > 1 weights recall more)."""
    b2 = beta**2
    return (1 + b2) * precision * recall / (b2 * precision + recall + 1e-8)


def best_threshold(
    logits: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 1.0,
    grid: tuple[float, ...] = (0.01, 0.02, 0.03, *(round(0.05 * i, 2) for i in range(1, 20))),
) -> tuple[float, float]:
    """Threshold on ``grid`` maximising F-beta; returns (threshold, score); ties keep the lower."""
    best = (grid[0], -1.0)
    for thr in grid:
        m = compute_metrics(logits, targets, thr)
        score = fbeta(m["precision"], m["recall"], beta)
        if score > best[1] + 1e-12:
            best = (thr, score)
    return best


def paired_bootstrap_diff(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    targets: torch.Tensor,
    thr_a: float,
    thr_b: float,
    key: str = "f1",
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Bootstrap of metric(B) - metric(A) over the SAME resampled images (paired).

    ``p_not_better`` is the share of resamples where B is not better than A.
    """
    rng = np.random.default_rng(seed)
    n = targets.shape[0]
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a = compute_metrics(logits_a[idx], targets[idx], thr_a)[key]
        b = compute_metrics(logits_b[idx], targets[idx], thr_b)[key]
        diffs.append(b - a)
    d = np.array(diffs)
    return {
        "metric": key,
        "mean_diff": float(d.mean()),
        "ci95_lo": float(np.percentile(d, 2.5)),
        "ci95_hi": float(np.percentile(d, 97.5)),
        "p_not_better": float((d <= 0).mean()),
    }
