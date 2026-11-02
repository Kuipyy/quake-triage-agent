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
