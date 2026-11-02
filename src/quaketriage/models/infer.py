"""Sliding-window inference over a large pre/post image pair.

Pure numpy/torch (no rasterio) so it is unit-testable; GeoTIFF IO lives in
``scripts/infer_area.py``.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from quaketriage.data.kate_cd import _MEAN, _STD
from quaketriage.data.quality import QualityInput, check_image_quality
from quaketriage.uncertainty import predict_tta


def window_starts(size: int, tile: int, stride: int) -> list[int]:
    """Start offsets covering [0, size) with the last window flush to the edge."""
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


def to_tensor(pre: np.ndarray, post: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 ×2 → [1, 6, H, W] ImageNet-normalised (same as KateCDDataset)."""
    mean = torch.tensor(_MEAN).view(3, 1, 1)
    std = torch.tensor(_STD).view(3, 1, 1)
    chans = []
    for img in (pre, post):
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        chans.append((t - mean) / std)
    return torch.cat(chans, 0).unsqueeze(0)


@torch.no_grad()
def predict_tile(
    model: torch.nn.Module, x: torch.Tensor, tta: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (prob HxW, std HxW or None). With ``tta``: 8-view TTA, std = uncertainty."""
    if tta:
        probs = predict_tta(model, x)
        return probs.mean(0)[0, 0].cpu().numpy(), probs.std(0)[0, 0].cpu().numpy()
    return torch.sigmoid(model(x))[0, 0].cpu().numpy(), None


@dataclass
class AreaResult:
    prob: np.ndarray                     # HxW float32, NaN where no valid tile
    uncertainty: np.ndarray | None       # HxW float32 or None
    n_tiles: int = 0
    rejected: list[tuple[int, int, list[str]]] = field(default_factory=list)  # (y, x, reasons)


def infer_area(
    pre: np.ndarray,
    post: np.ndarray,
    predict: Callable[[torch.Tensor], tuple[np.ndarray, np.ndarray | None]],
    tile: int = 512,
    overlap: int = 64,
    use_guard: bool = True,
) -> AreaResult:
    """Run ``predict`` on overlapping windows, average overlaps, skip tiles failing the guard."""
    if pre.shape != post.shape or pre.ndim != 3 or pre.shape[2] != 3:
        raise ValueError(f"pre/post must be HxWx3 with equal shape, got {pre.shape}, {post.shape}")
    if not 0 <= overlap < tile:
        raise ValueError("overlap must satisfy 0 <= overlap < tile")

    h, w = pre.shape[:2]
    ph, pw = max(h, tile) - h, max(w, tile) - w  # pad small images up to one tile
    if ph or pw:
        pad = ((0, ph), (0, pw), (0, 0))
        pre, post = np.pad(pre, pad), np.pad(post, pad)
    H, W = pre.shape[:2]

    prob_sum = np.zeros((H, W), np.float32)
    std_sum = np.zeros((H, W), np.float32)
    count = np.zeros((H, W), np.float32)
    result = AreaResult(prob=prob_sum, uncertainty=None)
    have_std = False
    stride = tile - overlap

    for y in window_starts(H, tile, stride):
        for x in window_starts(W, tile, stride):
            p, q = pre[y:y + tile, x:x + tile], post[y:y + tile, x:x + tile]
            result.n_tiles += 1
            if use_guard:
                report = check_image_quality(QualityInput(pre=p, post=q))
                if not report.ok:
                    result.rejected.append((y, x, report.reasons))
                    continue
            prob, std = predict(to_tensor(p, q))
            prob_sum[y:y + tile, x:x + tile] += prob
            count[y:y + tile, x:x + tile] += 1
            if std is not None:
                std_sum[y:y + tile, x:x + tile] += std
                have_std = True

    valid = count > 0
    prob_out = np.full((H, W), np.nan, np.float32)
    prob_out[valid] = prob_sum[valid] / count[valid]
    result.prob = prob_out[:h, :w]
    if have_std:
        std_out = np.full((H, W), np.nan, np.float32)
        std_out[valid] = std_sum[valid] / count[valid]
        result.uncertainty = std_out[:h, :w]
    return result
