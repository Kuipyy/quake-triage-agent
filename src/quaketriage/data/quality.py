"""Image quality guard for pre/post tile pairs.

Checks (thresholds from ``configs/default.yaml`` → ``thresholds.guard_*``):
  * blank ratio   — share of black / no-data pixels
  * bright ratio  — share of saturated pixels (cloud / glare proxy)
  * alignment     — block-wise SSIM between pre and post on areas expected
                    to be unchanged (outside ``change_mask`` when given,
                    otherwise the median over blocks, which is robust to
                    local damage).
Pure numpy: no extra dependency.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from quaketriage.config import get

_C1 = (0.01 * 255) ** 2
_C2 = (0.03 * 255) ** 2


class QualityInput(BaseModel):
    """Tool input: two HxWx3 uint8 RGB tiles (+ optional HxW bool change mask)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pre: np.ndarray
    post: np.ndarray
    change_mask: np.ndarray | None = None
    blank_threshold: int = Field(default=5, ge=0, le=255)
    bright_threshold: int = Field(default=250, ge=0, le=255)
    block: int = Field(default=32, ge=4)


class QualityReport(BaseModel):
    blank_ratio: float
    bright_ratio: float
    ssim: float | None
    ok: bool
    reasons: list[str]


def _gray(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float64).mean(axis=-1)


def blank_ratio(img: np.ndarray, threshold: int = 5) -> float:
    """Fraction of pixels whose every channel is <= threshold."""
    return float((img <= threshold).all(axis=-1).mean())


def bright_ratio(img: np.ndarray, threshold: int = 250) -> float:
    """Fraction of pixels whose every channel is >= threshold."""
    return float((img >= threshold).all(axis=-1).mean())


def block_ssim(a: np.ndarray, b: np.ndarray, block: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """SSIM per non-overlapping block of two grayscale images.

    Returns (scores, valid_mask_flat_index) as arrays shaped (ny, nx).
    """
    h, w = a.shape
    ny, nx = h // block, w // block
    scores = np.full((ny, nx), np.nan)
    for i in range(ny):
        for j in range(nx):
            pa = a[i * block:(i + 1) * block, j * block:(j + 1) * block]
            pb = b[i * block:(i + 1) * block, j * block:(j + 1) * block]
            ma, mb = pa.mean(), pb.mean()
            va, vb = pa.var(), pb.var()
            cov = ((pa - ma) * (pb - mb)).mean()
            scores[i, j] = ((2 * ma * mb + _C1) * (2 * cov + _C2)) / (
                (ma**2 + mb**2 + _C1) * (va + vb + _C2)
            )
    return scores, np.arange(ny * nx).reshape(ny, nx)


def alignment_ssim(
    pre: np.ndarray,
    post: np.ndarray,
    change_mask: np.ndarray | None = None,
    block: int = 32,
) -> float | None:
    """Alignment score: mean block SSIM on unchanged areas (None if no usable block)."""
    scores, _ = block_ssim(_gray(pre), _gray(post), block)
    if scores.size == 0:
        return None
    if change_mask is None:
        return float(np.nanmedian(scores))
    ny, nx = scores.shape
    keep = np.zeros((ny, nx), dtype=bool)
    for i in range(ny):
        for j in range(nx):
            patch = change_mask[i * block:(i + 1) * block, j * block:(j + 1) * block]
            keep[i, j] = not patch.any()
    if not keep.any():
        return None
    return float(np.nanmean(scores[keep]))


def check_image_quality(inp: QualityInput) -> QualityReport:
    """Agent tool: decide whether a pre/post pair is trustworthy enough for inference."""
    if inp.pre.shape != inp.post.shape or inp.pre.ndim != 3:
        raise ValueError(
            f"pre/post must share HxWxC shape, got {inp.pre.shape} vs {inp.post.shape}"
        )

    blank = max(
        blank_ratio(img, inp.blank_threshold) for img in (inp.pre, inp.post)
    )
    bright = max(
        bright_ratio(img, inp.bright_threshold) for img in (inp.pre, inp.post)
    )
    ssim = alignment_ssim(inp.pre, inp.post, inp.change_mask, inp.block)

    reasons: list[str] = []
    if blank > get("thresholds.guard_blank_ratio"):
        reasons.append(f"blank/no-data ratio {blank:.2f} > {get('thresholds.guard_blank_ratio')}")
    if bright > get("thresholds.guard_bright_ratio"):
        reasons.append(f"bright/cloud ratio {bright:.2f} > {get('thresholds.guard_bright_ratio')}")
    if ssim is not None and ssim < get("thresholds.guard_ssim_min"):
        reasons.append(f"alignment SSIM {ssim:.2f} < {get('thresholds.guard_ssim_min')}")

    return QualityReport(
        blank_ratio=round(blank, 4),
        bright_ratio=round(bright, 4),
        ssim=None if ssim is None else round(ssim, 4),
        ok=not reasons,
        reasons=reasons,
    )
