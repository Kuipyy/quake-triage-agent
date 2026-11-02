"""Unit tests for the image quality guard."""
import numpy as np
import pytest

from quaketriage.data.quality import QualityInput, check_image_quality

RNG = np.random.default_rng(0)


def _tile(h=128, w=128):
    return RNG.integers(30, 220, size=(h, w, 3), dtype=np.uint8)


def test_identical_tiles_pass():
    t = _tile()
    r = check_image_quality(QualityInput(pre=t, post=t.copy()))
    assert r.ok and r.ssim > 0.99 and r.reasons == []


def test_blank_tile_fails():
    t = _tile()
    post = t.copy()
    post[:, :64] = 0  # 50% black
    r = check_image_quality(QualityInput(pre=t, post=post))
    assert not r.ok and r.blank_ratio == pytest.approx(0.5)
    assert any("blank" in x for x in r.reasons)


def test_cloudy_tile_fails():
    t = _tile()
    post = t.copy()
    post[:64] = 255
    r = check_image_quality(QualityInput(pre=t, post=post))
    assert not r.ok and any("bright" in x for x in r.reasons)


def test_misaligned_tiles_fail_ssim():
    t = _tile()
    r = check_image_quality(QualityInput(pre=t, post=_tile()))  # unrelated content
    assert not r.ok and any("SSIM" in x for x in r.reasons)


def test_change_mask_excludes_damaged_blocks():
    t = _tile()
    post = t.copy()
    post[:64, :64] = _tile(64, 64)  # damaged region
    mask = np.zeros((128, 128), dtype=bool)
    mask[:64, :64] = True
    r = check_image_quality(QualityInput(pre=t, post=post, change_mask=mask))
    assert r.ok and r.ssim > 0.99


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        check_image_quality(QualityInput(pre=_tile(64, 64), post=_tile(128, 128)))
