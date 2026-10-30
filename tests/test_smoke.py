"""Smoke tests — run with: uv run pytest -q"""
import importlib
import torch


def test_quaketriage_importable():
    mod = importlib.import_module("quaketriage")
    assert mod is not None


def test_torch_device_available():
    """MPS (Apple Silicon) or CPU must be available."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    t = torch.tensor([1.0]).to(device)
    assert t.device.type in ("mps", "cpu")


def test_project_crs():
    from quaketriage.data.geo import get_project_crs
    crs = get_project_crs()
    assert crs.startswith("EPSG:")
