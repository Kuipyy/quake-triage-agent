"""Geo helpers — all spatial work uses a single CRS (configs/default.yaml → geo.crs)."""
import yaml
from pathlib import Path

_CONFIG_PATH = Path(__file__).parents[3] / "configs" / "default.yaml"


def get_project_crs() -> str:
    """Return the project CRS string (e.g. 'EPSG:32637')."""
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg["geo"]["crs"]
