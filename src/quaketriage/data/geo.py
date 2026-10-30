"""Geo helpers — all spatial work uses a single CRS (configs/default.yaml → geo.crs)."""
from quaketriage.config import get


def get_project_crs() -> str:
    """Return the project CRS string (e.g. 'EPSG:32637')."""
    return get("geo.crs")
