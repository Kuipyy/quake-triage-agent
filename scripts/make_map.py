"""Write reports/map_preview.html from data/processed/buildings.gpkg.

Usage: uv run python scripts/make_map.py
"""
from pathlib import Path

import geopandas as gpd

from quaketriage.config import get
from quaketriage.ui.preview_map import build_preview_map

ROOT = Path(__file__).parents[1]


def main() -> None:
    buildings = gpd.read_file(ROOT / get("paths.data_processed") / "buildings.gpkg")
    out = ROOT / get("paths.reports") / "map_preview.html"
    build_preview_map(buildings).save(str(out))
    size_mb = out.stat().st_size / 1e6
    print(f"{len(buildings)} buildings → {out.relative_to(ROOT)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
