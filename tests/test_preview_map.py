"""Smoke test for the folium preview map."""
import re

import geopandas as gpd
from shapely.geometry import box

from quaketriage.ui.preview_map import build_preview_map


def test_preview_map_contains_buildings_and_disclaimer():
    geoms = [box(500000, 4000000, 500010, 4000010), box(500020, 4000000, 500030, 4000010)]
    g = gpd.GeoDataFrame({
        "building": ["yes", "yes"], "damage_prob": [0.9, 0.05], "uncertainty": [0.1, 0.3],
        "estimated_persons": [12.0, 3.0], "destroyed_truth": [True, False],
        "needs_verification": [False, True], "verification_reason": ["", "uncertainty > 0.2"],
    }, geometry=geoms, crs="EPSG:32637")
    html = build_preview_map(g).get_root().render()
    assert "SIMULATED detector" in html and "verification queue" in html
    # the 0.9-probability building gets the darkest fill (not just the legend swatch)
    assert re.search(r"fillColor[^#]{0,6}#b30000", html)
    assert len(re.findall(r'"type":\s*"Feature"', html)) >= 2
