"""Static Leaflet map preview of the building table (folium, self-contained HTML)."""
from __future__ import annotations

import folium
import geopandas as gpd
import shapely

from quaketriage.data.geo import get_project_crs

_POPUP_FIELDS = ["building", "name", "floors", "estimated_persons", "damage_prob", "uncertainty",
                 "destroyed_truth", "needs_verification", "verification_reason"]


_BREAKS = ((0.6, "#b30000"), (0.4, "#e34a33"), (0.2, "#fc8d59"), (0.1, "#fdcc8a"))


def _colour(p: float) -> str:
    return next((colour for limit, colour in _BREAKS if p > limit), "#d9d9d9")


def build_preview_map(buildings: gpd.GeoDataFrame, simplify_m: float = 0.5) -> folium.Map:
    """Colour buildings by damage probability; outline the verification queue and the truth."""
    proj = buildings.to_crs(get_project_crs()).copy()
    proj["geometry"] = proj.geometry.simplify(simplify_m)
    ll = proj.to_crs(4326)
    ll["geometry"] = shapely.set_precision(ll.geometry.values, 1e-6)
    cols = [c for c in _POPUP_FIELDS if c in ll.columns]
    for c in ("damage_prob", "uncertainty", "estimated_persons"):
        if c in ll:
            ll[c] = ll[c].round(2)
    minx, miny, maxx, maxy = ll.total_bounds
    m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], zoom_start=15,
                   tiles="OpenStreetMap", control_scale=True)

    def style(feat):
        pr = feat["properties"]
        queue = pr.get("needs_verification")
        return {"fillColor": _colour(pr.get("damage_prob") or 0.0), "fillOpacity": 0.8,
                "color": "#1f78b4" if queue else "#555555", "weight": 2.0 if queue else 0.3}

    tip_fields = [c for c in ("damage_prob", "estimated_persons") if c in cols]
    folium.GeoJson(ll[cols + ["geometry"]], name="buildings (simulated detector)",
                   style_function=style,
                   tooltip=folium.GeoJsonTooltip(fields=tip_fields)).add_to(m)
    if "destroyed_truth" in ll:
        outline = {"fillOpacity": 0, "color": "#000000", "weight": 1.2}
        folium.GeoJson(ll[ll.destroyed_truth][["geometry"]], name="HOTOSM destroyed (truth)",
                       style_function=lambda _: outline, show=False).add_to(m)
    folium.LayerControl().add_to(m)
    legend = ("<div style='position:fixed;bottom:25px;left:25px;z-index:9999;background:white;"
              "padding:8px 10px;border:1px solid #999;font:12px sans-serif'>"
              "<b>SIMULATED detector &mdash; demo only</b><br>damage prob.: "
              "<span style='color:#b30000'>&#9632;</span> &gt;0.6 "
              "<span style='color:#e34a33'>&#9632;</span> &gt;0.4 "
              "<span style='color:#fc8d59'>&#9632;</span> &gt;0.2 "
              "<span style='color:#fdcc8a'>&#9632;</span> &gt;0.1<br>"
              "<span style='color:#1f78b4'>&#9632;</span> blue outline = verification queue<br>"
              "Data: &copy; OpenStreetMap contributors, HOTOSM (ODbL); WorldPop (CC BY 4.0)</div>")
    m.get_root().html.add_child(folium.Element(legend))
    return m
