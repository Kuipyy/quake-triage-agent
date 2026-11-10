"""Leaflet map of a rescue plan: depot, one colour per team, numbered stops, closed roads."""
from __future__ import annotations

import folium
import geopandas as gpd
import networkx as nx
import pyproj

from quaketriage.data.geo import get_project_crs

TEAM_COLOURS = ("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02")


def _to_lonlat():
    t = pyproj.Transformer.from_crs(get_project_crs(), 4326, always_xy=True)
    return lambda x, y: t.transform(x, y)[::-1]  # folium wants (lat, lon)


def _leg(g: nx.MultiDiGraph, u, v, to_ll) -> list[tuple[float, float]]:
    """Road geometry of the fastest open path u -> v as (lat, lon) points."""
    path = nx.shortest_path(g, u, v, weight="travel_time_adj")
    pts: list[tuple[float, float]] = []
    for a, b in zip(path, path[1:], strict=False):
        data = min(g[a][b].values(), key=lambda d: d.get("travel_time_adj", float("inf")))
        geom = data.get("geometry")
        xy = list(geom.coords) if geom is not None else [(g.nodes[a]["x"], g.nodes[a]["y"]),
                                                          (g.nodes[b]["x"], g.nodes[b]["y"])]
        pts += [to_ll(x, y) for x, y in xy]
    return pts or [to_ll(g.nodes[u]["x"], g.nodes[u]["y"])]


def build_route_map(g: nx.MultiDiGraph, plan: dict, buildings: gpd.GeoDataFrame,
                    depot_node, depot_lonlat: tuple[float, float]) -> folium.Map:
    """``g`` is the debris-adjusted projected graph, ``plan`` the dict from ``plan_routes``."""
    to_ll = _to_lonlat()
    m = folium.Map(location=[depot_lonlat[1], depot_lonlat[0]], zoom_start=14,
                   tiles="OpenStreetMap", control_scale=True)
    closed = folium.FeatureGroup("closed roads (debris rule)", show=True)
    for u, v, d in g.edges(data=True):
        if d.get("blocked") and u < v:  # draw each two-way pair once
            geom = d.get("geometry")
            xy = list(geom.coords) if geom is not None else [(g.nodes[u]["x"], g.nodes[u]["y"]),
                                                              (g.nodes[v]["x"], g.nodes[v]["y"])]
            folium.PolyLine([to_ll(x, y) for x, y in xy], color="#b30000", weight=3,
                            opacity=0.8).add_to(closed)
    closed.add_to(m)

    cen = buildings.to_crs(get_project_crs()).geometry.centroid
    for route in plan["routes"]:
        colour = TEAM_COLOURS[route["team"] % len(TEAM_COLOURS)]
        grp = folium.FeatureGroup(f"team {route['team'] + 1} ({len(route['stops'])} stops)")
        prev = depot_node
        for order, stop in enumerate(route["stops"], 1):
            folium.PolyLine(_leg(g, prev, stop["node"], to_ll), color=colour, weight=4,
                            opacity=0.85).add_to(grp)
            prev = stop["node"]
            bx = cen.loc[stop["row"]]
            blat, blon = to_ll(bx.x, bx.y)
            if stop["walk_m"] > 0:  # walking leg: dashed line from the parking node
                nx_, ny_ = g.nodes[stop["node"]]["x"], g.nodes[stop["node"]]["y"]
                folium.PolyLine([to_ll(nx_, ny_), (blat, blon)], color=colour, weight=2,
                                dash_array="5 6").add_to(grp)
            tip = (f"team {route['team'] + 1}, stop {order}: priority {stop['priority']:.2f}, "
                   f"~{stop['estimated_persons']:.0f} persons, arrival "
                   f"{stop['arrival_s'] / 3600:.1f} h, walk {stop['walk_m']:.0f} m")
            folium.CircleMarker((blat, blon), radius=6, color="#222", weight=1, fill=True,
                                fill_color=colour, fill_opacity=0.95, tooltip=tip).add_to(grp)
        if route["stops"]:
            folium.PolyLine(_leg(g, prev, depot_node, to_ll), color=colour, weight=4,
                            opacity=0.85).add_to(grp)
        grp.add_to(m)
    folium.Marker((depot_lonlat[1], depot_lonlat[0]), tooltip="depot (scenario assumption: OSM "
                  "'AFAD' building, not a real operations centre)",
                  icon=folium.Icon(color="black", icon="home")).add_to(m)
    folium.LayerControl().add_to(m)
    legend = ("<div style='position:fixed;bottom:30px;right:10px;max-width:340px;z-index:9999;"
              "background:white;padding:8px 10px;border:1px solid #999;font:12px sans-serif'>"
              "<b>SIMULATED detector &mdash; demo plan, not an operational plan</b><br>"
              f"{plan['n_visited']} buildings, priority {plan['priority_collected']:.1f}, "
              f"~{plan['persons_reached']:.0f} persons<br>"
              "<span style='color:#b30000'>&#9644;</span> closed road &nbsp; dashed = on foot<br>"
              "Data: &copy; OpenStreetMap contributors, HOTOSM (ODbL); WorldPop (CC BY 4.0)</div>")
    m.get_root().html.add_child(folium.Element(legend))
    return m
