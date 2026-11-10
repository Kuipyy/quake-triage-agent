"""Road graph helpers: load the OSMnx drive graph and snap depots/buildings to graph nodes."""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox

from quaketriage.config import get
from quaketriage.data.geo import get_project_crs

ROOT = Path(__file__).parents[3]


def load_road_graph(path: str | Path | None = None) -> nx.MultiDiGraph:
    """Load the drive graph, keep its largest strongly connected component, project to the
    project CRS (metres). Dead-end fragments would make some nodes unreachable."""
    path = Path(path) if path else ROOT / get("paths.data_raw") / "demo" / "antakya_roads.graphml"
    g = ox.load_graphml(path)
    keep = max(nx.strongly_connected_components(g), key=len)
    return ox.project_graph(g.subgraph(keep).copy(), to_crs=get_project_crs())


def snap_to_nodes(g: nx.MultiDiGraph, points_lonlat: list[tuple[float, float]]
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Nearest graph node and snap distance (m) for each (lon, lat) point.

    ``g`` must be projected (``load_road_graph``); points are converted to the graph CRS.
    """
    import geopandas as gpd

    pts = gpd.GeoSeries(gpd.points_from_xy(*zip(*points_lonlat, strict=True)), crs=4326)
    pts = pts.to_crs(g.graph["crs"])
    nodes, dist = ox.distance.nearest_nodes(g, pts.x.to_numpy(), pts.y.to_numpy(),
                                            return_dist=True)
    return np.asarray(nodes), np.asarray(dist, float)
