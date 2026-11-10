from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from quaketriage.config import get
from quaketriage.data.geo import get_project_crs
from quaketriage.planning.roads import ROOT, load_road_graph, snap_to_nodes
from quaketriage.planning.topsis import TopsisInput


def _toy_graph():
    g = nx.MultiDiGraph(crs=get_project_crs())
    # two nodes ~1 km apart in UTM metres around Antakya (E 250000-ish, N 4000000-ish)
    g.add_node(1, x=250000.0, y=4000000.0)
    g.add_node(2, x=251000.0, y=4000000.0)
    g.add_edge(1, 2, length=1000.0)
    g.add_edge(2, 1, length=1000.0)
    return g


def test_snap_converts_lonlat_to_graph_crs():
    import geopandas as gpd
    g = _toy_graph()
    p = gpd.GeoSeries(gpd.points_from_xy([250100.0], [4000050.0]), crs=g.graph["crs"]).to_crs(4326)
    nodes, dist = snap_to_nodes(g, [(p.x[0], p.y[0])])
    assert nodes[0] == 1 and np.isclose(dist[0], np.hypot(100, 50), atol=1.0)


def test_default_depots_come_from_config():
    assert TopsisInput().depots == [tuple(d) for d in get("routing.depots")]
    lon, lat = get("routing.depots")[0]
    x0, y0, x1, y1 = get("geo.demo_bbox")
    assert x0 <= lon <= x1 and y0 <= lat <= y1  # lon/lat order, inside the demo area


@pytest.mark.skipif(not (Path(ROOT) / "data/raw/demo/antakya_roads.graphml").exists(),
                    reason="demo road graph not downloaded")
def test_config_depot_snaps_to_real_graph_node():
    g = load_road_graph()
    nodes, dist = snap_to_nodes(g, [tuple(d) for d in get("routing.depots")])
    assert nodes[0] in g and dist[0] < 150  # depot is on/near a street, not in a void
    assert nx.is_strongly_connected(g)
