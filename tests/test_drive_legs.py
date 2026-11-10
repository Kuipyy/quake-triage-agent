"""Vehicle legs must never use a closed edge; only walking legs may enter closed streets."""
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString

from quaketriage.config import get
from quaketriage.data.geo import get_project_crs
from quaketriage.planning.distances import drive_edges
from quaketriage.planning.roads import ROOT

X0, Y0 = 250000.0, 4000000.0


def _square(block_direct: bool):
    """0 -> 1 directly (500 m) or around via 2, 3 (1500 m); direct edge can be closed."""
    g = nx.MultiDiGraph(crs=get_project_crs())
    pos = {0: (0, 0), 1: (500, 0), 2: (0, 500), 3: (500, 500)}
    for n, (x, y) in pos.items():
        g.add_node(n, x=X0 + x, y=Y0 + y)
    for a, b, closed in ((0, 1, block_direct), (0, 2, False), (2, 3, False), (3, 1, False)):
        geom = LineString([(X0 + pos[a][0], Y0 + pos[a][1]), (X0 + pos[b][0], Y0 + pos[b][1])])
        g.add_edge(a, b, geometry=geom, blocked=closed,
                   travel_time_adj=np.inf if closed else geom.length / 10)
    return g


def test_open_direct_edge_is_used():
    assert [(a, b) for a, b, *_ in drive_edges(_square(False), 0, 1)] == [(0, 1)]


def test_vehicle_detours_around_closed_edge_even_if_longer():
    legs = drive_edges(_square(True), 0, 1)
    assert [(a, b) for a, b, *_ in legs] == [(0, 2), (2, 3), (3, 1)]
    assert not any(d["blocked"] for *_, d in legs)


def test_no_open_path_raises_instead_of_crossing_closure():
    g = _square(True)
    g.remove_edge(2, 3)
    with pytest.raises(nx.NetworkXNoPath):
        drive_edges(g, 0, 1)


@pytest.mark.skipif(not (Path(ROOT) / "reports/plan_demo.json").exists()
                    or not (Path(ROOT) / "data/raw/demo/antakya_roads.graphml").exists(),
                    reason="demo plan / road graph not available")
def test_real_plan_has_no_closed_edge_in_any_driving_leg():
    import geopandas as gpd

    from quaketriage.planning.debris import DebrisInput, apply_debris_risk
    from quaketriage.planning.roads import load_road_graph, snap_to_nodes
    from quaketriage.planning.topsis import TopsisInput, prioritise

    plan = json.loads((Path(ROOT) / "reports/plan_demo.json").read_text())
    depot = tuple(plan["depot_lonlat"])
    b = prioritise(gpd.read_file(Path(ROOT) / get("paths.data_processed") / "buildings.gpkg"),
                   TopsisInput(depots=[depot]))
    g = apply_debris_risk(load_road_graph(), b, DebrisInput())
    depot_node = int(snap_to_nodes(g, [depot])[0][0])
    n_legs = n_closed_in_graph = 0
    for route in plan["routes"]:
        nodes = [depot_node, *(s["node"] for s in route["stops"]), depot_node]
        for u, v in zip(nodes, nodes[1:], strict=False):
            for a, b_, key, _ in drive_edges(g, u, v):
                # re-read the flag from the graph, independent of the path search
                assert not g[a][b_][key]["blocked"], f"team {route['team']}: closed edge {a}->{b_}"
                n_legs += 1
        n_closed_in_graph = sum(d["blocked"] for *_, d in g.edges(data=True))
    assert n_legs > 0 and n_closed_in_graph > 0  # the check is not vacuous
