import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point

from quaketriage.data.geo import get_project_crs
from quaketriage.planning.debris import DebrisInput, apply_debris_risk, debris_sources

CRS = get_project_crs()
X0, Y0 = 250000.0, 4000000.0


def _graph():
    """A -- B -- C along x (1 km each); B is 2 km above on a bypass path A -- D -- C."""
    g = nx.MultiDiGraph(crs=CRS)
    pos = {"A": (0, 0), "B": (1000, 0), "C": (2000, 0), "D": (1000, 1500)}
    for n, (x, y) in pos.items():
        g.add_node(n, x=X0 + x, y=Y0 + y)
    for u, v in (("A", "B"), ("B", "C"), ("A", "D"), ("D", "C")):
        for a, b in ((u, v), (v, u)):
            geom = LineString([(g.nodes[a]["x"], g.nodes[a]["y"]),
                               (g.nodes[b]["x"], g.nodes[b]["y"])])
            g.add_edge(a, b, geometry=geom, length=geom.length, travel_time=geom.length / 10)
    return g


def _bld(n, x, y, prob=0.9, floors=5):
    return gpd.GeoDataFrame({"damage_prob": [prob] * n, "floors": [floors] * n},
                            geometry=[Point(X0 + x, Y0 + y).buffer(8) for _ in range(n)], crs=CRS)


def test_sources_need_damage_and_height():
    b = gpd.GeoDataFrame({"damage_prob": [0.9, 0.9, 0.2], "floors": [5, 1, 6]},
                         geometry=[Point(0, 0)] * 3, crs=CRS)
    assert len(debris_sources(b, DebrisInput())) == 1


def test_edges_near_risky_building_are_penalised():
    p = DebrisInput(buffer_m=50, block_count=5, penalty_factor=3)
    g = apply_debris_risk(_graph(), _bld(1, 500, 20), p)  # beside edge A-B only
    ab = g["A"]["B"][0]
    assert ab["debris_sources"] == 1 and not ab["blocked"]
    assert np.isclose(ab["travel_time_adj"], ab["travel_time"] * 3)
    bc = g["B"]["C"][0]
    assert bc["debris_sources"] == 0 and bc["travel_time_adj"] == bc["travel_time"]


def test_many_sources_close_the_edge_and_routing_detours():
    p = DebrisInput(buffer_m=50, block_count=3, penalty_factor=3)
    g = apply_debris_risk(_graph(), _bld(3, 500, 0), p)
    assert g["A"]["B"][0]["blocked"] and np.isinf(g["A"]["B"][0]["travel_time_adj"])
    assert nx.shortest_path(g, "A", "C", weight="travel_time_adj") == ["A", "D", "C"]
    assert nx.shortest_path(_graph(), "A", "C", weight="travel_time") == ["A", "B", "C"]


def test_no_sources_changes_nothing_and_input_graph_untouched():
    g0 = _graph()
    g = apply_debris_risk(g0, _bld(1, 500, 0, prob=0.1))
    assert all(d["travel_time_adj"] == d["travel_time"] for *_, d in g.edges(data=True))
    assert "travel_time_adj" not in g0["A"]["B"][0]
