import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point

from quaketriage.data.geo import get_project_crs
from quaketriage.planning.distances import travel_time_matrix
from quaketriage.planning.vrp import VrpInput, plan_routes, select_candidates

CRS = get_project_crs()
X0, Y0 = 250000.0, 4000000.0


def _line_graph(xs=(0, 1000, 2000, 3000)):
    """Nodes along x, 10 m/s two-way; node ids 0..n-1."""
    g = nx.MultiDiGraph(crs=CRS)
    for i, x in enumerate(xs):
        g.add_node(i, x=X0 + x, y=Y0)
    for i in range(len(xs) - 1):
        for a, b in ((i, i + 1), (i + 1, i)):
            geom = LineString([(g.nodes[a]["x"], Y0), (g.nodes[b]["x"], Y0)])
            g.add_edge(a, b, geometry=geom, travel_time=geom.length / 10,
                       travel_time_adj=geom.length / 10)
    return g


def _lonlat(x):
    return gpd.GeoSeries([Point(X0 + x, Y0)], crs=CRS).to_crs(4326).iloc[0]


def _buildings(xs, prios, persons=None):
    pts = [Point(X0 + x, Y0 + 5) for x in xs]
    return gpd.GeoDataFrame({"id": range(100, 100 + len(xs)), "priority": prios,
                             "estimated_persons": persons or [10.0] * len(xs)},
                            geometry=pts, crs=CRS)


def test_matrix_shortest_paths_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("quaketriage.planning.distances.ROOT", tmp_path)
    g = _line_graph()
    m = travel_time_matrix(g, [0, 3])
    assert np.allclose(m, [[0, 300], [300, 0]])
    assert list(tmp_path.rglob("tt_*.npy"))
    assert np.allclose(travel_time_matrix(g, [0, 3]), m)  # served from cache


def test_matrix_unreachable_is_inf_and_closed_edge_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr("quaketriage.planning.distances.ROOT", tmp_path)
    g = _line_graph()
    g[1][2][0]["travel_time_adj"] = np.inf
    m = travel_time_matrix(g, [0, 3])
    assert np.isinf(m[0, 1])


def test_candidates_are_top_n_and_nested():
    b = _buildings([0, 1, 2, 3], [0.1, 0.9, 0.5, 0.7])
    assert list(select_candidates(b, 2)["id"]) == [101, 103]
    assert set(select_candidates(b, 2)["id"]) <= set(select_candidates(b, 3)["id"])


def test_time_limit_forces_choice_of_higher_priority(tmp_path, monkeypatch):
    monkeypatch.setattr("quaketriage.planning.distances.ROOT", tmp_path)
    # depot at x=0; near low-priority at 1000 m (100 s), far high-priority at 3000 m (300 s).
    # service 30 min each; mission 1 h, one team -> only one visit fits.
    g = _line_graph()
    b = _buildings([1000, 3000], [0.2, 0.9])
    d = _lonlat(0)
    p = VrpInput(candidate_n=2, num_teams=1, mission_hours=1, intervention_minutes=30,
                 time_limit_s=1)
    out = plan_routes(g, b, (d.x, d.y), p)
    assert out["solved"] and out["n_visited"] == 1
    assert out["routes"][0]["stops"][0]["building_id"] == 101  # the 0.9 building
    assert out["SIMULATED_DETECTOR"] is True


def test_two_teams_visit_both_and_respect_horizon(tmp_path, monkeypatch):
    monkeypatch.setattr("quaketriage.planning.distances.ROOT", tmp_path)
    g = _line_graph()
    b = _buildings([1000, 3000], [0.2, 0.9])
    d = _lonlat(0)
    p = VrpInput(candidate_n=2, num_teams=2, mission_hours=1, time_limit_s=1)
    out = plan_routes(g, b, (d.x, d.y), p)
    assert out["n_visited"] == 2
    assert all(r["return_s"] <= 3600 for r in out["routes"])


def test_unreachable_building_is_counted_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr("quaketriage.planning.distances.ROOT", tmp_path)
    g = _line_graph()
    for a, b_ in ((1, 2), (2, 1)):
        g[a][b_][0]["travel_time_adj"] = np.inf  # cut the line between nodes 1 and 2
    b = _buildings([1000, 3000], [0.2, 0.9])
    d = _lonlat(0)
    out = plan_routes(g, b, (d.x, d.y), VrpInput(candidate_n=2, num_teams=1, time_limit_s=1))
    assert out["n_unreachable"] == 1 and out["n_visited"] == 1
