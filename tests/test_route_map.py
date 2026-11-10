import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point

from quaketriage.data.geo import get_project_crs
from quaketriage.ui.route_map import build_route_map

CRS = get_project_crs()
X0, Y0 = 250000.0, 4000000.0


def test_route_map_has_teams_depot_and_simulated_label():
    g = nx.MultiDiGraph(crs=CRS)
    for i in range(3):
        g.add_node(i, x=X0 + 500 * i, y=Y0)
    for a, b in ((0, 1), (1, 0), (1, 2), (2, 1)):
        geom = LineString([(X0 + 500 * a, Y0), (X0 + 500 * b, Y0)])
        g.add_edge(a, b, geometry=geom, travel_time_adj=50.0, blocked=(a, b) == (1, 2))
    b = gpd.GeoDataFrame({"priority": [0.8]}, geometry=[Point(X0 + 1000, Y0 + 30)], crs=CRS)
    stop = {"row": 0, "node": 1, "walk_m": 400.0, "priority": 0.8, "estimated_persons": 12.0,
            "arrival_s": 600}
    plan = {"routes": [{"team": 0, "stops": [stop], "return_s": 1800}, {"team": 1, "stops": [],
                                                                       "return_s": 0}],
            "n_visited": 1, "priority_collected": 0.8, "persons_reached": 12.0}
    html = build_route_map(g, plan, b, 0, (36.17, 36.2)).get_root().render()
    assert "team 1: 1 visits, 0.5 h, priority 0.80" in html and "team 2: 0 visits" in html
    assert "SIMULATED detector" in html and "team 1 (1 stops)" in html
    assert "closed roads" in html and "dash" in html.lower()
