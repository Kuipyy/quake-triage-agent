import geopandas as gpd
import numpy as np

from quaketriage.planning.baselines import greedy_nearest, priority_first, random_dispatch
from quaketriage.planning.vrp import Problem


def _prob(trans, horizon, teams=1):
    n = len(trans) - 1
    cand = gpd.GeoDataFrame({"priority": np.linspace(0.9, 0.1, n), "estimated_persons": [5.0] * n,
                             "stop_node": range(n), "walk_m": [0.0] * n})
    return Problem(cand, np.asarray(trans, np.int64), horizon, teams, 0, 0)


# node 0 depot; building 1 is 100 s away, building 2 is 300 s away; 50 s service at each
T = [[0, 100, 300], [150, 0, 250], [350, 300, 0]]


def test_greedy_picks_nearest_first_and_respects_horizon():
    out = greedy_nearest(_prob(T, horizon=2000))
    assert [s["row"] for s in out["routes"][0]["stops"]] == [0, 1]
    assert out["routes"][0]["return_s"] <= 2000
    tight = greedy_nearest(_prob(T, horizon=300))  # only building 1 fits (100 + 150 back)
    assert tight["n_visited"] == 1 and tight["routes"][0]["stops"][0]["row"] == 0


def test_random_is_seeded_and_feasible():
    a, b = random_dispatch(_prob(T, 2000), 1), random_dispatch(_prob(T, 2000), 1)
    assert a["routes"] == b["routes"]
    assert all(r["return_s"] <= 2000 for r in random_dispatch(_prob(T, 2000), 3)["routes"])


def test_more_teams_never_visit_a_building_twice():
    out = greedy_nearest(_prob(T, 5000, teams=2))
    rows = [s["row"] for r in out["routes"] for s in r["stops"]]
    assert len(rows) == len(set(rows)) == 2


def test_priority_first_goes_to_highest_priority_even_if_far():
    out = priority_first(_prob(T, horizon=2000))
    # priorities are 0.9 (row 0, near) and 0.1 (row 1, far): near one is also highest here,
    # so flip them to check that distance is ignored
    prob = _prob(T, horizon=2000)
    prob.cand["priority"] = [0.1, 0.9]
    far_first = priority_first(prob)
    assert [s["row"] for s in far_first["routes"][0]["stops"]][0] == 1
    assert [s["row"] for s in out["routes"][0]["stops"]][0] == 0
